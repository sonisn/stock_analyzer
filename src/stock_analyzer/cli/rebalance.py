"""Portfolio rebalance pipeline.

Run: python -m stock_analyzer.cli.rebalance

Extends the discover pipeline with three new steps:
  - holdings_fetch    SnapTrade positions + cash balance
  - holdings_data     fundamentals/technicals/risk factors for held tickers
  - review_holdings   Sonnet per holding → HOLD / TRIM / SELL verdict
  - rebalance         Opus combines verdicts + discover picks + cash into action list

User-configured behavior (locked in via conversation):
  - Sizing: self-fund from SELL/TRIM proceeds AND add available cash
  - Aggressiveness: AGGRESSIVE churn — actively recommend SELLs where a
    materially better alternative exists, not only when thesis is broken

Output: email with HTML body (charts for new picks inline) + PDF attachment.
Run history shares the discover.db SQLite file.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Any

from agno.db.sqlite import SqliteDb
from agno.workflow import Parallel, Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from dotenv import load_dotenv

from ..config import Settings
from ..data.brokerage import fetch_portfolio_holdings, fetch_total_cash
from ..data.chart_img import fetch_charts
from ..data.finnhub import batch_finnhub_signals
from ..data.fundamentals import batch_fundamentals
from ..data.insider_selling import insider_selling_mentions
from ..data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ..data.share_trades import batch_share_trade_data
from ..data.technical_indicators import batch_technicals
from ..data.transactions import fetch_transaction_history, to_tax_payloads
from ..data.transcripts import batch_transcript_snippets
from ..discover.peers import batch_peer_comparison
from ..discover.persistence import (
    connect,
    fetch_recent_holdings_history,
    insert_candidate,
    insert_holdings_review,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from ..discover.rebalancer import Rebalancer
from ..discover.report import (
    Section,
    build_sections,
    parse_confidence,
    parse_rebalance_status,
    parse_verdict,
    print_terminal_summary,
    render_html_email,
    render_pdf,
)
from ..discover.reviewer import Reviewer, review_batch
from ..logging import current_log_file, get_logger
from ..preflight import PreflightError, preflight
from ..reporting.smtp import SmtpServer
from .discover import (
    DiscoverPipeline,
    _QUARTERLY_MDA_CHARS,
    _RISK_FACTORS_CHARS,
    _TRANSCRIPT_CHARS,
    _trim,
)

logger = get_logger(__name__)


def _save_local_pdf(pdf_bytes: bytes, filename: str) -> Path:
    """Persist the PDF to ~/.stock_analyzer/reports/ so a missed email
    never costs the user the report. Override via REPORTS_DIR env."""
    reports_dir = Path(
        os.path.expanduser(os.getenv("REPORTS_DIR", "~/.stock_analyzer/reports"))
    )
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / filename
    path.write_bytes(pdf_bytes)
    return path


def _build_history_block(db_path: str, *, n_runs: int = 3) -> str:
    """Reach into the discover DB and produce a compact `Previous decisions`
    block for the rebalancer prompt. Per-holding format:

        AAPL: HOLD-8 (2026-05-05) -> HOLD-7 (2026-05-10) -> today

    Oldest first so the LLM sees chronological drift left-to-right. Returns
    an empty string if no history exists (first run, fresh DB, or every
    prior run was a discover-only run).
    """
    try:
        with connect(db_path) as conn:
            history = fetch_recent_holdings_history(conn, n_runs=n_runs)
    except Exception as e:
        logger.warning("history fetch failed (%s) — proceeding without it", e)
        return ""
    if not history:
        return ""
    lines: list[str] = []
    for ticker in sorted(history.keys()):
        entries = history[ticker]
        parts: list[str] = []
        for e in entries:
            verdict = e.get("verdict") or "?"
            conf = e.get("confidence")
            run_at = (e.get("run_at") or "")[:10]
            label = f"{verdict}-{conf}" if conf is not None else verdict
            parts.append(f"{label} ({run_at})")
        parts.append("today")
        lines.append(f"{ticker}: {' -> '.join(parts)}")
    return "\n".join(lines)


def _log_full_analysis(
    *,
    delivered: bool,
    delivery_error: str | None,
    local_pdf_path: str,
    rebalance_text: str,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    holdings_reviews: dict[str, Any],
) -> None:
    """Dump every analyst-produced section to the logger so the user can
    recover the full report from the log file when email fails."""
    from ..discover.schemas import HoldingReview
    bar = "=" * 70
    if not delivered:
        logger.error(
            "%s\nEMAIL NOT DELIVERED — full analysis follows in this log.\n"
            "Reason: %s\nPDF: %s\n%s",
            bar,
            delivery_error or "EMAIL_TO not configured",
            local_pdf_path,
            bar,
        )
    logger.info("%s\nREBALANCE PLAN\n%s\n%s", bar, bar, rebalance_text)
    logger.info("%s\nRANKER — discover picks\n%s\n%s", bar, bar, ranker_text)
    logger.info("%s\nRED TEAM — bear cases\n%s\n%s", bar, bar, redteam_text)
    logger.info("%s\nSIZER — allocation\n%s\n%s", bar, bar, sizer_text)
    for ticker in sorted(holdings_reviews):
        review = holdings_reviews.get(ticker)
        text = (
            review.full_text if isinstance(review, HoldingReview)
            else (review or "(review unavailable)")
        )
        logger.info("%s\nHOLDING REVIEW — %s\n%s\n%s", bar, ticker, bar, text)


def _aggregate_positions(
    holdings: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Collapse holdings-across-accounts into one position per ticker."""
    agg: dict[str, dict[str, float]] = {}
    for items in holdings.values():
        for h in items:
            ticker = h.get("ticker")
            units = h.get("units") or 0
            avg = h.get("average_purchase_price") or 0
            if not ticker or not units:
                continue
            cur = agg.setdefault(ticker, {"units": 0.0, "cost": 0.0})
            cur["units"] += float(units)
            cur["cost"] += float(units) * float(avg)
    out: dict[str, dict[str, float]] = {}
    for ticker, v in agg.items():
        if v["units"]:
            out[ticker] = {
                "units": v["units"],
                "avg_buy_price": v["cost"] / v["units"],
                "cost_basis": v["cost"],
            }
    return out


class RebalancePipeline(DiscoverPipeline):
    """Discovery + per-holding review + Opus rebalance plan delivery."""

    # --- new step executors -------------------------------------------------

    def step_holdings_fetch(self, step_input: StepInput) -> StepOutput:
        try:
            holdings = fetch_portfolio_holdings()
        except Exception as e:
            raise RuntimeError(f"Could not fetch SnapTrade holdings: {e}")
        positions = _aggregate_positions(holdings)
        if not positions:
            raise RuntimeError(
                "No SnapTrade positions found — rebalance requires existing holdings"
            )
        cash = fetch_total_cash()
        self.state["holdings_positions"] = positions
        self.state["cash_balance"] = cash
        self.state["holdings_tickers"] = list(positions.keys())
        cash_str = f"${cash:,.0f}" if cash is not None else "unknown"
        return StepOutput(
            content=f"Holdings: {len(positions)} positions; cash {cash_str}"
        )

    def step_holdings_data(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["holdings_tickers"]
        self.state["holdings_fundamentals"] = batch_fundamentals(tickers)
        self.state["holdings_technicals"] = batch_technicals(tickers)
        self.state["holdings_risk_factors"] = batch_risk_factors(tickers)
        # Forward-narrative + peers + transcript for holdings — same data the
        # discover pipeline fetches for survivors. Reviewer uses these to
        # judge whether the current holding still ranks against peers.
        self.state["holdings_quarterly_mda"] = batch_quarterly_mda(tickers)
        target_meta = {
            t: {
                "name": (self.state["holdings_fundamentals"].get(t) or {}).get("name"),
                "sector": (self.state["holdings_fundamentals"].get(t) or {}).get("sector"),
            }
            for t in tickers
        }
        self.state["holdings_peers"] = batch_peer_comparison(tickers, target_meta)
        self.state["holdings_transcripts"] = batch_transcript_snippets(tickers)
        return StepOutput(
            content=(
                f"Holdings enrichment: fundamentals={len(self.state['holdings_fundamentals'])}, "
                f"10-Q MD&A={len(self.state['holdings_quarterly_mda'])}, "
                f"peers={len(self.state['holdings_peers'])}, "
                f"transcripts={len(self.state['holdings_transcripts'])}"
            )
        )

    def step_transaction_history(self, step_input: StepInput) -> StepOutput:
        """Pull 3yr of SnapTrade activities and build per-ticker tax lot summaries.
        Runs independently of survivors — relies only on SnapTrade auth."""
        summaries = fetch_transaction_history(years_back=3)
        self.state["tax_lots"] = to_tax_payloads(summaries)
        n_lots = sum(s.get("lot_count", 0) for s in self.state["tax_lots"].values())
        return StepOutput(
            content=(
                f"Tax lots: {len(self.state['tax_lots'])} tickers, "
                f"{n_lots} total lots over 3yr lookback"
            )
        )

    def step_insider_selling(self, step_input: StepInput) -> StepOutput:
        """Override: include holdings tickers so reviewer sees selling on them too."""
        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["insider_selling"] = {}
            return StepOutput(content="insider_selling: no tickers; skipping")
        self.state["insider_selling"] = insider_selling_mentions(tickers, days=14)
        return StepOutput(
            content=f"Insider selling: {len(self.state['insider_selling'])} flagged"
        )

    def step_share_trades(self, step_input: StepInput) -> StepOutput:
        """Override: fetch insider/institutional data for both survivors AND holdings."""
        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["share_trades"] = {}
            return StepOutput(content="share_trades: no tickers; skipping")
        self.state["share_trades"] = batch_share_trade_data(list(tickers))
        return StepOutput(
            content=f"Share trades fetched for {len(self.state['share_trades'])}/{len(tickers)}"
        )

    def step_finnhub_signals(self, step_input: StepInput) -> StepOutput:
        """Earnings surprise + recommendation trend + price targets +
        Form-4 insider activity for survivors AND current holdings."""
        tickers = set(self.state.get("survivor_tickers") or [])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        if not tickers:
            self.state["finnhub_signals"] = {}
            return StepOutput(content="finnhub_signals: no tickers; skipping")
        self.state["finnhub_signals"] = batch_finnhub_signals(list(tickers))
        n = sum(1 for v in self.state["finnhub_signals"].values() if v)
        return StepOutput(
            content=f"Finnhub signals: {n}/{len(tickers)} tickers covered"
        )

    def step_review_holdings(self, step_input: StepInput) -> StepOutput:
        positions = self.state["holdings_positions"]
        fund = self.state["holdings_fundamentals"]
        tech = self.state["holdings_technicals"]
        rfs = self.state["holdings_risk_factors"]
        selling = self.state.get("insider_selling", {})
        finnhub_signals = self.state.get("finnhub_signals", {})
        eps_revisions = self.state.get("eps_revisions", {})

        payloads: dict[str, dict[str, Any]] = {}
        for ticker, pos in positions.items():
            t = tech.get(ticker) or {}
            current = t.get("price")
            avg = pos["avg_buy_price"]
            units = pos["units"]
            pnl = None
            pnl_pct = None
            if current and avg:
                pnl = (current - avg) * units
                pnl_pct = (current - avg) / avg * 100
            fh = finnhub_signals.get(ticker) or {}
            insider_activity: Any = fh.get("insider_activity") or {
                "mention_count": selling.get(ticker, 0)
            }
            payloads[ticker] = {
                "position": {
                    "units": units,
                    "avg_buy_price": avg,
                    "current_price": current,
                    "cost_basis": pos["cost_basis"],
                    "unrealized_pnl": pnl,
                    "unrealized_pnl_pct": pnl_pct,
                },
                "fundamentals": fund.get(ticker) or {},
                "technicals": t,
                "insider_activity": insider_activity,
                "earnings_surprise_history": fh.get("earnings_surprise") or [],
                "recommendation_trend": fh.get("recommendation_trend") or [],
                "analyst_price_targets": fh.get("price_targets") or {},
                "eps_revisions": eps_revisions.get(ticker) or {},
                "share_trades": self.state.get("share_trades", {}).get(ticker),
                "risk_factors_10k": _trim(
                    (rfs.get(ticker) or {}).get("risk_factors"),
                    _RISK_FACTORS_CHARS,
                ),
                "quarterly_mda": _trim(
                    (self.state.get("holdings_quarterly_mda", {}).get(ticker) or {}).get("mda"),
                    _QUARTERLY_MDA_CHARS,
                ),
                "peers": self.state.get("holdings_peers", {}).get(ticker),
                "earnings_transcript": _trim(
                    (self.state.get("holdings_transcripts", {}).get(ticker) or {}).get("snippet"),
                    _TRANSCRIPT_CHARS,
                ),
                "tax_lots": self.state.get("tax_lots", {}).get(ticker),
            }

        reviewer = Reviewer("claude", self.settings.discover_sonnet_model)
        self.state["holdings_reviews"] = review_batch(reviewer, payloads)
        return StepOutput(
            content=f"Reviewed {len(self.state['holdings_reviews'])} holdings"
        )

    def step_rebalance(self, step_input: StepInput) -> StepOutput:
        history_block = _build_history_block(self.settings.discover_db_path)
        if history_block:
            logger.info(
                "Cross-run context: %d holdings have prior decisions",
                history_block.count("\n"),
            )
        rebalancer = Rebalancer("claude", self.settings.discover_opus_model)
        plan = rebalancer.decide(
            self.state["holdings_reviews"],
            self.state["ranker_text"],
            self.state.get("cash_balance"),
            self.state.get("macro_summary", ""),
            aggressiveness=self.settings.discover_rebalance_aggressiveness,
            history_block=history_block,
            market_themes_block=self.state.get("market_themes_block", ""),
        )
        self.state["rebalance_plan"] = plan
        # `rebalance_text` is the prose rendering, kept under the same key
        # so the PDF/email layer and the log-dump fallback need no change.
        self.state["rebalance_text"] = plan.full_text
        return StepOutput(
            content=(
                f"Rebalance plan generated "
                f"(status={plan.status}, "
                f"aggressiveness={plan.aggressiveness_applied}, "
                f"actions={len(plan.actions)})"
            )
        )

    def step_persist_and_email_rebalance(self, step_input: StepInput) -> StepOutput:
        # Persist run + candidates + scorecards + picks + outputs.
        with connect(self.settings.discover_db_path) as conn:
            run_id = insert_run(
                conn,
                universe_size=len(self.state["candidates"]),
                survivors=len(self.state["survivors"]),
                picks=len(self.state["picks"]),
                opus_model=self.settings.discover_opus_model,
                sonnet_model=self.settings.discover_sonnet_model,
                cash_budget=self.state.get("cash_balance"),
                kind="rebalance",
            )
            for c in self.state["candidates"]:
                insert_candidate(
                    conn,
                    run_id,
                    c["ticker"],
                    passed_filter=c["passed_filter"],
                    fail_reasons=c["fail_reasons"],
                    score=c["score"],
                    score_components=c["score_components"],
                    score_breakdown=c["score_breakdown"],
                    sources=c["sources"],
                    conviction=c["conviction"],
                    sector=c["sector"],
                    price=c["price"],
                )
            for ticker, report in self.state["analyses"].items():
                analyst_text = getattr(report, "full_text", None) or (
                    report if isinstance(report, str) else ""
                )
                insert_scorecard(conn, run_id, ticker, analyst_text)
            for ticker, review in self.state.get("holdings_reviews", {}).items():
                if not review:
                    continue
                # `review` is a HoldingReview (Phase 4a). Fall back gracefully
                # if a legacy str leaks through.
                review_text = getattr(review, "full_text", None) or (
                    review if isinstance(review, str) else ""
                )
                insert_holdings_review(
                    conn,
                    run_id,
                    ticker,
                    verdict=parse_verdict(review),
                    confidence=parse_confidence(review),
                    review_text=review_text,
                )
            for rank, ticker, _ in self.state["picks"]:
                insert_pick(
                    conn,
                    run_id,
                    rank=rank,
                    ticker=ticker,
                    ranker_text=self.state["ranker_text"],
                    bear_case_text=self.state["redteam_text"],
                    allocation_text=self.state["sizer_text"],
                )
            plan = self.state.get("rebalance_plan")
            # Use .get() with empty-string defaults for the rebalancer's
            # outputs so a failed rebalance step (e.g. LLM error after
            # max retries) doesn't compound into a KeyError that
            # tanks persistence too.
            insert_run_outputs(
                conn,
                run_id,
                ranker_full=self.state.get("ranker_text", "") or "",
                redteam_full=self.state.get("redteam_text", "") or "",
                sizer_full=self.state.get("sizer_text", "") or "",
                holdings_summary=self.state.get("holdings_summary", "") or "",
                rebalance_text=self.state.get("rebalance_text", "") or "",
                dashboard_data=plan.model_dump(mode="json") if plan else None,
            )

        # Charts for the BUY candidates (discover picks).
        pick_tickers = [t for _, t, _ in self.state["picks"]]
        charts: dict[str, bytes] = {}
        try:
            charts = fetch_charts(pick_tickers)
        except Exception as e:
            logger.warning("Chart fetch failed (%s) — report will omit charts", e)
        chart_cids = {t: f"chart-{t.replace('.', '-')}" for t in charts}

        sections = _build_rebalance_sections(
            rebalance_text=self.state["rebalance_text"],
            holdings_reviews=self.state["holdings_reviews"],
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
            candidates=self.state["candidates"],
            cash_balance=self.state.get("cash_balance"),
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            holdings_positions=self.state.get("holdings_positions", {}),
            holdings_technicals=self.state.get("holdings_technicals", {}),
            holdings_fundamentals=self.state.get("holdings_fundamentals", {}),
            track_record_block=self.state.get("track_record_block", ""),
            rebalance_plan=self.state.get("rebalance_plan"),
            market_themes=self.state.get("market_themes"),
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        today = date.today()
        subject = f"Portfolio Rebalance — {today.strftime('%b-%d')}"
        pdf_filename = f"rebalance-{today.isoformat()}.pdf"

        # Save PDF locally BEFORE the email attempt so a delivery failure
        # (SMTP outage, wrong creds, etc.) never costs the user the report.
        local_pdf_path = _save_local_pdf(pdf_bytes, pdf_filename)
        logger.info("Saved rebalance PDF locally: %s", local_pdf_path)

        delivered = False
        delivery_error: str | None = None
        if self.settings.email_to:
            try:
                SmtpServer().send_email(
                    self.settings.email_to,
                    subject,
                    html_body,
                    content_type="html",
                    inline_images={
                        chart_cids[t]: data for t, data in charts.items()
                    } or None,
                    attachments=[(pdf_filename, pdf_bytes, "pdf")],
                )
                delivered = True
                logger.info(
                    "Sent rebalance email to %s", self.settings.email_to
                )
            except Exception as e:
                delivery_error = str(e)
                logger.error("Email delivery failed: %s", e)
        else:
            logger.warning("EMAIL_TO not set; skipping email")

        # Always dump the full analysis to the log so the user can recover
        # every section — even when email is offline or wasn't configured.
        _log_full_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            rebalance_text=self.state["rebalance_text"],
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
            holdings_reviews=self.state["holdings_reviews"],
        )

        print_terminal_summary(self.state["ranker_text"], self.state["sizer_text"])
        print("\n" + "=" * 60)
        print("REBALANCE PLAN")
        print("=" * 60)
        print(self.state["rebalance_text"])
        print(f"\nPDF saved: {local_pdf_path}")
        log_path = current_log_file()
        if log_path:
            print(f"Log file:  {log_path}")

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        self.state["local_pdf_path"] = str(local_pdf_path)
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=(
                f"Rebalance run #{run_id} {status}; PDF {len(pdf_bytes)} bytes "
                f"(saved to {local_pdf_path})"
            )
        )

    # --- workflow assembly -------------------------------------------------

    def build_workflow(self) -> Workflow:
        db_path = Path(os.path.expanduser(self.settings.discover_db_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return Workflow(
            name="Portfolio Rebalance",
            description=(
                "Discover new picks + review current holdings + emit "
                "aggressive rebalance plan"
            ),
            db=SqliteDb(
                db_file=str(db_path), session_table="workflow_session"
            ),
            steps=[
                Step(name="universe", executor=self.step_universe),
                Parallel(
                    Step(name="fundamentals", executor=self.step_fundamentals),
                    Step(name="technicals", executor=self.step_technicals),
                    Step(name="sector_rotation", executor=self.step_sector_rotation),
                    Step(name="macro_regime", executor=self.step_macro_regime),
                    Step(name="track_record", executor=self.step_track_record),
                    Step(name="holdings_fetch", executor=self.step_holdings_fetch),
                    Step(
                        name="transaction_history",
                        executor=self.step_transaction_history,
                    ),
                    name="market_data",
                ),
                # Market themes after market_data (depends on sector_rotation +
                # macro_regime from inside the parallel block).
                Step(name="market_themes", executor=self.step_market_themes),
                Step(name="screen", executor=self.step_screen),
                Parallel(
                    Step(name="risk_factors", executor=self.step_risk_factors),
                    Step(name="quarterly_mda", executor=self.step_quarterly_mda),
                    Step(name="news", executor=self.step_news),
                    Step(name="earnings", executor=self.step_earnings),
                    Step(name="insider_selling", executor=self.step_insider_selling),
                    Step(name="share_trades", executor=self.step_share_trades),
                    Step(name="peer_comparison", executor=self.step_peer_comparison),
                    Step(name="earnings_transcripts", executor=self.step_earnings_transcripts),
                    Step(name="finnhub_signals", executor=self.step_finnhub_signals),
                    Step(name="eps_revisions", executor=self.step_eps_revisions),
                    Step(name="holdings_data", executor=self.step_holdings_data),
                    name="enrichment",
                ),
                Step(name="analyst", executor=self.step_analyst),
                Step(name="holdings", executor=self.step_holdings),
                Step(name="ranker", executor=self.step_ranker),
                Step(name="redteam", executor=self.step_redteam),
                Step(name="sizer", executor=self.step_sizer),
                Step(name="review_holdings", executor=self.step_review_holdings),
                Step(name="rebalance", executor=self.step_rebalance),
                Step(
                    name="persist_and_email_rebalance",
                    executor=self.step_persist_and_email_rebalance,
                ),
            ],
        )


def _build_rebalance_sections(
    *,
    rebalance_text: str,
    holdings_reviews: dict[str, Any],
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    cash_balance: float | None,
    macro_summary: str,
    sector_rotation: dict[str, Any] | None,
    holdings_positions: dict[str, dict[str, Any]],
    holdings_technicals: dict[str, dict[str, Any]],
    holdings_fundamentals: dict[str, dict[str, Any]],
    track_record_block: str = "",
    rebalance_plan: object = None,
    market_themes: object = None,
) -> list[Section]:
    """Rebalance-specific layout — status banner + metrics + dashboard +
    sector pie at the top, then the LLM's plan + per-holding reviews +
    discover-picks appendix.

    `rebalance_plan` is the structured RebalancePlan from the LLM (Phase 3);
    `rebalance_text` is the prose rendering (plan.full_text). The plan is
    used to determine status without regex; the prose is what we render."""
    today = date.today().isoformat()

    # ---- Parse status + collect dashboard rows ----------------------------
    # Prefer the structured plan; fall back to regex on text only when not
    # available (legacy or partial runs).
    status = parse_rebalance_status(rebalance_plan or rebalance_text)
    status_label = (
        "STATUS: NO ACTION RECOMMENDED" if status == "NO_ACTION"
        else "STATUS: ACTION RECOMMENDED" if status == "ACTION"
        else "STATUS: REVIEW REQUIRED"
    )

    dashboard_rows: list[dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0
    sector_value: dict[str, float] = {}
    for ticker in sorted(holdings_positions.keys()):
        pos = holdings_positions[ticker]
        tech = holdings_technicals.get(ticker, {})
        fund = holdings_fundamentals.get(ticker, {})
        review = holdings_reviews.get(ticker) or ""
        current = tech.get("price")
        units = pos.get("units", 0)
        cost = pos.get("cost_basis", 0)
        value = (current or 0) * units
        total_value += value
        total_cost += cost
        pnl_pct = ((value - cost) / cost * 100) if cost else None
        sector = fund.get("sector") or "Unknown"
        if value > 0:
            sector_value[sector] = sector_value.get(sector, 0) + value
        dashboard_rows.append({
            "ticker": ticker,
            "verdict": parse_verdict(review),
            "confidence": parse_confidence(review),
            "pnl_pct": pnl_pct,
            "sector": sector,
            "note": "",
        })

    total_pnl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost else None

    # ---- Build sections ---------------------------------------------------
    sections: list[Section] = [
        Section(kind="heading", text=f"Portfolio Rebalance — {today}", level=1),
        Section(kind="status_banner", text=status_label, status=status),
    ]

    metrics: list[tuple[str, str]] = [
        ("Holdings", f"{len(holdings_positions)}"),
        ("Portfolio value", f"${total_value:,.0f}" if total_value else "—"),
        ("Total P/L", f"{total_pnl_pct:+.1f}%" if total_pnl_pct is not None else "—"),
        ("Cash", f"${cash_balance:,.0f}" if cash_balance is not None else "—"),
    ]
    sections.append(Section(kind="metric_strip", metrics=metrics))

    if dashboard_rows:
        sections.append(Section(kind="heading", text="Holdings dashboard", level=2))
        sections.append(
            Section(kind="holdings_dashboard", holdings=dashboard_rows)
        )

    if sector_value:
        pie_data = sorted(sector_value.items(), key=lambda x: x[1], reverse=True)
        sections.append(Section(kind="heading", text="Sector allocation", level=2))
        sections.append(Section(kind="sector_pie", pie_data=pie_data))

    if track_record_block:
        sections.append(Section(kind="heading", text="Track record", level=2))
        sections.append(Section(kind="preformatted", text=track_record_block))

    from ..discover.schemas import MarketThemes
    if isinstance(market_themes, MarketThemes) and market_themes.themes:
        sections.append(Section(kind="heading", text="Current market themes", level=2))
        sections.append(Section(
            kind="market_themes_panel",
            data={
                "themes": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "strength": t.strength,
                        "trending": t.trending,
                        "member_tickers": list(t.member_tickers),
                    }
                    for t in market_themes.themes
                ],
            },
        ))

    if macro_summary:
        sections.append(Section(kind="heading", text="Macro regime", level=2))
        sections.append(Section(kind="blockquote", text=macro_summary))

    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Rebalance plan (action list)", level=1))
    # Structured action table when status=ACTION — one styled row per
    # action with a SELL/TRIM/ADD/BUY badge. NO_ACTION runs have no
    # actions; skip the table entirely (status banner already conveys
    # the verdict).
    from ..discover.rebalance_schema import RebalancePlan
    plan = rebalance_plan if isinstance(rebalance_plan, RebalancePlan) else None
    if plan and plan.actions:
        sections.append(Section(
            kind="rebalance_action_table",
            data={
                "actions": [
                    {"action": a.action, "ticker": a.ticker, "sizing": a.sizing}
                    for a in plan.actions
                ],
                "summary": plan.summary,
            },
        ))
    sections.append(Section(kind="preformatted", text=rebalance_text))

    sections.append(Section(kind="page_break"))
    sections.append(Section(kind="heading", text="Per-holding reviews", level=1))
    from ..discover.schemas import HoldingReview
    for ticker in sorted(holdings_reviews.keys()):
        review = holdings_reviews[ticker]
        if isinstance(review, HoldingReview):
            # Structured card — pills, labeled sections, tax lot list,
            # wash-sale callout. Renders much nicer than the monospace
            # preformatted dump.
            sections.append(Section(
                kind="holding_review_card",
                data={
                    "ticker": ticker,
                    "verdict": review.verdict,
                    "confidence": review.confidence,
                    "trim_pct": review.trim_pct,
                    "position_context": review.position_context,
                    "forward_outlook": review.forward_outlook,
                    "reasoning": review.reasoning,
                    "tax_lot_plan": list(review.tax_lot_plan),
                    "what_would_change_mind": review.what_would_change_mind,
                    "wash_sale_notice": review.wash_sale_notice,
                },
            ))
        else:
            # Legacy free-text path — keep the old layout.
            text = review or ""
            sections.append(Section(kind="heading", text=ticker, level=2))
            sections.append(Section(kind="preformatted", text=text))

    # Discover picks as appendix — drop their own H1 + summary since we have ours.
    discover_sections = build_sections(
        ranker_text=ranker_text,
        redteam_text=redteam_text,
        sizer_text=sizer_text,
        candidates=candidates,
        universe_size=len(candidates),
        holdings_summary="",
        macro_summary="",
        sector_rotation=sector_rotation,
    )
    sections.append(Section(kind="page_break"))
    sections.append(
        Section(kind="heading", text="Discover picks (input to rebalancer)", level=1)
    )
    sections.extend(discover_sections[2:])
    return sections


def run() -> None:
    load_dotenv()
    settings = Settings.from_env()
    try:
        preflight(
            settings,
            needs_llm=True,
            needs_brokerage=True,
            needs_finnhub=bool(settings.finnhub_api_key),
            needs_email=bool(settings.email_to),
        )
    except PreflightError as e:
        logger.error("%s", e)
        raise SystemExit(2) from e
    pipeline = RebalancePipeline(settings)
    workflow = pipeline.build_workflow()
    logger.info("=== Portfolio rebalance pipeline starting ===")
    workflow.print_response(input="rebalance", stream=True)
    if pipeline.state.get("run_id"):
        print(
            f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}"
        )


def main() -> None:
    run()


if __name__ == "__main__":
    main()
