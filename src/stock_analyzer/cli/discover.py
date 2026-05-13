"""Stock discovery pipeline — Agno Workflow version.

Run via:   python -m stock_analyzer.cli.discover

Declarative shape:
  universe
  ├ Parallel(fundamentals, technicals)
  screen
  ├ Parallel(risk_factors, news)
  analyst (Sonnet, parallel fan-out inside step)
  holdings
  ranker (Opus + extended thinking)
  redteam (Opus)
  sizer (Opus)
  persist_and_report

Workflow's SqliteDb logs every run (per-step input/output/timing) into the
SAME discover.db file we use for domain tables. Single source of truth.

State is shared across steps via a DiscoverPipeline instance — each step is a
bound method that reads/writes self.state. Cleaner than threading dicts
through StepOutput.content.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import yfinance as yf
from agno.db.sqlite import SqliteDb
from agno.workflow import Parallel, Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from dotenv import load_dotenv

from ..config import Settings
from ..data.brokerage import fetch_portfolio_holdings
from ..data.chart_img import fetch_charts
from ..data.earnings_calendar import batch_earnings_flags
from ..data.fred_macro import fetch_regime_data, regime_summary_text
from ..data.fundamentals import batch_fundamentals
from ..data.finnhub import batch_finnhub_signals
from ..data.insider_selling import insider_selling_mentions
from ..data.sec_edgar import batch_quarterly_mda, batch_risk_factors
from ..data.sector_rotation import sector_bias, sector_rotation_summary
from ..data.share_trades import batch_share_trade_data
from ..data.technical_indicators import batch_technicals
from ..data.transcripts import batch_transcript_snippets
from ..discover.analyst import Analyst, analyze_batch
from ..discover.peers import batch_peer_comparison
from ..discover.persistence import (
    connect,
    insert_candidate,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from ..discover.ranker import Ranker
from ..discover.redteam import RedTeam
from ..discover.track_record import (
    format_track_record_block,
    format_track_record_summary,
    measure_track_record,
)
from ..discover.report import (
    build_sections,
    parse_picks,
    print_terminal_summary,
    render_html_email,
    render_pdf,
)
from ..discover.screen import passes_hard_filter, score_candidate
from ..discover.sizer import Sizer
from ..discover.universe import build_universe
from ..logging import current_log_file, get_logger
from ..preflight import PreflightError, preflight
from ..reporting.smtp import SmtpServer

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


def _log_discover_analysis(
    *,
    delivered: bool,
    delivery_error: str | None,
    local_pdf_path: str,
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
) -> None:
    """Dump every analyst-produced section to the logger so the user can
    recover the full report from the log file when email fails."""
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
    logger.info("%s\nRANKER — discover picks\n%s\n%s", bar, bar, ranker_text)
    logger.info("%s\nRED TEAM — bear cases\n%s\n%s", bar, bar, redteam_text)
    logger.info("%s\nSIZER — allocation\n%s\n%s", bar, bar, sizer_text)


MAX_CANDIDATES_FOR_LLM = 25

# Per-field char caps applied to analyst/reviewer payloads to keep each call
# well under Sonnet's 30k input-tokens/min rate limit. 10-K text is mostly
# boilerplate; MD&A and transcripts retain most signal at these sizes.
_RISK_FACTORS_CHARS = 3500
_QUARTERLY_MDA_CHARS = 4000
_TRANSCRIPT_CHARS = 2500


def _trim(text: str | None, max_chars: int) -> str | None:
    if not text:
        return text
    return text[:max_chars]


# --- small helpers (used by step executors) ----------------------------------


def _fetch_news(ticker: str, limit: int = 3) -> list[dict[str, Any]]:
    try:
        items = yf.Ticker(ticker).news or []
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for it in items[:limit]:
        title = it.get("title") or (it.get("content") or {}).get("title")
        link = it.get("link") or (
            (it.get("content") or {}).get("canonicalUrl") or {}
        ).get("url")
        if title:
            out.append({"title": title, "link": link})
    return out


def _batch_news(tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    results: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        for ticker, news in zip(tickers, ex.map(_fetch_news, tickers)):
            results[ticker] = news
    return results


def _holdings_summary(holdings: dict[str, list[dict[str, Any]]]) -> str:
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
    if not agg:
        return ""
    lines: list[str] = []
    for ticker, v in sorted(agg.items()):
        avg = v["cost"] / v["units"] if v["units"] else 0
        lines.append(f"  - {ticker}: {v['units']:.0f} shares @ avg ${avg:,.2f}")
    return "\n".join(lines)


# --- pipeline ----------------------------------------------------------------


class DiscoverPipeline:
    """Holds shared state across Workflow steps.

    Each `step_*` method is bound to this instance, so steps read/write
    self.state instead of round-tripping data through StepOutput content.
    Fatal conditions (empty universe, no survivors) raise RuntimeError —
    the Workflow aborts cleanly and the run shows as failed in workflow_session.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state: dict[str, Any] = {}

    # --- step executors ------------------------------------------------

    def step_universe(self, step_input: StepInput) -> StepOutput:
        universe = build_universe(watchlist=self.settings.discover_watchlist)
        if not universe:
            raise RuntimeError(
                "Universe empty — no candidates from insider/billionaire/watchlist. "
                "Set DISCOVER_WATCHLIST or check TAVILY_API_KEY."
            )
        self.state["universe"] = universe
        self.state["tickers"] = list(universe.keys())
        return StepOutput(content=f"Universe: {len(universe)} candidates")

    def step_fundamentals(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["tickers"]
        self.state["fundamentals"] = batch_fundamentals(tickers)
        return StepOutput(
            content=f"Fundamentals: {len(self.state['fundamentals'])}/{len(tickers)}"
        )

    def step_technicals(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["tickers"]
        self.state["technicals"] = batch_technicals(tickers)
        return StepOutput(
            content=f"Technicals: {len(self.state['technicals'])}/{len(tickers)}"
        )

    def step_sector_rotation(self, step_input: StepInput) -> StepOutput:
        self.state["sector_rotation"] = sector_rotation_summary(months=6)
        leaders = self.state["sector_rotation"].get("leaders", [])
        laggards = self.state["sector_rotation"].get("laggards", [])
        return StepOutput(
            content=f"Sector leaders (6mo): {leaders}; laggards: {laggards}"
        )

    def step_macro_regime(self, step_input: StepInput) -> StepOutput:
        data = fetch_regime_data(self.settings.fred_api_key)
        self.state["macro_data"] = data
        self.state["macro_summary"] = regime_summary_text(data)
        # Truncate for terminal preview.
        return StepOutput(content=self.state["macro_summary"][:200])

    def step_track_record(self, step_input: StepInput) -> StepOutput:
        record = measure_track_record(self.settings.discover_db_path)
        self.state["track_record"] = record
        self.state["track_record_summary"] = format_track_record_summary(record)
        self.state["track_record_block"] = format_track_record_block(record)
        return StepOutput(content=self.state["track_record_summary"])

    def step_screen(self, step_input: StepInput) -> StepOutput:
        universe = self.state["universe"]
        fundamentals = self.state["fundamentals"]
        technicals = self.state["technicals"]

        candidates: list[dict[str, Any]] = []
        for ticker in self.state["tickers"]:
            f = fundamentals.get(ticker)
            t = technicals.get(ticker)
            u = universe[ticker]
            passes, reasons = passes_hard_filter(f, t)
            cand: dict[str, Any] = {
                "ticker": ticker,
                "passed_filter": passes,
                "fail_reasons": reasons,
                "sources": u["sources"],
                "conviction": u["conviction"],
                "sector": (f or {}).get("sector"),
                "price": (t or {}).get("price"),
                "score": None,
                "score_components": None,
                "score_breakdown": None,
            }
            if passes and f and t:
                scored = score_candidate(f, t, u)
                cand["score"] = scored["score"]
                cand["score_components"] = scored["components"]
                cand["score_breakdown"] = scored["breakdown"]
            cand["sector_bias"] = sector_bias(
                cand["sector"], self.state.get("sector_rotation", {})
            )
            candidates.append(cand)

        survivors = sorted(
            [c for c in candidates if c["passed_filter"]],
            key=lambda c: c["score"] or 0,
            reverse=True,
        )[:MAX_CANDIDATES_FOR_LLM]
        passed = sum(1 for c in candidates if c["passed_filter"])
        self.state["candidates"] = candidates
        self.state["survivors"] = survivors
        self.state["survivor_tickers"] = [c["ticker"] for c in survivors]
        if not survivors:
            raise RuntimeError(
                "No candidates passed hard filters — pipeline halted. "
                "Inspect candidates table for fail_reasons."
            )
        return StepOutput(
            content=f"Screen: {passed}/{len(candidates)} passed; top {len(survivors)} → LLM"
        )

    def step_risk_factors(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        self.state["risk_factors"] = batch_risk_factors(tickers)
        return StepOutput(
            content=f"SEC 10-K: {len(self.state['risk_factors'])}/{len(tickers)}"
        )

    def step_news(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        self.state["news"] = _batch_news(tickers)
        return StepOutput(content=f"News fetched for {len(tickers)}")

    def step_earnings(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        self.state["earnings_alerts"] = batch_earnings_flags(tickers, within_days=5)
        return StepOutput(
            content=(
                f"Earnings within 5d: {len(self.state['earnings_alerts'])}/"
                f"{len(tickers)} flagged"
            )
        )

    def step_insider_selling(self, step_input: StepInput) -> StepOutput:
        # Kept for back-compat when FINNHUB_API_KEY is unset; the new
        # Finnhub-backed insider activity in `step_finnhub_signals` is
        # strictly richer (real Form 4 filings vs news-mention heuristic).
        tickers = set(self.state["survivor_tickers"])
        self.state["insider_selling"] = insider_selling_mentions(tickers, days=14)
        return StepOutput(
            content=f"Insider selling: {len(self.state['insider_selling'])} survivors flagged"
        )

    def step_finnhub_signals(self, step_input: StepInput) -> StepOutput:
        tickers = list(self.state["survivor_tickers"])
        self.state["finnhub_signals"] = batch_finnhub_signals(tickers)
        n = sum(1 for v in self.state["finnhub_signals"].values() if v)
        return StepOutput(
            content=f"Finnhub signals: {n}/{len(tickers)} tickers covered"
        )

    def step_quarterly_mda(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        self.state["quarterly_mda"] = batch_quarterly_mda(tickers)
        return StepOutput(
            content=f"10-Q MD&A: {len(self.state['quarterly_mda'])}/{len(tickers)}"
        )

    def step_peer_comparison(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        fundamentals = self.state.get("fundamentals", {})
        target_meta = {
            t: {
                "name": (fundamentals.get(t) or {}).get("name"),
                "sector": (fundamentals.get(t) or {}).get("sector"),
            }
            for t in tickers
        }
        self.state["peer_comparison"] = batch_peer_comparison(tickers, target_meta)
        return StepOutput(
            content=f"Peer comparison: {len(self.state['peer_comparison'])}/{len(tickers)}"
        )

    def step_earnings_transcripts(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        self.state["earnings_transcripts"] = batch_transcript_snippets(tickers)
        return StepOutput(
            content=f"Transcripts: {len(self.state['earnings_transcripts'])}/{len(tickers)}"
        )

    def step_share_trades(self, step_input: StepInput) -> StepOutput:
        tickers = self.state["survivor_tickers"]
        self.state["share_trades"] = batch_share_trade_data(tickers)
        signals = {
            (data.get("insider_summary_6mo") or {}).get("insider_signal", "neutral")
            for data in self.state["share_trades"].values()
        }
        return StepOutput(
            content=(
                f"Share trades fetched for {len(self.state['share_trades'])}"
                f"/{len(tickers)}; signals seen: {sorted(signals)}"
            )
        )

    def step_analyst(self, step_input: StepInput) -> StepOutput:
        survivors = self.state["survivors"]
        fundamentals = self.state["fundamentals"]
        technicals = self.state["technicals"]
        risk_factors = self.state["risk_factors"]
        news = self.state["news"]

        earnings_alerts = self.state.get("earnings_alerts", {})
        insider_selling = self.state.get("insider_selling", {})
        share_trades = self.state.get("share_trades", {})
        finnhub_signals = self.state.get("finnhub_signals", {})

        payloads: dict[str, dict[str, Any]] = {}
        for c in survivors:
            ticker = c["ticker"]
            fh = finnhub_signals.get(ticker) or {}
            # Prefer Finnhub's Form 4 record when available; fall back to
            # the Tavily news-mention count for tickers Finnhub doesn't cover.
            insider_activity: Any = fh.get("insider_activity") or {
                "mention_count": insider_selling.get(ticker, 0)
            }
            payloads[ticker] = {
                "fundamentals": fundamentals.get(ticker) or {},
                "technicals": technicals.get(ticker) or {},
                "universe_signals": {
                    "sources": c["sources"],
                    "conviction": c["conviction"],
                },
                "score": c["score"],
                "score_breakdown": c["score_breakdown"],
                "sector_bias": c.get("sector_bias"),
                "earnings_alert": earnings_alerts.get(ticker),
                "insider_activity": insider_activity,
                "earnings_surprise_history": fh.get("earnings_surprise") or [],
                "recommendation_trend": fh.get("recommendation_trend") or [],
                "analyst_price_targets": fh.get("price_targets") or {},
                "share_trades": share_trades.get(ticker),
                "risk_factors_10k": _trim(
                    (risk_factors.get(ticker) or {}).get("risk_factors"),
                    _RISK_FACTORS_CHARS,
                ),
                "quarterly_mda": _trim(
                    (self.state.get("quarterly_mda", {}).get(ticker) or {}).get("mda"),
                    _QUARTERLY_MDA_CHARS,
                ),
                "peers": self.state.get("peer_comparison", {}).get(ticker),
                "earnings_transcript": _trim(
                    (self.state.get("earnings_transcripts", {}).get(ticker) or {}).get("snippet"),
                    _TRANSCRIPT_CHARS,
                ),
                "news": news.get(ticker, []),
            }

        analyst = Analyst("claude", self.settings.discover_sonnet_model)
        self.state["analyses"] = analyze_batch(analyst, payloads)
        if not self.state["analyses"]:
            raise RuntimeError("All analyst calls failed — halting before Opus stages")
        return StepOutput(content=f"Analyst: {len(self.state['analyses'])} scorecards")

    def step_holdings(self, step_input: StepInput) -> StepOutput:
        try:
            holdings = fetch_portfolio_holdings()
            self.state["holdings_summary"] = _holdings_summary(holdings)
        except Exception as e:
            logger.warning("Could not fetch holdings (%s) — proceeding without", e)
            self.state["holdings_summary"] = ""
        n = (
            self.state["holdings_summary"].count("\n") + 1
            if self.state["holdings_summary"]
            else 0
        )
        return StepOutput(
            content=f"Holdings: {n} positions" if n else "Holdings: none"
        )

    def step_ranker(self, step_input: StepInput) -> StepOutput:
        ranker = Ranker(
            "claude",
            self.settings.discover_opus_model,
            consensus_runs=self.settings.discover_consensus_runs,
        )
        output = ranker.rank(
            self.state["analyses"],
            self.state["holdings_summary"],
            macro_context=self.state.get("macro_summary", ""),
            track_record_block=self.state.get("track_record_block", ""),
        )
        self.state["ranker_output"] = output
        self.state["ranker_text"] = output.full_text
        self.state["picks"] = parse_picks(output)
        picked = [t for _, t, _ in self.state["picks"]]
        return StepOutput(content=f"Ranker picked {len(picked)}: {picked}")

    def step_redteam(self, step_input: StepInput) -> StepOutput:
        redteam = RedTeam("claude", self.settings.discover_opus_model)
        redteam_output = redteam.critique(self.state["ranker_text"])
        self.state["redteam_output"] = redteam_output
        self.state["redteam_text"] = redteam_output.full_text
        return StepOutput(content="Red-team critique complete")

    def step_sizer(self, step_input: StepInput) -> StepOutput:
        sizer = Sizer("claude", self.settings.discover_opus_model)
        self.state["sizer_text"] = sizer.allocate(
            self.state["ranker_text"],
            self.state["redteam_text"],
            self.state["holdings_summary"],
            self.settings.discover_cash_budget,
        )
        return StepOutput(content="Position sizing complete")

    def step_persist_and_report(self, step_input: StepInput) -> StepOutput:
        # 1. SQLite persistence (same as before)
        with connect(self.settings.discover_db_path) as conn:
            run_id = insert_run(
                conn,
                universe_size=len(self.state["candidates"]),
                survivors=len(self.state["survivors"]),
                picks=len(self.state["picks"]),
                opus_model=self.settings.discover_opus_model,
                sonnet_model=self.settings.discover_sonnet_model,
                cash_budget=self.settings.discover_cash_budget,
                kind="discover",
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
            insert_run_outputs(
                conn,
                run_id,
                ranker_full=self.state["ranker_text"],
                redteam_full=self.state["redteam_text"],
                sizer_full=self.state["sizer_text"],
                holdings_summary=self.state["holdings_summary"],
            )

        # 2. Fetch a chart for each pick (existing chart-img.com client).
        pick_tickers = [t for _, t, _ in self.state["picks"]]
        charts: dict[str, bytes] = {}
        try:
            charts = fetch_charts(pick_tickers)
        except Exception as e:
            logger.warning("Chart fetch failed (%s) — report will omit charts", e)
        chart_cids = {t: f"chart-{t.replace('.', '-')}" for t in charts}

        # 3. Build shared section list, then render both HTML and PDF from it.
        sections = build_sections(
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
            candidates=self.state["candidates"],
            universe_size=len(self.state["candidates"]),
            holdings_summary=self.state["holdings_summary"],
            macro_summary=self.state.get("macro_summary", ""),
            sector_rotation=self.state.get("sector_rotation"),
            track_record_block=self.state.get("track_record_block", ""),
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        # 4. Send email (or fall back to logging if EMAIL_TO unset).
        today = date.today()
        picks_summary = ", ".join(pick_tickers[:5])
        subject = (
            f"Stock Discovery — {today.strftime('%b-%d')}: {picks_summary}"
            if pick_tickers
            else f"Stock Discovery — {today.strftime('%b-%d')}"
        )
        pdf_filename = f"discover-{today.isoformat()}.pdf"

        # Save PDF locally BEFORE the email attempt so a delivery failure
        # (SMTP outage, wrong creds, etc.) never costs the user the report.
        local_pdf_path = _save_local_pdf(pdf_bytes, pdf_filename)
        logger.info("Saved discover PDF locally: %s", local_pdf_path)

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
                logger.info("Sent discovery email to %s", self.settings.email_to)
            except Exception as e:
                delivery_error = str(e)
                logger.error("Email delivery failed: %s", e)
        else:
            logger.warning(
                "EMAIL_TO not set; skipping email delivery. "
                "Run %d's HTML/PDF available via state if you want to inspect them.",
                run_id,
            )

        # Always dump the full analysis to the log so the user can recover
        # every section — even when email is offline or wasn't configured.
        _log_discover_analysis(
            delivered=delivered,
            delivery_error=delivery_error,
            local_pdf_path=str(local_pdf_path),
            ranker_text=self.state["ranker_text"],
            redteam_text=self.state["redteam_text"],
            sizer_text=self.state["sizer_text"],
        )

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        self.state["html_body"] = html_body
        self.state["local_pdf_path"] = str(local_pdf_path)
        print_terminal_summary(self.state["ranker_text"], self.state["sizer_text"])
        print(f"\nPDF saved: {local_pdf_path}")
        log_path = current_log_file()
        if log_path:
            print(f"Log file:  {log_path}")
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=(
                f"Run #{run_id} {status}; PDF {len(pdf_bytes)} bytes "
                f"(saved to {local_pdf_path})"
            )
        )

    # --- workflow assembly --------------------------------------------

    def build_workflow(self) -> Workflow:
        db_path = Path(os.path.expanduser(self.settings.discover_db_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return Workflow(
            name="Stock Discovery",
            description="Find mid-long term holds via screen + Sonnet + Opus reasoning",
            db=SqliteDb(
                db_file=str(db_path),
                session_table="workflow_session",
            ),
            steps=[
                Step(name="universe", executor=self.step_universe),
                Parallel(
                    Step(name="fundamentals", executor=self.step_fundamentals),
                    Step(name="technicals", executor=self.step_technicals),
                    Step(name="sector_rotation", executor=self.step_sector_rotation),
                    Step(name="macro_regime", executor=self.step_macro_regime),
                    Step(name="track_record", executor=self.step_track_record),
                    name="market_data",
                ),
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
                    name="enrichment",
                ),
                Step(name="analyst", executor=self.step_analyst),
                Step(name="holdings", executor=self.step_holdings),
                Step(name="ranker", executor=self.step_ranker),
                Step(name="redteam", executor=self.step_redteam),
                Step(name="sizer", executor=self.step_sizer),
                Step(
                    name="persist_and_report", executor=self.step_persist_and_report
                ),
            ],
        )


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
    pipeline = DiscoverPipeline(settings)
    workflow = pipeline.build_workflow()
    logger.info("=== Stock discovery pipeline starting ===")
    workflow.print_response(input="discover", stream=True)
    if pipeline.state.get("run_id"):
        print(f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
