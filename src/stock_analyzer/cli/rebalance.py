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
from ..data.fundamentals import batch_fundamentals
from ..data.insider_selling import insider_selling_mentions
from ..data.sec_edgar import batch_risk_factors
from ..data.share_trades import batch_share_trade_data
from ..data.technical_indicators import batch_technicals
from ..data.transactions import fetch_transaction_history, to_tax_payloads
from ..discover.persistence import (
    connect,
    insert_candidate,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from ..discover.rebalancer import Rebalancer
from ..discover.report import (
    Section,
    build_sections,
    print_terminal_summary,
    render_html_email,
    render_pdf,
)
from ..discover.reviewer import Reviewer, review_batch
from ..logging import get_logger
from ..reporting.smtp import SmtpServer
from .discover import DiscoverPipeline

logger = get_logger(__name__)


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
        return StepOutput(content=f"Holdings data fetched for {len(tickers)} tickers")

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
        tickers = set(self.state["survivor_tickers"])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        self.state["insider_selling"] = insider_selling_mentions(tickers, days=14)
        return StepOutput(
            content=f"Insider selling: {len(self.state['insider_selling'])} flagged"
        )

    def step_share_trades(self, step_input: StepInput) -> StepOutput:
        """Override: fetch insider/institutional data for both survivors AND holdings."""
        tickers = set(self.state["survivor_tickers"])
        if self.state.get("holdings_tickers"):
            tickers |= set(self.state["holdings_tickers"])
        self.state["share_trades"] = batch_share_trade_data(list(tickers))
        return StepOutput(
            content=f"Share trades fetched for {len(self.state['share_trades'])}/{len(tickers)}"
        )

    def step_review_holdings(self, step_input: StepInput) -> StepOutput:
        positions = self.state["holdings_positions"]
        fund = self.state["holdings_fundamentals"]
        tech = self.state["holdings_technicals"]
        rfs = self.state["holdings_risk_factors"]
        selling = self.state.get("insider_selling", {})

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
                "insider_selling_mentions": selling.get(ticker, 0),
                "share_trades": self.state.get("share_trades", {}).get(ticker),
                "risk_factors_10k": (rfs.get(ticker) or {}).get("risk_factors"),
                "tax_lots": self.state.get("tax_lots", {}).get(ticker),
            }

        reviewer = Reviewer("claude", self.settings.discover_sonnet_model)
        self.state["holdings_reviews"] = review_batch(reviewer, payloads)
        return StepOutput(
            content=f"Reviewed {len(self.state['holdings_reviews'])} holdings"
        )

    def step_rebalance(self, step_input: StepInput) -> StepOutput:
        rebalancer = Rebalancer("claude", self.settings.discover_opus_model)
        self.state["rebalance_text"] = rebalancer.decide(
            self.state["holdings_reviews"],
            self.state["ranker_text"],
            self.state.get("cash_balance"),
            self.state.get("macro_summary", ""),
        )
        return StepOutput(content="Rebalance plan generated")

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
            for ticker, text in self.state["analyses"].items():
                insert_scorecard(conn, run_id, ticker, text)
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
        )
        html_body = render_html_email(sections, chart_cids)
        pdf_bytes = render_pdf(sections, charts)

        today = date.today()
        subject = f"Portfolio Rebalance — {today.strftime('%b-%d')}"
        pdf_filename = f"rebalance-{today.isoformat()}.pdf"

        delivered = False
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
                logger.error("Email delivery failed: %s", e)
        else:
            logger.warning("EMAIL_TO not set; skipping email")

        print_terminal_summary(self.state["ranker_text"], self.state["sizer_text"])
        print("\n" + "=" * 60)
        print("REBALANCE PLAN")
        print("=" * 60)
        print(self.state["rebalance_text"])

        self.state["run_id"] = run_id
        self.state["pdf_bytes"] = pdf_bytes
        status = "emailed" if delivered else "persisted (no email)"
        return StepOutput(
            content=f"Rebalance run #{run_id} {status}; PDF {len(pdf_bytes)} bytes"
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
                    Step(name="holdings_fetch", executor=self.step_holdings_fetch),
                    Step(
                        name="transaction_history",
                        executor=self.step_transaction_history,
                    ),
                    name="market_data",
                ),
                Step(name="screen", executor=self.step_screen),
                Parallel(
                    Step(name="risk_factors", executor=self.step_risk_factors),
                    Step(name="news", executor=self.step_news),
                    Step(name="earnings", executor=self.step_earnings),
                    Step(name="insider_selling", executor=self.step_insider_selling),
                    Step(name="share_trades", executor=self.step_share_trades),
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
    holdings_reviews: dict[str, str],
    ranker_text: str,
    redteam_text: str,
    sizer_text: str,
    candidates: list[dict[str, Any]],
    cash_balance: float | None,
    macro_summary: str,
    sector_rotation: dict[str, Any] | None,
) -> list[Section]:
    """Rebalance-specific section layout: action plan first, then reviews,
    then the discover picks as appendix."""
    today = date.today().isoformat()
    cash_line = (
        f"Cash available: ${cash_balance:,.0f}."
        if cash_balance is not None
        else "Cash: unknown."
    )
    sections: list[Section] = [
        Section("heading", f"Portfolio Rebalance — {today}", level=1),
        Section(
            "para",
            f"Holdings reviewed: {len(holdings_reviews)}. {cash_line}",
        ),
    ]
    if macro_summary:
        sections.append(Section("heading", "Macro regime", level=2))
        sections.append(Section("blockquote", macro_summary))

    sections.append(Section("page_break"))
    sections.append(Section("heading", "Rebalance plan (action list)", level=1))
    sections.append(Section("preformatted", rebalance_text))

    sections.append(Section("page_break"))
    sections.append(Section("heading", "Per-holding reviews", level=1))
    for ticker in sorted(holdings_reviews.keys()):
        sections.append(Section("heading", ticker, level=2))
        sections.append(Section("preformatted", holdings_reviews[ticker]))

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
    sections.append(Section("page_break"))
    sections.append(
        Section("heading", "Discover picks (input to rebalancer)", level=1)
    )
    sections.extend(discover_sections[2:])
    return sections


def run() -> None:
    load_dotenv()
    settings = Settings.from_env()
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
