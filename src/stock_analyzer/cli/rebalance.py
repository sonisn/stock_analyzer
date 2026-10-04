"""Portfolio rebalance pipeline.

Run: python -m stock_analyzer.cli.rebalance

Extends the discover pipeline with three new steps:
  - holdings_fetch    SnapTrade positions + cash balance
  - holdings_data     fundamentals/technicals/risk factors for held tickers
  - review_holdings   Sonnet per holding → HOLD / TRIM / SELL verdict
  - rebalance         Opus combines verdicts + discover picks + cash into action list

User-configured behavior (locked in via conversation):
  - Sizing: self-fund from SELL/TRIM proceeds AND add available cash
  - Horizon: every holding and pick is a long-term (3-5 year) investment;
    sells need a broken thesis, a clearly better long-term use of the
    money, concentration, or a tax loss — never short-term price action

Output: email with HTML body (charts for new picks inline) + PDF attachment.
Run history shares the discover.db SQLite file.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger
from ..pipeline import Parallel, PipelineFailed, Step, run_pipeline
from ..preflight import PreflightError, preflight
from ..usage import log_usage_summary
from .discover import DiscoverPipeline
from .rebalance_steps.data_steps import RebalanceDataSteps
from .rebalance_steps.helpers import (
    _build_position_splits,
    _build_rebalance_sections,
    build_email_subject,
)
from .rebalance_steps.plan_steps import RebalancePlanSteps
from .rebalance_steps.report_steps import RebalanceReportSteps
from .rebalance_steps.review_steps import RebalanceReviewSteps

# Imported from here by portfolio, tax-planner and the tests.
__all__ = [
    "RebalancePipeline",
    "_build_position_splits",
    "_build_rebalance_sections",
    "build_email_subject",
    "main",
    "run",
]

logger = get_logger(__name__)


class RebalancePipeline(
    RebalanceDataSteps,
    RebalanceReviewSteps,
    RebalancePlanSteps,
    RebalanceReportSteps,
    DiscoverPipeline,
):
    """Discovery + per-holding review + Opus rebalance plan delivery."""

    # Rebalance runs also review every holding, so the discover-side
    # Analyst fan-out gets a smaller slice of the budget.
    ANALYST_BUDGET_SHARE = 0.3

    # --- pipeline -------------------------------------------------------------

    def steps(self) -> list[Step | Parallel]:
        return [
            Step("universe", self.step_universe),
            Parallel(
                Step("fundamentals", self.step_fundamentals),
                Step("technicals", self.step_technicals),
                Step("sector_rotation", self.step_sector_rotation),
                Step("macro_regime", self.step_macro_regime),
                Step("track_record", self.step_track_record),
                Step("eps_revisions", self.step_eps_revisions),
                Step("holdings_fetch", self.step_holdings_fetch),
                Step("transaction_history", self.step_transaction_history),
                name="market_data",
            ),
            # Market themes after market_data (depends on sector_rotation +
            # macro_regime from inside the parallel block).
            Step("market_themes", self.step_market_themes),
            Step("screen", self.step_screen),
            Step("thesis_check", self.step_thesis_check),
            Parallel(
                Step("risk_factors", self.step_risk_factors),
                Step("quarterly_mda", self.step_quarterly_mda),
                Step("news", self.step_news),
                Step("earnings", self.step_earnings),
                Step("insider_selling", self.step_insider_selling),
                Step("share_trades", self.step_share_trades),
                Step("peer_comparison", self.step_peer_comparison),
                Step("earnings_transcripts", self.step_earnings_transcripts),
                Step("finnhub_signals", self.step_finnhub_signals),
                Step("holdings_data", self.step_holdings_data),
                name="enrichment",
            ),
            Step("analyst", self.step_analyst),
            Step("holdings", self.step_holdings),
            Step("ranker", self.step_ranker),
            Step("redteam", self.step_redteam),
            Step("sizer", self.step_sizer),
            Step("contracted_book", self.step_contracted_book),
            Step("review_holdings", self.step_review_holdings),
            Step("cc_data", self.step_cc_data),
            Step("csp_data", self.step_csp_data),
            Step("tax_harvest", self.step_tax_harvest),
            Step("rebalance", self.step_rebalance),
            Step("premortem", self.step_premortem),
            Step("persist_and_email_rebalance", self.step_persist_and_email_rebalance),
            Step("history_upkeep", self.step_history_upkeep),
            Step("dashboard", self.step_refresh_dashboard),
        ]


def run() -> None:
    load_dotenv()
    # Pacing knobs live in the environment, and these modules are
    # imported before `.env` is loaded — re-read them now.
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
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
    try:
        run_pipeline("Portfolio rebalance", pipeline.steps(), db_path=settings.discover_db_path)
    except PipelineFailed as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
    finally:
        log_usage_summary()
    if pipeline.state.get("run_id"):
        print(f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}")


def main() -> None:
    argparse.ArgumentParser(
        prog="rebalance-portfolio",
        description="Discover picks, review every holding, and email a rebalance plan. "
        "Makes paid model calls (capped by DISCOVER_MAX_COST_USD).",
    ).parse_args()
    run()


if __name__ == "__main__":
    main()
