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

import os
from pathlib import Path

from agno.db.sqlite import SqliteDb
from agno.workflow import Step, Workflow
from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger
from ..preflight import PreflightError, preflight
from ..usage import log_usage_summary
from .discover import DiscoverPipeline
from .discover_steps.helpers import parallel, without_step_retries
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

    # --- workflow assembly -------------------------------------------------

    def build_workflow(self) -> Workflow:
        db_path = Path(os.path.expanduser(self.settings.discover_db_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return without_step_retries(
            Workflow(
                name="Portfolio Rebalance",
                description=(
                    "Discover new picks + review current holdings + emit aggressive rebalance plan"
                ),
                db=SqliteDb(db_file=str(db_path), session_table="workflow_session"),
                steps=[
                    Step(name="universe", executor=self.step_universe),
                    parallel(
                        Step(name="fundamentals", executor=self.step_fundamentals),
                        Step(name="technicals", executor=self.step_technicals),
                        Step(name="sector_rotation", executor=self.step_sector_rotation),
                        Step(name="macro_regime", executor=self.step_macro_regime),
                        Step(name="track_record", executor=self.step_track_record),
                        Step(name="eps_revisions", executor=self.step_eps_revisions),
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
                    Step(name="thesis_check", executor=self.step_thesis_check),
                    parallel(
                        Step(name="risk_factors", executor=self.step_risk_factors),
                        Step(name="quarterly_mda", executor=self.step_quarterly_mda),
                        Step(name="news", executor=self.step_news),
                        Step(name="earnings", executor=self.step_earnings),
                        Step(name="insider_selling", executor=self.step_insider_selling),
                        Step(name="share_trades", executor=self.step_share_trades),
                        Step(name="peer_comparison", executor=self.step_peer_comparison),
                        Step(name="earnings_transcripts", executor=self.step_earnings_transcripts),
                        Step(name="finnhub_signals", executor=self.step_finnhub_signals),
                        Step(name="holdings_data", executor=self.step_holdings_data),
                        name="enrichment",
                    ),
                    Step(name="analyst", executor=self.step_analyst),
                    Step(name="holdings", executor=self.step_holdings),
                    Step(name="ranker", executor=self.step_ranker),
                    Step(name="redteam", executor=self.step_redteam),
                    Step(name="sizer", executor=self.step_sizer),
                    Step(name="contracted_book", executor=self.step_contracted_book),
                    Step(name="review_holdings", executor=self.step_review_holdings),
                    Step(name="cc_data", executor=self.step_cc_data),
                    Step(name="csp_data", executor=self.step_csp_data),
                    Step(name="tax_harvest", executor=self.step_tax_harvest),
                    Step(name="rebalance", executor=self.step_rebalance),
                    Step(name="premortem", executor=self.step_premortem),
                    Step(
                        name="persist_and_email_rebalance",
                        executor=self.step_persist_and_email_rebalance,
                    ),
                    Step(name="history_upkeep", executor=self.step_history_upkeep),
                    Step(name="dashboard", executor=self.step_refresh_dashboard),
                ],
            )
        )


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
    workflow = pipeline.build_workflow()
    logger.info("=== Portfolio rebalance pipeline starting ===")
    try:
        workflow.print_response(input="rebalance", stream=True)
    finally:
        log_usage_summary()
    if pipeline.state.get("run_id"):
        print(f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
