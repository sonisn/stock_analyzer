"""Stock discovery pipeline — Agno Workflow version.

Run via:   python -m stock_analyzer.cli.discover

Declarative shape:
  universe
  ├ Parallel(fundamentals, technicals)
  screen
  ├ Parallel(risk_factors, news)
  analyst (Sonnet, parallel fan-out inside step)
  holdings
  ranker (multi-provider consensus — one round per DISCOVER_RANKER_PROVIDERS entry)
  macro_veto (deterministic, suppresses high-momentum picks in a risk-off regime)
  redteam (DISCOVER_REDTEAM_PROVIDER, default a different provider than the ranker's)
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
from pathlib import Path
from typing import Any

from agno.db.sqlite import SqliteDb
from agno.workflow import Step, Workflow
from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger
from ..preflight import PreflightError, preflight
from ..usage import BUDGET, TRACKER, log_usage_summary, set_extra_prices
from .discover_steps.analysis_steps import AnalysisSteps
from .discover_steps.data_steps import DataSteps
from .discover_steps.helpers import parallel, without_step_retries
from .discover_steps.report_steps import ReportSteps

logger = get_logger(__name__)


class DiscoverPipeline(DataSteps, AnalysisSteps, ReportSteps):
    """Holds shared state across Workflow steps.

    Each `step_*` method is bound to this instance, so steps read/write
    self.state instead of round-tripping data through StepOutput content.
    Fatal conditions (empty universe, no survivors) raise RuntimeError —
    the Workflow aborts cleanly and the run shows as failed in workflow_session.
    """

    # Share of the post-reserve budget the Analyst fan-out may plan to use;
    # the rest is left for the Ranker rounds (and, in rebalance, reviews).
    ANALYST_BUDGET_SHARE = 0.5

    def __init__(self, settings: Settings):
        self.settings = settings
        self.state: dict[str, Any] = {}
        TRACKER.reset()
        set_extra_prices(settings.llm_prices)
        BUDGET.configure(settings.discover_max_cost_usd)

    # --- workflow assembly --------------------------------------------

    def build_workflow(self) -> Workflow:
        db_path = Path(os.path.expanduser(self.settings.discover_db_path))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        return without_step_retries(
            Workflow(
                name="Stock Discovery",
                description="Find mid-long term holds via screen + Sonnet + Opus reasoning",
                db=SqliteDb(
                    db_file=str(db_path),
                    session_table="workflow_session",
                ),
                steps=[
                    Step(name="universe", executor=self.step_universe),
                    # Technicals first, alone among the Yahoo-backed steps: one
                    # request per name buys the trend gate, which decides who is
                    # worth the three-requests-per-name fetches below.
                    parallel(
                        Step(name="technicals", executor=self.step_technicals),
                        Step(name="sector_rotation", executor=self.step_sector_rotation),
                        Step(name="macro_regime", executor=self.step_macro_regime),
                        Step(name="track_record", executor=self.step_track_record),
                        name="market_data",
                    ),
                    Step(name="prescreen", executor=self.step_prescreen),
                    parallel(
                        Step(name="fundamentals", executor=self.step_fundamentals),
                        # EPS revisions run here so the score function can pick
                        # up the +/-5 trend bonus from direction_30d.
                        Step(name="eps_revisions", executor=self.step_eps_revisions),
                        Step(
                            name="historical_volatility",
                            executor=self.step_historical_volatility,
                        ),
                        name="candidate_data",
                    ),
                    # Market themes need sector_rotation + macro_regime as input,
                    # so it runs sequentially after the market_data block.
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
                        Step(name="contracted_book", executor=self.step_contracted_book),
                        name="enrichment",
                    ),
                    Step(name="analyst", executor=self.step_analyst),
                    Step(name="holdings", executor=self.step_holdings),
                    Step(name="ranker", executor=self.step_ranker),
                    Step(name="macro_veto", executor=self.step_macro_veto),
                    Step(name="redteam", executor=self.step_redteam),
                    Step(name="sizer", executor=self.step_sizer),
                    Step(name="persist_and_report", executor=self.step_persist_and_report),
                    Step(name="history_upkeep", executor=self.step_history_upkeep),
                    Step(name="dashboard", executor=self.step_refresh_dashboard),
                ],
            )
        )


def run() -> None:
    from ..market_time import use_market_timezone

    use_market_timezone()
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
            needs_discover_providers=True,
        )
    except PreflightError as e:
        logger.error("%s", e)
        raise SystemExit(2) from e
    pipeline = DiscoverPipeline(settings)
    workflow = pipeline.build_workflow()
    logger.info("=== Stock discovery pipeline starting ===")
    try:
        workflow.print_response(input="discover", stream=True)
    finally:
        # Request budget for the run: how much Yahoo traffic it took, how
        # often it was throttled, and what the pacer settled on. Read this
        # before touching YF_RATE_LIMIT_PER_MIN.
        yf_gateway.log_stats("discover run")
        log_usage_summary()
        unavailable = yf_gateway.unavailable_symbols()
        if unavailable:
            logger.info(
                "Symbols Yahoo had no data for this run (skipped after the first miss): %s",
                ", ".join(sorted(unavailable)),
            )
    if pipeline.state.get("run_id"):
        print(f"\nRun #{pipeline.state['run_id']} stored in {settings.discover_db_path}")


def main() -> None:
    run()


if __name__ == "__main__":
    main()
