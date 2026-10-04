"""Stock discovery pipeline.

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

Steps run through `pipeline.run_pipeline`, which keeps each step's timing
and summary (or error) in the `pipeline_steps` table of the same
discover.db the domain tables live in.

State is shared across steps via a DiscoverPipeline instance — each step is a
bound method that reads/writes self.state and returns a summary line.
"""

from __future__ import annotations

import argparse
from typing import Any

from dotenv import load_dotenv

from ..config import Settings
from ..data import finnhub, yf_gateway
from ..logging import get_logger
from ..pipeline import Parallel, PipelineFailed, Step, run_pipeline
from ..preflight import PreflightError, preflight
from ..usage import BUDGET, TRACKER, log_usage_summary, set_extra_prices
from .discover_steps.analysis_steps import AnalysisSteps
from .discover_steps.data_steps import DataSteps
from .discover_steps.report_steps import ReportSteps

logger = get_logger(__name__)


class DiscoverPipeline(DataSteps, AnalysisSteps, ReportSteps):
    """Holds shared state across pipeline steps.

    Each `step_*` method is bound to this instance and reads/writes
    self.state. Fatal conditions (an empty universe) raise RuntimeError —
    the run stops there and the step shows as failed in pipeline_steps.
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

    # --- pipeline -------------------------------------------------------------

    def steps(self) -> list[Step | Parallel]:
        return [
            Step("universe", self.step_universe),
            # Technicals first, alone among the Yahoo-backed steps: one
            # request per name buys the trend gate, which decides who is
            # worth the three-requests-per-name fetches below.
            Parallel(
                Step("technicals", self.step_technicals),
                Step("sector_rotation", self.step_sector_rotation),
                Step("macro_regime", self.step_macro_regime),
                Step("track_record", self.step_track_record),
                name="market_data",
            ),
            Step("prescreen", self.step_prescreen),
            Parallel(
                Step("fundamentals", self.step_fundamentals),
                # EPS revisions run here so the score function can pick
                # up the +/-5 trend bonus from direction_30d.
                Step("eps_revisions", self.step_eps_revisions),
                Step("historical_volatility", self.step_historical_volatility),
                name="candidate_data",
            ),
            # Market themes need sector_rotation + macro_regime as input,
            # so it runs sequentially after the market_data block.
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
                Step("contracted_book", self.step_contracted_book),
                name="enrichment",
            ),
            Step("analyst", self.step_analyst),
            Step("holdings", self.step_holdings),
            Step("ranker", self.step_ranker),
            Step("macro_veto", self.step_macro_veto),
            Step("redteam", self.step_redteam),
            Step("sizer", self.step_sizer),
            Step("persist_and_report", self.step_persist_and_report),
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
            needs_discover_providers=True,
        )
    except PreflightError as e:
        logger.error("%s", e)
        raise SystemExit(2) from e
    pipeline = DiscoverPipeline(settings)
    try:
        run_pipeline("Stock discovery", pipeline.steps(), db_path=settings.discover_db_path)
    except PipelineFailed as e:
        logger.error("%s", e)
        raise SystemExit(1) from e
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
    argparse.ArgumentParser(
        prog="discover-stocks",
        description="Find long-term picks: screen, research, rank, size, and email the report. "
        "Makes paid model calls (capped by DISCOVER_MAX_COST_USD).",
    ).parse_args()
    run()


if __name__ == "__main__":
    main()
