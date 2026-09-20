"""`train-model` — label past candidates, then train and validate the
forward-return ranking model on a multi-year S&P 500 price history.

No LLM calls. Network use is yfinance price downloads only (batched, and
cached for the day). A model is saved every time for the record, but the
screen only ever uses the latest version whose walk-forward validation
accepted it.

    uv run train-model                 # 63-day model, 15 years, gated population
    uv run train-model --horizon 21
    uv run train-model --no-save       # report only
    uv run train-model --horizon 21 --label beta_adj
"""

from __future__ import annotations

import argparse

import pandas as pd
from dotenv import load_dotenv

from ..config import Settings
from ..data.universe_base import load_base_universe
from ..logging import get_logger
from ..model.dataset import DATASET_HORIZONS, build_dataset, load_panel
from ..model.fundamental_features import FUNDAMENTAL_FEATURES
from ..model.labels import label_candidates
from ..model.ranker_model import format_model_report, save_model, walk_forward

logger = get_logger(__name__)


def _point_in_time(args, universe: list[str], panel, settings):
    """Point-in-time fundamentals for the training dates, or None.

    Sampled monthly and carried forward: a filing is the latest known
    fact until the next one lands. The free plan holds five years, so a
    longer `--years` still trains on prices alone before that.
    """
    if not args.fundamentals:
        return None
    from datetime import timedelta

    from ..model.fundamental_features import (
        align_to_dates,
        fetch_history,
        fill_cross_section,
        month_ends,
        to_frame,
    )

    calendar = panel.close.index
    dates = pd.DatetimeIndex(sorted({d for d in calendar}))
    # Five years back from the end of the price panel, or its start.
    start = max(dates.min().date(), (dates.max() - timedelta(days=365 * 5)).date())
    as_of_dates = month_ends(start, dates.max().date())
    print(f"Point-in-time fundamentals: {len(as_of_dates)} monthly as-of dates from {start}")
    history = fetch_history(universe, as_of_dates, cache_dir=settings.model_cache_dir)
    if not history:
        print("No point-in-time fundamentals available — training on prices alone")
        return None
    frame = to_frame(history)
    aligned = align_to_dates(frame, dates, sorted(panel.close.columns))
    covered = aligned.notna().any(axis=1).mean() * 100
    print(f"  {len(history)} dates fetched, {covered:.0f}% of rows covered before median fill")
    return fill_cross_section(aligned)


def main(argv: list[str] | None = None) -> None:
    from ..market_time import use_market_timezone

    use_market_timezone()
    load_dotenv()
    parser = argparse.ArgumentParser(prog="train-model", description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon", type=int, default=63, choices=DATASET_HORIZONS)
    parser.add_argument("--population", default="gated", choices=("gated", "all"))
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument(
        "--label",
        default="excess",
        choices=("excess", "beta_adj"),
        help="excess = return minus SPY; beta_adj = return minus trailing beta x SPY",
    )
    parser.add_argument(
        "--fundamentals",
        action="store_true",
        help="add point-in-time margins and leverage as filed on each date "
        "(needs WISESHEETS_API_KEY; the free plan holds 5 years of history)",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings()

    if not args.skip_labels:
        n = label_candidates(settings.discover_db_path)
        print(f"Candidate outcomes labeled this run: {n}")

    universe = list(load_base_universe())
    panel = load_panel(universe, settings.model_cache_dir, years=args.years)
    data = build_dataset(panel, fundamentals=_point_in_time(args, universe, panel, settings))
    extra = [c for c in FUNDAMENTAL_FEATURES if c in data.columns]
    print(
        f"Training set: {len(data):,} weekly rows, {data.index.get_level_values('ticker').nunique()} "
        f"tickers, {int(data['gated'].sum()):,} passing the trend gate"
        + (f", {len(extra)} point-in-time fundamental feature(s)" if extra else "")
    )
    result = walk_forward(
        data,
        panel.spy.dropna().index,
        horizon=args.horizon,
        population=args.population,
        label_kind=args.label,
    )
    print(format_model_report(result))
    if not args.no_save:
        version = save_model(settings.discover_db_path, result)
        print(f"Saved as model version {version}")


if __name__ == "__main__":
    main()
