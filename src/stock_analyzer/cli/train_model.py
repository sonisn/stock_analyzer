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

from dotenv import load_dotenv

from ..config import Settings
from ..data.universe_base import load_base_universe
from ..logging import get_logger
from ..model.dataset import HORIZONS, build_dataset, load_panel
from ..model.labels import label_candidates
from ..model.ranker_model import format_model_report, save_model, walk_forward

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    from ..market_time import use_market_timezone

    use_market_timezone()
    load_dotenv()
    parser = argparse.ArgumentParser(prog="train-model", description=__doc__.split("\n\n")[0])
    parser.add_argument("--horizon", type=int, default=63, choices=HORIZONS)
    parser.add_argument("--population", default="gated", choices=("gated", "all"))
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument(
        "--label",
        default="excess",
        choices=("excess", "beta_adj"),
        help="excess = return minus SPY; beta_adj = return minus trailing beta x SPY",
    )
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--skip-labels", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings()

    if not args.skip_labels:
        n = label_candidates(settings.discover_db_path)
        print(f"Candidate outcomes labeled this run: {n}")

    panel = load_panel(list(load_base_universe()), settings.model_cache_dir, years=args.years)
    data = build_dataset(panel)
    print(
        f"Training set: {len(data):,} weekly rows, {data.index.get_level_values('ticker').nunique()} "
        f"tickers, {int(data['gated'].sum()):,} passing the trend gate"
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
