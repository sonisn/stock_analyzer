"""`validate-screen` — grade the screen score and the ranker's forecasts.

Two retrospective checks over the existing `discover.db`, no LLM calls and
no new data collection:

  score       Mean forward alpha by score quintile, plus an information
              coefficient per sub-component. Tells you which parts of the
              0-105 composite in `screen.py` actually predict returns.

  calibration EV error, conviction ordering, and scenario reliability for
              the ranker's own forecasts.

Both read point-in-time values that were stored at decision time, so
neither can leak the outcome into the feature.
"""

from __future__ import annotations

import argparse
import os
import sys

from ..config import Settings
from ..discover.calibration import format_calibration_block, measure_calibration
from ..discover.score_validation import format_validation_report, validate_score


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validate-screen",
        description=(
            "Grade the screen score (and the ranker's forecasts) against realized forward returns."
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=90,
        help="Forward window in days to grade the score over (default: 90).",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=540,
        help="How far back to pull candidates/picks, in days (default: 540).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to discover.db (default: DISCOVER_DB_PATH from settings).",
    )
    parser.add_argument(
        "--what",
        choices=("score", "calibration", "both"),
        default="both",
        help="Which check to run (default: both).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from ..market_time import use_market_timezone

    use_market_timezone()
    args = _parse_args(argv)
    settings = Settings()  # type: ignore[call-arg]
    db_path = os.path.expanduser(args.db or settings.discover_db_path)

    if not os.path.exists(db_path):
        print(
            f"No database at {db_path}. Run `uv run discover-stocks` at least "
            f"once — every run stores its candidate scores, which is what this "
            f"command grades.",
            file=sys.stderr,
        )
        return 1

    if args.what in ("score", "both"):
        report = validate_score(db_path, horizon_days=args.horizon, lookback_days=args.lookback)
        print(format_validation_report(report))

    if args.what in ("calibration", "both"):
        record = measure_calibration(db_path, lookback_days=args.lookback)
        block = format_calibration_block(record)
        print()
        print("=" * 72)
        print("RANKER FORECAST CALIBRATION")
        print("=" * 72)
        print(
            block
            or (
                "Nothing scorable yet — no pick has a recorded forecast whose "
                f"{record.ev_horizon_days or 270}-day horizon has elapsed.\n"
                "Conviction, EV and scenario probabilities are persisted from "
                "now on, so this fills in as runs mature."
            )
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
