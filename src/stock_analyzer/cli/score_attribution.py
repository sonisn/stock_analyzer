"""`score-attribution` — which parts of the screen's score predict anything.

The screen awards points across ten components and none had ever been
measured against a realized return. This reports the rank correlation of
each against forward excess return, and refuses to draw conclusions from
too few run dates.

    uv run score-attribution
    uv run score-attribution --horizon 21
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from ..config import Settings
from ..discover.score_attribution import (
    MIN_DATES,
    attribute,
    enough_data,
    format_report,
    load_rows,
)
from ..logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="score-attribution", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="forward horizon in days (default: both 21 and 63)",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    db = Settings.from_env().discover_db_path
    for horizon in [args.horizon] if args.horizon else [21, 63]:
        rows = load_rows(db, horizon)
        results = attribute(rows)
        print()
        print(format_report(results, horizon, len(rows)))
        if results and not enough_data(results):
            dates = max(r.dates for r in results)
            print(
                f"\nWaiting on outcomes: candidates scored in September have full\n"
                f"breakdowns but their {horizon}-day horizon has not elapsed. This\n"
                f"becomes answerable once {MIN_DATES - dates} more run date(s) mature."
            )


if __name__ == "__main__":
    main()
