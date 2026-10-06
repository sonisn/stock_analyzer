"""`factor-study` — fifteen years of month-ends: which fundamentals rank the
next six to twelve months' returns, and does sizing by volatility help?

    uv run factor-study                  # both studies
    uv run factor-study --part risk      # prices only: seconds once bars are on disk
    uv run factor-study --part fundamentals --refresh-sec

Free data only: the bar store (Yahoo, extended to today) and the SEC's
company facts (one request per company, cached for 30 days). No LLM calls.
The report is printed and saved under the reports directory.
"""

from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

from ..config import Settings
from ..logging import get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="factor-study", description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=("all", "fundamentals", "risk"), default="all")
    parser.add_argument("--years", type=int, default=15)
    parser.add_argument(
        "--refresh-sec", action="store_true", help="download SEC company facts even if cached"
    )
    args = parser.parse_args(argv)
    settings = Settings()

    from ..data import yf_gateway
    from ..data.industry_groups import sector_map
    from ..data.universe_base import sp500
    from ..model import factor_study as fs
    from ..model.dataset import panel_from_bars
    from ..model.sec_history import build_panel

    universe = sorted(sp500())
    start = date.today() - timedelta(days=round(args.years * 365.25))
    bars = yf_gateway.daily_bars_many(sorted({*universe, "SPY"}), start=start, what="factor-study")
    panel = panel_from_bars(bars)
    # Month-ends that are over, and a year in: the volatility and beta
    # features need 252 sessions of history.
    sessions = fs.month_end_sessions(panel.calendar())
    dates = [d for d in sessions[12:] if d < date.today().replace(day=1)]
    sections = []
    if args.part in ("all", "risk"):
        sections.append(fs.format_risk(fs.risk_study(panel, dates)))
    if args.part in ("all", "fundamentals"):
        fund = build_panel(
            universe,
            bars,
            dates,
            cache_dir=settings.model_cache_dir,
            refresh=args.refresh_sec,
        )
        study = fs.fundamentals_study(fund, fs.outcomes(panel, dates), sector_map())
        sections.append(fs.format_fundamentals(study))
    report = "\n\n".join(sections)
    print(report)
    # Beside the dashboard, in the reports directory.
    out_dir = Path(os.path.expanduser(settings.dashboard_path)).parent
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"factor_study_{args.part}_{date.today():%Y-%m-%d}.txt"
        path.write_text(report + "\n")
        print(f"\nSaved to {path}")
    except OSError as e:
        logger.warning("Could not save the report (%s)", e)


if __name__ == "__main__":
    main()
