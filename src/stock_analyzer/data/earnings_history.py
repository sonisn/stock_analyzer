"""A company's last year of results: did it beat before, and is revenue growing?

The earnings-standout check (discover/earnings_standouts.py) asks this of a
clear beat before calling it a standout, so a lone quarter doesn't make
the daily email. Two yfinance
requests per company, and only for the handful that passed every other
gate:

  - `get_earnings_dates`: reported vs estimated EPS for years of quarters;
  - `quarterly_income_stmt`: revenue for the last ~5 quarters, for the
    same quarter a year earlier.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from ..logging import get_logger
from . import yf_gateway

logger = get_logger(__name__)

# Only the quarter before: a company that has just turned the corner is
# exactly what a six-month idea wants, and a single earlier beat is enough
# to show the latest one wasn't a lone blip. Deeper checks are the daily
# analysis's job once the name is in front of it.
PRIOR_QUARTERS = 1


def prior_beats(dates: pd.DataFrame | None, report_day: date) -> tuple[int, int]:
    """(beats, quarters) over the PRIOR_QUARTERS reports before `report_day`.
    A beat is reported EPS at or above the estimate."""
    if dates is None or dates.empty:
        return 0, 0
    rows = []
    for when, row in dates.iterrows():
        day = date.fromisoformat(str(when)[:10])  # local report date
        est, eps = row.get("EPS Estimate"), row.get("Reported EPS")
        # A few days' slack: the calendar and Yahoo can differ on the date.
        if day < report_day - timedelta(days=5) and pd.notna(est) and pd.notna(eps):
            rows.append((day, float(eps) >= float(est)))
    rows = sorted(rows, reverse=True)[:PRIOR_QUARTERS]
    return sum(beat for _, beat in rows), len(rows)


def year_ago_revenue(stmt: pd.DataFrame | None, report_day: date) -> float | None:
    """Revenue of the quarter that ended about a year before this one did.
    A quarter is reported within ~3 months of its end, so the year-ago
    quarter ended 12-15 months before `report_day`."""
    if stmt is None or stmt.empty or "Total Revenue" not in stmt.index:
        return None
    lo, hi = report_day - timedelta(days=365 + 100), report_day - timedelta(days=365)
    for col, value in stmt.loc["Total Revenue"].items():
        end = date.fromisoformat(str(col)[:10])
        if lo <= end <= hi and pd.notna(value) and float(value) > 0:
            return float(value)
    return None


def fetch_track_record(ticker: str, report_day: date, revenue_now: float | None) -> dict[str, Any]:
    """{"prior_beats", "prior_quarters", "revenue_yoy_pct"}; counts are 0
    and the growth None when Yahoo has nothing."""
    dates = yf_gateway.ticker_call(
        ticker, "earnings_dates", lambda t: t.get_earnings_dates(limit=12)
    )
    stmt = yf_gateway.ticker_call(
        ticker, "quarterly_income_stmt", lambda t: t.quarterly_income_stmt
    )
    beats, quarters = prior_beats(dates, report_day)
    before = year_ago_revenue(stmt, report_day)
    growth = (revenue_now / before - 1) * 100 if revenue_now and before else None
    return {"prior_beats": beats, "prior_quarters": quarters, "revenue_yoy_pct": growth}
