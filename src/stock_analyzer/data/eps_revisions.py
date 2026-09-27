"""Analyst EPS-estimate revisions over the last 7 and 30 days.

One of the single most predictive equity signals: when analysts are
collectively *raising* their forward EPS estimates for a ticker, the
stock tends to outperform — and the reverse holds for falling estimates.
This isn't captured by the recommendation_mean or target_price snapshots
already in fundamentals; those are slow-moving levels. Revisions are the
*flow*.

Source: yfinance's `Ticker.eps_revisions`, which returns a 4-row table
indexed by period (current quarter '0q', next quarter '+1q', current
year '0y', next year '+1y') and columns counting analysts who raised /
lowered their EPS estimate in the last 7 and 30 days.

We aggregate to a per-ticker summary:
  - `current_quarter_up_30d` / `current_quarter_down_30d`
  - `current_year_up_30d` / `current_year_down_30d`
  - `net_revisions_30d` — current-quarter + current-year, net = ups - downs
  - `direction_30d` — 'raising' / 'lowering' / 'stable'

Net direction is the signal the LLM should weight: 'raising across both
windows' is a strong forward-thesis confirmation; 'lowering' is a yellow
flag the system shouldn't ignore behind a HOLD verdict.
"""

from __future__ import annotations

from typing import Any

import polars as pl

from ..logging import get_logger
from . import fetch_cache, frames, yf_gateway

logger = get_logger(__name__)

_MAX_WORKERS = 4

# Period rows in yfinance's DataFrame. Capital letters / lowercase
# differences are real yfinance quirks — we tolerate both.
_PERIODS = {
    "current_quarter": "0q",
    "next_quarter": "+1q",
    "current_year": "0y",
    "next_year": "+1y",
}


def _get_cell(df: pl.DataFrame, period: str, col: str) -> int:
    """Read a single cell from the revisions table (frames.table_from_pandas
    shape), tolerating yfinance's slight column-name inconsistencies
    (upLast7days vs upLast7Days, etc.)."""
    target = col.lower()
    for actual in df.columns:
        if actual != "index" and actual.lower() == target:
            val = frames.cell(df, period, actual)
            try:
                return int(val) if val is not None and val == val else 0
            except TypeError, ValueError:
                return 0
    return 0


def fetch_eps_revisions(ticker: str) -> dict[str, Any] | None:
    """Return the per-ticker EPS revision summary, or None on any error."""
    revs = frames.table_from_pandas(
        yf_gateway.ticker_call(ticker, "eps_revisions", lambda t: t.eps_revisions)
    )
    if revs is None:
        return None

    cq_up_30 = _get_cell(revs, "0q", "upLast30days")
    cq_down_30 = _get_cell(revs, "0q", "downLast30days")
    cq_up_7 = _get_cell(revs, "0q", "upLast7days")
    cq_down_7 = _get_cell(revs, "0q", "downLast7days")
    cy_up_30 = _get_cell(revs, "0y", "upLast30days")
    cy_down_30 = _get_cell(revs, "0y", "downLast30days")
    nq_up_30 = _get_cell(revs, "+1q", "upLast30days")
    nq_down_30 = _get_cell(revs, "+1q", "downLast30days")
    ny_up_30 = _get_cell(revs, "+1y", "upLast30days")
    ny_down_30 = _get_cell(revs, "+1y", "downLast30days")

    # Aggregate net revisions across current quarter + current year
    # (the two windows that matter most for a 6-12 month thesis).
    net_30d = (cq_up_30 - cq_down_30) + (cy_up_30 - cy_down_30)
    net_7d = cq_up_7 - cq_down_7
    if net_30d >= 2:
        direction_30d = "raising"
    elif net_30d <= -2:
        direction_30d = "lowering"
    else:
        direction_30d = "stable"

    return {
        "ticker": ticker,
        "current_quarter_up_30d": cq_up_30,
        "current_quarter_down_30d": cq_down_30,
        "current_quarter_up_7d": cq_up_7,
        "current_quarter_down_7d": cq_down_7,
        "next_quarter_up_30d": nq_up_30,
        "next_quarter_down_30d": nq_down_30,
        "current_year_up_30d": cy_up_30,
        "current_year_down_30d": cy_down_30,
        "next_year_up_30d": ny_up_30,
        "next_year_down_30d": ny_down_30,
        "net_revisions_30d": net_30d,
        "net_revisions_7d": net_7d,
        "direction_30d": direction_30d,
    }


def batch_eps_revisions(
    tickers: list[str], *, refresh: list[str] | tuple[str, ...] = ()
) -> dict[str, dict[str, Any]]:
    """Fetch revisions for many tickers in parallel."""
    # Cached for a week like fundamentals, and expired after the same
    # earnings report: the answer is stamped with the next report date from
    # the cached fundamentals (fetched first in every caller that screens).
    upcoming = {
        t: (e.get("value") or {}).get("next_earnings")
        for t, e in fetch_cache.entries("fundamentals").items()
    }

    def fetch(todo: list[str]):
        for ticker, r in yf_gateway.map_symbols(fetch_eps_revisions, todo, workers=_MAX_WORKERS):
            yield ticker, ({**r, "next_earnings": upcoming.get(ticker)} if r else None)

    return fetch_cache.fetch_many("eps_revisions", tickers, fetch, refresh=refresh)


def fetch_estimate_change(ticker: str, period: str = "+1y") -> float | None:
    """How much the consensus EPS estimate for `period` (default next
    fiscal year) has moved over the last 30 days, as a fraction: 0.05 is
    raised 5%. From yfinance's `Ticker.eps_trend`, which carries the
    estimate as it stood 7/30/60/90 days ago — no stored history needed.
    None when either value is missing or the base is not positive (a
    loss-making year's percentage change means nothing)."""
    trend = yf_gateway.ticker_call(ticker, "eps_trend", lambda t: t.eps_trend)
    trend = frames.table_from_pandas(trend)
    now, before = frames.cell(trend, period, "current"), frames.cell(trend, period, "30daysAgo")
    try:
        now, before = float(now), float(before)
    except TypeError, ValueError:
        return None
    if now != now or before != before or before <= 0:
        return None
    return now / before - 1


__all__ = ["fetch_eps_revisions", "batch_eps_revisions", "fetch_estimate_change"]
