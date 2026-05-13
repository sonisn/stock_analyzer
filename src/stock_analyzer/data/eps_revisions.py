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

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import yfinance as yf

from ..logging import get_logger

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


def _get_cell(df: pd.DataFrame, period: str, col: str) -> int:
    """Read a single cell from the revisions DataFrame, tolerating
    yfinance's slight column-name inconsistencies (upLast7days vs
    upLast7Days, etc.)."""
    if period not in df.index:
        return 0
    # Look up case-insensitively against the DataFrame's actual columns.
    target = col.lower()
    for actual in df.columns:
        if str(actual).lower() == target:
            val = df.loc[period, actual]
            try:
                return int(val) if pd.notna(val) else 0
            except (TypeError, ValueError):
                return 0
    return 0


def fetch_eps_revisions(ticker: str) -> dict[str, Any] | None:
    """Return the per-ticker EPS revision summary, or None on any error."""
    try:
        revs = yf.Ticker(ticker).eps_revisions
    except Exception as e:
        logger.debug("eps_revisions failed for %s: %s", ticker, e)
        return None
    if revs is None or revs.empty:
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


def batch_eps_revisions(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch revisions for many tickers in parallel."""
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, r in zip(tickers, ex.map(fetch_eps_revisions, tickers), strict=False):
            if r:
                results[ticker] = r
    return results


__all__ = ["fetch_eps_revisions", "batch_eps_revisions"]
