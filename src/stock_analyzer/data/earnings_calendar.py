"""Earnings proximity — flag tickers reporting within the next N trading days.

Don't auto-reject "earnings tomorrow" candidates; just surface the date so
the analyst payload and the markdown report can warn that buying now means
betting on the print.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..logging import get_logger
from . import yf_gateway

logger = get_logger(__name__)

_MAX_WORKERS = 5


def _coerce_date(value: Any) -> date | None:
    """yfinance returns earnings dates in mixed shapes — dict-of-list, datetime,
    pandas Timestamp, plain date. Normalize to date or None."""
    if value is None:
        return None
    if hasattr(value, "date") and callable(value.date):
        try:
            d = value.date()
            return d if isinstance(d, date) else None
        except TypeError:
            pass
    if isinstance(value, date):
        return value
    return None


def next_earnings_date(ticker: str) -> date | None:
    cal = yf_gateway.ticker_call(ticker, "earnings_calendar", lambda t: t.calendar)
    if cal is None or (hasattr(cal, "empty") and cal.empty):
        return None

    # Dict shape (newer yfinance):
    if isinstance(cal, dict):
        ed = cal.get("Earnings Date")
        if isinstance(ed, list) and ed:
            return _coerce_date(ed[0])
        return _coerce_date(ed)

    # DataFrame shape (older yfinance):
    if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
        try:
            return _coerce_date(cal["Earnings Date"].iloc[0])
        except Exception:
            return None
    return None


def earnings_within_days(ticker: str, days: int = 5) -> dict[str, Any] | None:
    ed = next_earnings_date(ticker)
    if ed is None:
        return None
    delta = (ed - date.today()).days
    if 0 <= delta <= days:
        return {
            "ticker": ticker,
            "earnings_date": ed.isoformat(),
            "days_until": delta,
        }
    return None


def batch_earnings_flags(tickers: list[str], within_days: int = 5) -> dict[str, dict[str, Any]]:
    """Return only tickers with earnings in the next N days. Others are omitted."""

    def _check(t: str) -> dict[str, Any] | None:
        return earnings_within_days(t, within_days)

    results: dict[str, dict[str, Any]] = {}
    for ticker, r in yf_gateway.map_symbols(_check, tickers, workers=_MAX_WORKERS):
        if r:
            results[ticker] = r
    return results
