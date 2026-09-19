"""Slow-changing per-stock facts, cached in `ticker_reference`.

A stock's sector, industry and name almost never change, and its next
earnings date changes a few times a year, yet every daily email and
pipeline run looked them up again. They are now kept one row per ticker —
overwritten on refresh, so the table stays bounded by the number of stocks
ever seen — and refetched only when stale:

  - profile (name / sector / industry): after PROFILE_TTL_DAYS;
  - next earnings date: after EARNINGS_TTL_DAYS, or as soon as the stored
    date has passed (the company has reported; a new date is needed).

Reads happen first, the missing ones are fetched (in parallel where the
source allows), and all writes happen together at the end — so parallel
lookups never write to SQLite concurrently.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

from ..db.session import get_session
from ..db.tables import TickerReference
from ..logging import get_logger

logger = get_logger(__name__)

PROFILE_TTL_DAYS = 30
EARNINGS_TTL_DAYS = 3


def _fresh(stamp: str | None, ttl_days: int, today: date) -> bool:
    return bool(stamp) and date.fromisoformat(stamp) >= today - timedelta(days=ttl_days)


def _rows(db_path: str, tickers: list[str]) -> dict[str, TickerReference]:
    with get_session(db_path) as session:
        out = {}
        for t in tickers:
            row = session.get(TickerReference, t)
            if row is not None:
                session.expunge(row)
                out[t] = row
        return out


def _write(db_path: str, updates: dict[str, dict[str, Any]]) -> None:
    if not updates:
        return
    with get_session(db_path) as session:
        for t, fields in updates.items():
            row = session.get(TickerReference, t) or TickerReference(ticker=t)
            for k, v in fields.items():
                setattr(row, k, v)
            session.add(row)


def profiles(
    tickers: list[str],
    db_path: str,
    *,
    fetch: Callable[[list[str]], dict[str, dict[str, Any]]] | None = None,
    today: date | None = None,
) -> dict[str, dict[str, str | None]]:
    """{ticker: {name, sector, industry}} — cached, refreshed when stale.
    `fetch` defaults to fundamentals.batch_fundamentals."""
    today = today or date.today()
    rows = _rows(db_path, tickers)
    stale = [
        t
        for t in tickers
        if not (t in rows and _fresh(rows[t].profile_updated, PROFILE_TTL_DAYS, today))
    ]
    fetched: dict[str, dict[str, Any]] = {}
    if stale:
        if fetch is None:
            from .fundamentals import batch_fundamentals as fetch
        fetched = fetch(stale) or {}
        _write(
            db_path,
            {
                t: {
                    "name": f.get("name"),
                    "sector": f.get("sector"),
                    "industry": f.get("industry"),
                    "profile_updated": today.isoformat(),
                }
                for t, f in fetched.items()
                if f and (f.get("sector") or f.get("name"))
            },
        )
    out = {}
    for t in tickers:
        f = fetched.get(t)
        if f and (f.get("sector") or f.get("name")):
            out[t] = {
                "name": f.get("name"),
                "sector": f.get("sector"),
                "industry": f.get("industry"),
            }
        elif t in rows:  # stale but better than nothing when a refresh failed
            r = rows[t]
            out[t] = {"name": r.name, "sector": r.sector, "industry": r.industry}
    return out


def next_earnings_dates(
    tickers: list[str],
    db_path: str,
    *,
    fetch: Callable[[str], date | None] | None = None,
    today: date | None = None,
) -> dict[str, date | None]:
    """{ticker: next earnings date or None} — cached for EARNINGS_TTL_DAYS
    unless the stored date has already passed."""
    from . import yf_gateway

    today = today or date.today()
    if fetch is None:
        from .earnings_calendar import next_earnings_date as fetch
    rows = _rows(db_path, tickers)
    out: dict[str, date | None] = {}
    stale = []
    for t in tickers:
        r = rows.get(t)
        cached = date.fromisoformat(r.next_earnings) if r and r.next_earnings else None
        if (
            r
            and _fresh(r.earnings_updated, EARNINGS_TTL_DAYS, today)
            and (cached is None or cached >= today)
        ):
            out[t] = cached
        else:
            stale.append(t)
    fetched = dict(yf_gateway.map_symbols(fetch, stale, workers=4)) if stale else {}
    _write(
        db_path,
        {
            t: {
                "next_earnings": d.isoformat() if d else None,
                "earnings_updated": today.isoformat(),
            }
            for t, d in fetched.items()
        },
    )
    out.update(fetched)
    return out
