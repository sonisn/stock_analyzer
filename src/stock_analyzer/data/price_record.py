"""A permanent record of what things were worth, one close per day.

Advice is only gradeable against the price on the day it was given. This
system fetched such prices constantly and kept none of them, so "what was
AVGO worth when we said sell it?" needed a network call and a hope that
the vendor still agreed with itself. Worse, a grade computed that way is
not reproducible: run it twice a week apart and the history underneath
may have been restated.

So prices are written down. `record_prices` appends today's closes;
`stored_history` hands the existing graders a frame in the shape they
already expect, which is what lets `reporting.quarterly.grade_suggestions`
run offline against the record instead of the internet.

Append-only and small: sixteen tickers for a year is roughly four
thousand rows.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from ..db.session import get_session
from ..db.tables import TickerPrice
from ..logging import get_logger

logger = get_logger(__name__)


def store_close(session: Session, ticker: str, day: str, close: float) -> bool:
    """Write one close. Last write wins within a day; returns True when the
    row is new, so a caller can report how much was actually added."""
    row = session.get(TickerPrice, (ticker.upper(), day))
    if row is None:
        session.add(TickerPrice(ticker=ticker.upper(), day=day, close=float(close)))
        return True
    row.close = float(close)
    return False


def missing_today(db_path: str, tickers: list[str], *, today: date | None = None) -> list[str]:
    """Which tickers have no close stored for `today` — the whole point of
    the record is that a second run the same day asks the network nothing."""
    day = (today or date.today()).isoformat()
    wanted = {t.upper() for t in tickers if t}
    if not wanted:
        return []
    with get_session(db_path) as session:
        have = {
            r for r in session.exec(select(TickerPrice.ticker).where(TickerPrice.day == day)).all()
        }
    return sorted(wanted - have)


def record_prices(
    db_path: str,
    tickers: list[str],
    *,
    today: date | None = None,
    fetch: Any = None,
) -> dict[str, float]:
    """Store today's close for any ticker missing one. Returns what was
    stored. Never raises: a missing price is a gap in the record, not a
    reason to fail the run that called it."""
    day = (today or date.today()).isoformat()
    todo = missing_today(db_path, tickers, today=today)
    if not todo:
        logger.info("Price record: already complete for %s", day)
        return {}
    if fetch is None:
        from . import yf_gateway

        def fetch(symbols: list[str]) -> dict[str, float]:
            frame = yf_gateway.download(
                symbols, what="price.record", period="5d", auto_adjust=True, group_by="column"
            )
            if frame is None or frame.empty:
                return {}
            closes = frame.get("Close", frame)
            out: dict[str, float] = {}
            for sym in symbols:
                if sym in closes:
                    series = closes[sym].dropna()
                    if not series.empty:
                        out[sym] = float(series.iloc[-1])
            return out

    try:
        prices = fetch(todo)
    except Exception as e:  # noqa: BLE001 — a gap beats a failed run
        logger.warning("Price record: fetch failed (%s) — %d ticker(s) unrecorded", e, len(todo))
        return {}
    added = 0
    with get_session(db_path) as session:
        for ticker, close in prices.items():
            if close and close > 0:
                added += store_close(session, ticker, day, close)
        session.commit()
    logger.info(
        "Price record: %d new close(s) for %s (%d requested, %d returned)",
        added,
        day,
        len(todo),
        len(prices),
    )
    return prices


def stored_history(db_path: str) -> Any:
    """A `fetch(ticker, start, end)` backed by the record, shaped like the
    yfinance frame the graders already take — so they work unchanged, and
    offline."""

    def fetch(ticker: str, start: date, end: date) -> pd.DataFrame | None:
        with get_session(db_path) as session:
            rows = session.exec(
                select(TickerPrice.day, TickerPrice.close)
                .where(
                    TickerPrice.ticker == ticker.upper(),
                    TickerPrice.day >= start.isoformat(),
                    TickerPrice.day <= end.isoformat(),
                )
                .order_by(TickerPrice.day)
            ).all()
        if not rows:
            return None
        return pd.DataFrame(
            {"Close": [c for _, c in rows]},
            index=pd.DatetimeIndex([pd.Timestamp(d) for d, _ in rows]),
        )

    return fetch


def coverage(db_path: str) -> dict[str, Any]:
    """How much record there is, for the dashboard's own honesty."""
    with get_session(db_path) as session:
        rows = session.exec(select(TickerPrice.ticker, TickerPrice.day)).all()
    if not rows:
        return {"rows": 0, "tickers": 0, "first": None, "last": None}
    days = sorted({d for _, d in rows})
    return {
        "rows": len(rows),
        "tickers": len({t for t, _ in rows}),
        "first": days[0],
        "last": days[-1],
    }


def backfill_from_panel(db_path: str, tickers: list[str], *, days: int = 400) -> int:
    """Seed the record from the on-disk bar store so grading has history
    from day one instead of starting blind. Store-only — no network."""
    from . import bar_store

    added = 0
    start = (date.today() - timedelta(days=days)).isoformat()
    with get_session(db_path) as session:
        for ticker in {t.upper() for t in tickers} | {"SPY"}:
            stored = bar_store.load(ticker)
            if stored is None or "Close" not in stored.frame:
                continue
            for stamp, value in stored.frame["Close"].dropna().items():
                day = pd.Timestamp(stamp).date().isoformat()
                if day >= start and value and value > 0:
                    added += store_close(session, ticker, day, float(value))
        session.commit()
    logger.info("Price record: backfilled %d close(s) from the bar store", added)
    return added
