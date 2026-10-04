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

from datetime import UTC, date, datetime, timedelta
from typing import Any

import polars as pl
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.session import get_session
from ..db.tables import Suggestion, TickerPrice
from ..logging import get_logger
from . import frames
from .frames import DATE

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
            r
            for r in session.scalars(select(TickerPrice.ticker).where(TickerPrice.day == day)).all()
        }
    return sorted(wanted - have)


def record_tickers(db_path: str, held: list[str]) -> list[str]:
    """What the record keeps a close for: the holdings, every ticker a
    suggestion named (and its reinvestment), and SPY to grade against."""
    with get_session(db_path) as session:
        named = session.execute(select(Suggestion.ticker, Suggestion.reinvest_into)).all()
    return sorted(set(held) | {t for t, _ in named} | {r for _, r in named if r} | {"SPY"})


def _closed(day: str, now: datetime | None) -> bool:
    """Whether `day`'s close is final (bar_store's 4:30 PM New York rule)."""
    from .bar_store import last_final_close

    return last_final_close(now or datetime.now(UTC)).date().isoformat() == day


def record_prices(
    db_path: str,
    tickers: list[str],
    *,
    today: date | None = None,
    fetch: Any = None,
    now: datetime | None = None,
) -> dict[str, float]:
    """Store today's close for any ticker missing one, once it is final. Returns what was
    stored. Never raises: a missing price is a gap in the record, not a
    reason to fail the run that called it."""
    day = (today or date.today()).isoformat()
    if fetch is None and not _closed(day, now):
        # Before 4:30 PM New York time (or on a weekend) the latest close is
        # an earlier session's; stored under `day` it would read as today's
        # and block the real close from being recorded after the bell.
        logger.info("Price record: %s has no final close yet — nothing recorded", day)
        return {}
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
                    # yfinance's pandas frame crosses into the bar shape here.
                    bars = frames.closes(
                        frames.bars_from_pandas(closes[[sym]].rename(columns={sym: "Close"}))
                    )
                    if bars is not None and bars.height:  # a symbol with no data is skipped
                        out[sym] = float(bars["Close"][-1])
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

    def fetch(ticker: str, start: date, end: date) -> pl.DataFrame | None:
        with get_session(db_path) as session:
            rows = session.execute(
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
        return pl.DataFrame(
            {DATE: [date.fromisoformat(d) for d, _ in rows], "Close": [float(c) for _, c in rows]}
        )

    return fetch


def coverage(db_path: str) -> dict[str, Any]:
    """How much record there is, for the dashboard's own honesty."""
    with get_session(db_path) as session:
        rows = session.execute(select(TickerPrice.ticker, TickerPrice.day)).all()
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
    """Fill gaps in the record from the on-disk bar store — the days a run
    missed, or history before the record began. Store-only, no network,
    and it never overwrites a close already recorded."""
    from . import bar_store

    added = 0
    start = (date.today() - timedelta(days=days)).isoformat()
    with get_session(db_path) as session:
        for ticker in {t.upper() for t in tickers} | {"SPY"}:
            stored = bar_store.load(ticker)
            closes = frames.closes(stored.frame) if stored is not None else None
            if closes is None:
                continue
            have = set(
                session.scalars(
                    select(TickerPrice.day).where(TickerPrice.ticker == ticker.upper())
                ).all()
            )
            for day, value in closes.iter_rows():
                d = day.isoformat()
                if d >= start and d not in have and value and value > 0:
                    added += store_close(session, ticker, d, float(value))
        session.commit()
    logger.info("Price record: backfilled %d close(s) from the bar store", added)
    return added
