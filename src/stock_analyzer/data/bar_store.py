"""Adjusted daily bars kept on disk, so a run downloads only what is new.

Every run used to download two years of daily history per symbol, and the
model's training panel re-downloaded up to fifteen years of ~500 symbols
whenever its pickle was a day old. Most of those bytes had not changed
since the last run.

Each symbol's split- and dividend-adjusted bars now live in one Parquet
file. A sync asks Yahoo only for the days since the last stored bar (plus
a short overlap) and appends them.

The catch is that adjusted prices are rewritten backwards: a dividend or
a split rescales every earlier bar, so appending to the old history would
silently mix two adjustments. A sync therefore replaces the symbol's whole
history when either
  - the new days carry a dividend or a split, or
  - an overlapping bar no longer matches what was stored (a restatement
    the event columns did not announce).
The last stored bar is left out of that comparison: it may have been
written mid-session, and its close is expected to change.

Frames are Polars, in the package's bar shape (data/frames.py: a `date`
column plus yfinance's column names). Files written before the switch hold
the pandas layout (the index saved as a timezone-aware `Date`) and are read
through the same conversion, so the existing cache stays valid.

Files are written to a temp name and renamed, so a reader in another
process (two cron jobs overlapping) sees the old file or the new one,
never half of one. `YF_BARS_DIR` moves the store; `YF_BARS_DIR=off`
turns it off, and every read then goes to Yahoo as before.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
import pyarrow.parquet as pq

from ..logging import get_logger
from . import frames
from .frames import DATE

logger = get_logger(__name__)

DEFAULT_DIR = "~/.stock_analyzer/cache/bars"
# Stored bars re-requested on each sync. A week covers a long weekend plus
# a holiday, so a partial bar written mid-session is always refetched.
OVERLAP_DAYS = 7
# Relative difference at which a stored close counts as restated. Yahoo's
# adjusted closes round-trip to ~1e-9; a real dividend adjustment is 1e-3+.
RESTATED_REL_TOL = 1e-5
_EVENT_COLUMNS = ("Dividends", "Stock Splits")
_META_FROM = b"stock_analyzer.requested_from"
_META_CHECKED = b"stock_analyzer.checked_at"


@dataclass
class StoredBars:
    frame: pl.DataFrame
    # The start the stored download asked for. A symbol younger than that
    # has a first bar later than it, which is still complete coverage.
    requested_from: date
    # Wall-clock time of the last successful sync with Yahoo.
    checked_at: float


def store_dir() -> Path | None:
    raw = os.getenv("YF_BARS_DIR", DEFAULT_DIR).strip()
    if not raw or raw.lower() in {"off", "0", "false", "none"}:
        return None
    return Path(os.path.expanduser(raw))


def _path(symbol: str) -> Path | None:
    root = store_dir()
    if root is None:
        return None
    # Symbols like BRK-B are file-safe; guard against a stray slash anyway.
    return root / f"{symbol.upper().replace('/', '_')}.parquet"


def load(symbol: str) -> StoredBars | None:
    path = _path(symbol)
    if path is None or not path.exists():
        return None
    try:
        table = pq.read_table(path)
        meta = table.schema.metadata or {}
        frame = frames.from_parquet_table(pl.from_arrow(table))  # ty: ignore[invalid-argument-type]
        return StoredBars(
            frame=frame,
            requested_from=date.fromisoformat(meta[_META_FROM].decode()),
            checked_at=float(meta[_META_CHECKED].decode()),
        )
    except Exception as e:  # noqa: BLE001 — a bad file is a cache miss
        logger.warning("Bar store: unreadable %s (%s) — refetching", path.name, e)
        return None


def save(
    symbol: str, frame: pl.DataFrame, requested_from: date, *, checked_at: float | None = None
) -> None:
    path = _path(symbol)
    if path is None or frame is None or frame.is_empty():
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = frame.to_arrow()
        meta = dict(table.schema.metadata or {})
        meta[_META_FROM] = requested_from.isoformat().encode()
        meta[_META_CHECKED] = repr(checked_at if checked_at is not None else time.time()).encode()
        table = table.replace_schema_metadata(meta)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 — failing to cache must not fail the read
        logger.warning("Bar store: could not write %s (%s)", path.name, e)


def last_day(stored: pl.DataFrame) -> date:
    return stored[DATE][-1]


def overlap_start(stored: pl.DataFrame) -> date:
    """First day a sync asks for: a week before the last stored bar."""
    return last_day(stored) - timedelta(days=OVERLAP_DAYS)


def _nonzero(col: pl.Expr) -> pl.Expr:
    return col.fill_nan(0.0).fill_null(0.0) != 0


def needs_full_refetch(stored: pl.DataFrame, delta: pl.DataFrame) -> str | None:
    """Why the stored history can no longer be extended, or None if it can."""
    last = last_day(stored)
    new_days = delta.filter(pl.col(DATE) > last)
    for col in _EVENT_COLUMNS:
        if col in new_days.columns and new_days.select(_nonzero(pl.col(col)).any()).item():
            return f"new {col.lower()}"
    # Complete bars both frames have; the last stored bar may be partial.
    if "Close" in stored.columns and "Close" in delta.columns:
        both = (
            stored.head(stored.height - 1)
            .select(DATE, pl.col("Close").alias("old"))
            .join(delta.select(DATE, pl.col("Close").alias("new")), on=DATE, how="inner")
        )
        if both.height:
            rel = ((pl.col("new") - pl.col("old")).abs() / pl.col("old").abs()).fill_nan(0.0)
            rel = pl.when(pl.col("old") == 0).then(0.0).otherwise(rel).fill_null(0.0)
            if both.select((rel > RESTATED_REL_TOL).any()).item():
                return "restated closes"
    return None


def extend(stored: pl.DataFrame, delta: pl.DataFrame | None) -> pl.DataFrame:
    """Stored bars before the delta's first day, then the delta."""
    if delta is None or delta.is_empty():
        return stored
    head = stored.filter(pl.col(DATE) < delta[DATE][0])
    if head.is_empty():
        return delta
    return frames.normalize(pl.concat([head, delta], how="diagonal_relaxed"))


# US equity session, New York time. Bars are treated as final this long
# after the 16:00 close, once Yahoo has settled the day's last print.
_EXCHANGE_TZ = ZoneInfo("America/New_York")
_SESSION_OPEN = dtime(9, 30)
_BARS_FINAL = dtime(16, 30)


def last_final_close(now: datetime) -> datetime:
    """The most recent weekday `_BARS_FINAL` at or before `now`. Holidays
    are not known here; on one, a sync simply finds nothing new."""
    local = now.astimezone(_EXCHANGE_TZ)
    day = local.date()
    if local.time() < _BARS_FINAL:
        day -= timedelta(days=1)
    while day.weekday() >= 5:
        day -= timedelta(days=1)
    return datetime.combine(day, _BARS_FINAL, tzinfo=_EXCHANGE_TZ)


def in_session(now: datetime) -> bool:
    local = now.astimezone(_EXCHANGE_TZ)
    return local.weekday() < 5 and _SESSION_OPEN <= local.time() < _BARS_FINAL


def is_current(stored: StoredBars, max_age_seconds: float, *, now: float | None = None) -> bool:
    """Whether the stored bars are as new as a download would be.

    Outside market hours nothing changes until the next close, so bars
    synced after the last close are served with no request at all — an
    evening, weekend or second run the same night asks Yahoo nothing.
    During the session today's bar is still moving; it is refetched once
    the sync is `max_age_seconds` old, as the in-memory cache always did.
    """
    now_s = time.time() if now is None else now
    if now_s - stored.checked_at < max_age_seconds:
        return True
    at = datetime.fromtimestamp(now_s, _EXCHANGE_TZ)
    if in_session(at):
        return False
    return stored.checked_at >= last_final_close(at).timestamp()


def trim(frame: pl.DataFrame, start: date) -> pl.DataFrame:
    """Bars from `start` on."""
    return frames.since(frame, start)
