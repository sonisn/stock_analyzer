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
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..logging import get_logger

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
    frame: pd.DataFrame
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
        frame = table.to_pandas()
        return StoredBars(
            frame=frame,
            requested_from=date.fromisoformat(meta[_META_FROM].decode()),
            checked_at=float(meta[_META_CHECKED].decode()),
        )
    except Exception as e:  # noqa: BLE001 — a bad file is a cache miss
        logger.warning("Bar store: unreadable %s (%s) — refetching", path.name, e)
        return None


def save(
    symbol: str, frame: pd.DataFrame, requested_from: date, *, checked_at: float | None = None
) -> None:
    path = _path(symbol)
    if path is None or frame is None or frame.empty:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(frame, preserve_index=True)
        meta = dict(table.schema.metadata or {})
        meta[_META_FROM] = requested_from.isoformat().encode()
        meta[_META_CHECKED] = repr(checked_at if checked_at is not None else time.time()).encode()
        table = table.replace_schema_metadata(meta)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except Exception as e:  # noqa: BLE001 — failing to cache must not fail the read
        logger.warning("Bar store: could not write %s (%s)", path.name, e)


def overlap_start(stored: pd.DataFrame) -> date:
    """First day a sync asks for: a week before the last stored bar."""
    # Bar stamps are midnight in the exchange's zone, so the text's date part
    # is the trading day (a NaT-free path to a plain `date`).
    last = date.fromisoformat(str(stored.index[-1])[:10])
    return last - timedelta(days=OVERLAP_DAYS)


def needs_full_refetch(stored: pd.DataFrame, delta: pd.DataFrame) -> str | None:
    """Why the stored history can no longer be extended, or None if it can."""
    last = stored.index[-1]
    new_days = delta[delta.index > last]
    for col in _EVENT_COLUMNS:
        if col in new_days and (new_days[col].fillna(0) != 0).any():
            return f"new {col.lower()}"
    # Complete bars both frames have; the last stored bar may be partial.
    common = stored.index[:-1].intersection(delta.index)
    if len(common) and "Close" in stored and "Close" in delta:
        old = stored.loc[common, "Close"].astype(float)
        new = delta.loc[common, "Close"].astype(float)
        rel = ((new - old).abs() / old.abs().where(old != 0)).fillna(0)
        if (rel > RESTATED_REL_TOL).any():
            return "restated closes"
    return None


def extend(stored: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    """Stored bars before the delta's first day, then the delta."""
    if delta is None or delta.empty:
        return stored
    head = stored[stored.index < delta.index[0]]
    if head.empty:
        return delta
    merged = pd.concat([head, delta.reindex(columns=head.columns.union(delta.columns))])
    return merged[~merged.index.duplicated(keep="last")].sort_index()


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


def trim(frame: pd.DataFrame, start: date) -> pd.DataFrame:
    """Bars from `start` on, whatever the index's timezone."""
    idx: Any = frame.index
    if not isinstance(idx, pd.DatetimeIndex):
        return frame
    return frame[idx >= pd.Timestamp(start).tz_localize(idx.tz)]
