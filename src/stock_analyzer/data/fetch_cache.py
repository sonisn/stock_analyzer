"""Per-ticker Yahoo answers kept for a week, refreshed a slice a night.

Fundamentals and EPS revisions cost a Yahoo request per ticker each, and
with a 600-name screen fetching them fresh on every run was ~1,200
requests and most of a discover run (521s + 240s of 13 minutes on
2026-09-26). Neither moves in a way that matters for a 3-5 year decision
between earnings reports, so:

  - an answer is reused for up to FETCH_CACHE_DAYS (8);
  - it expires early once the company has reported since it was fetched
    (the answer carries its `next_earnings` date), so a quarter's new
    numbers are picked up the night after they come out;
  - the nightly job (cli/earnings_watch) refreshes the oldest fifth of
    the screen's names each night (`refresh_oldest`), so the week's
    refetch is spread over five nights instead of landing on one;
  - price-dependent fields are brought up to today's close by the caller
    (fundamentals.reprice), not refetched.

A run then asks Yahoo only for names it has never seen, and a night asks
for ~120 plus that day's reporters. One small JSON file per kind under
`~/.stock_analyzer/cache/`, latest answer per ticker; expired answers are
dropped on every write. Failures (None) are not stored.
`FETCH_CACHE_DAYS=0` turns it off. Writes go through a temp file and a
rename, so two processes can't leave half a file; the later writer wins.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

DEFAULT_DIR = "~/.stock_analyzer/cache"
# A day over the week the nightly rotation takes, so a name renewed last
# Monday night is still fresh until this Monday night's renewal.
DEFAULT_DAYS = 8.0
# The nightly job runs five nights a week; a fifth a night is the week.
NIGHTLY_SHARE = 0.2
_LOCK = threading.Lock()

Entries = dict[str, dict[str, Any]]


def max_age_seconds() -> float:
    raw = os.getenv("FETCH_CACHE_DAYS", "").strip()
    try:
        days = float(raw) if raw else DEFAULT_DAYS
    except ValueError:
        days = DEFAULT_DAYS
    return max(days, 0.0) * 86400


def _path(kind: str) -> Path | None:
    raw = os.getenv("FETCH_CACHE_DIR", DEFAULT_DIR).strip()
    if not raw or raw.lower() in {"off", "0", "false", "none"} or max_age_seconds() <= 0:
        return None
    return Path(os.path.expanduser(raw)) / f"{kind}.json"


def reported_since_fetch(entry: dict[str, Any], today: date) -> bool:
    """True when the earnings date the answer was fetched ahead of has passed."""
    nxt = (entry.get("value") or {}).get("next_earnings")
    fetched = date.fromtimestamp(float(entry.get("at") or 0))
    try:
        when = date.fromisoformat(nxt) if nxt else None
    except ValueError:
        return False
    return when is not None and fetched <= when < today


def _fresh(entry: Any, now: float) -> bool:
    return (
        isinstance(entry, dict)
        and float(entry.get("at") or 0) >= now - max_age_seconds()
        and not reported_since_fetch(entry, date.fromtimestamp(now))
    )


def _load(path: Path, now: float) -> Entries:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        logger.warning("Fetch cache %s unreadable (%s) — starting fresh", path.name, e)
        return {}
    return {t: e for t, e in (raw or {}).items() if _fresh(e, now)}


def _save(path: Path, entries: Entries) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(entries, separators=(",", ":")))
    os.replace(tmp, path)


def entries(kind: str) -> Entries:
    """The unexpired answers of one kind, {ticker: {"at", "value"}}."""
    path = _path(kind)
    if path is None:
        return {}
    with _LOCK:
        return _load(path, time.time())


def fetch_many(
    kind: str,
    tickers: Iterable[str],
    fetch: Callable[[list[str]], Iterator[tuple[str, Any]]],
    *,
    refresh: Iterable[str] = (),
) -> dict[str, Any]:
    """{ticker: answer} for every ticker `fetch` (or the cache) answered.
    `fetch` gets only the tickers without a fresh cached answer, plus any in
    `refresh`, and yields (ticker, answer-or-None)."""
    wanted = list(dict.fromkeys(tickers))
    path = _path(kind)
    if path is None:
        return {t: r for t, r in fetch(wanted) if r}
    now = time.time()
    with _LOCK:
        cached = _load(path, now)
    again = set(refresh)
    out = {t: cached[t]["value"] for t in wanted if t in cached and t not in again}
    todo = [t for t in wanted if t not in out]
    fresh = {t: r for t, r in fetch(todo) if r} if todo else {}
    for t in todo:  # a failed refresh keeps last week's answer rather than nothing
        if t not in fresh and t in cached:
            out[t] = cached[t]["value"]
    out.update(fresh)
    logger.info(
        "%s: %d/%d from the cache, %d fetched",
        kind,
        len(wanted) - len(todo),
        len(wanted),
        len(todo),
    )
    if fresh:
        with _LOCK:
            try:
                # Re-read: another run may have written since we loaded.
                stored = _load(path, time.time())
                stored.update({t: {"at": now, "value": r} for t, r in fresh.items()})
                _save(path, stored)
            except OSError as e:  # a cache that cannot write is only slower
                logger.warning("Fetch cache %s not written (%s)", path.name, e)
    return out


def oldest(kind: str, tickers: Iterable[str], share: float = NIGHTLY_SHARE) -> list[str]:
    """The `share` of `tickers` with the oldest cached answers (uncached
    ones are fetched anyway and don't count). The nightly job refreshes
    these, so every name's answer is renewed about once a week."""
    wanted = list(dict.fromkeys(tickers))
    cached = entries(kind)
    have = sorted((t for t in wanted if t in cached), key=lambda t: float(cached[t]["at"]))
    return have[: math.ceil(len(wanted) * share)]
