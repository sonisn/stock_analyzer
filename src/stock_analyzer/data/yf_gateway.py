"""Single chokepoint for every yfinance call in the codebase.

Yahoo has no published quota, but it throttles hard: once a run exceeds
roughly a couple of requests a second, it answers `Too Many Requests`
for a while and whole pipeline steps come back empty. That is exactly
what happened before this module existed — the discover workflow runs
`Parallel(...)` blocks in which six market-data steps (fundamentals,
technicals, eps_revisions, track_record, ...) each opened their own
4-10 thread pool, so 30+ requests could be in flight at once with no
global pacing anywhere.

Everything yfinance-related now goes through here, which gives the whole
process:

  - a hard cap on concurrent requests (`YF_MAX_CONCURRENCY`), shared by
    every module and every thread pool
  - AIMD pacing: requests leave at most `YF_RATE_LIMIT_PER_MIN` apart;
    a 429 halves the rate and parks every thread on a global cooldown,
    and sustained success walks the rate back up. Self-tuning beats a
    hardcoded interval because Yahoo's tolerance varies by time of day.
  - bounded retries around rate limits, so a throttled ticker comes back
    with data instead of a warning line
  - the same bounded retries around dropped connections (Yahoo closes
    sockets under load), which used to lose a ticker's earnings dates to
    a single SSL error
  - a shared `yf.Ticker` cache, so the four modules that each want
    `.info` for the same symbol cost one network round trip, not four
  - a negative cache for symbols Yahoo has no data for (money-market
    funds like SPAXX, spot-crypto ETFs like IBIT), so a dead symbol is
    looked up once per run instead of once per step
  - a shared daily-bars cache (`daily_bars`): the first module to want a
    symbol's adjusted daily history fetches two years of it, and every later
    window in the run (technicals, realized vol, track record, the daily
    email's trends) is sliced from that one download
  - yfinance's own chatty 404 logging demoted to DEBUG

Tuning knobs (env vars, all optional):
  YF_MAX_CONCURRENCY      max in-flight requests        (default 4)
  YF_RATE_LIMIT_PER_MIN   starting/ceiling request rate (default 150)
  YF_MIN_RATE_PER_MIN     floor the AIMD backoff stops at (default 20)
  YF_MAX_ATTEMPTS         tries per call before giving up (default 4)
  YF_COOLDOWN_SECONDS     base global pause after a 429  (default 15)
  YF_BATCH_WORKERS        threads a batch helper may use (default 8)
  YF_TICKER_CACHE_TTL     seconds a cached Ticker lives  (default 1800)
"""

from __future__ import annotations

import contextlib
import logging
import os
import random
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        logger.warning("%s=%r is not a number; using %d", name, raw, default)
        return default
    return value if value > 0 else default


_DEFAULTS = {
    "YF_MAX_CONCURRENCY": 4,
    "YF_RATE_LIMIT_PER_MIN": 150,
    "YF_MIN_RATE_PER_MIN": 20,
    "YF_MAX_ATTEMPTS": 4,
    "YF_COOLDOWN_SECONDS": 15,
    "YF_BATCH_WORKERS": 8,
    "YF_TICKER_CACHE_TTL": 1800,
}

MAX_CONCURRENCY = _env_int("YF_MAX_CONCURRENCY", _DEFAULTS["YF_MAX_CONCURRENCY"])
RATE_LIMIT_PER_MIN = _env_int("YF_RATE_LIMIT_PER_MIN", _DEFAULTS["YF_RATE_LIMIT_PER_MIN"])
MIN_RATE_PER_MIN = min(
    _env_int("YF_MIN_RATE_PER_MIN", _DEFAULTS["YF_MIN_RATE_PER_MIN"]), RATE_LIMIT_PER_MIN
)
MAX_ATTEMPTS = _env_int("YF_MAX_ATTEMPTS", _DEFAULTS["YF_MAX_ATTEMPTS"])
BASE_COOLDOWN_SECONDS = _env_int("YF_COOLDOWN_SECONDS", _DEFAULTS["YF_COOLDOWN_SECONDS"])
BATCH_WORKERS = _env_int("YF_BATCH_WORKERS", _DEFAULTS["YF_BATCH_WORKERS"])
TICKER_CACHE_TTL = _env_int("YF_TICKER_CACHE_TTL", _DEFAULTS["YF_TICKER_CACHE_TTL"])

# Rate recovery: after this many clean calls, give back a slice of the
# rate the last 429 took away. Slow on the way up, fast on the way down.
_RECOVERY_AFTER_SUCCESSES = 40
_RECOVERY_STEP_PER_MIN = 15
_MAX_COOLDOWN_SECONDS = 120.0

# Local pause before re-trying a dropped connection. Short and per-call:
# the process is not being throttled, one socket died.
_TRANSIENT_BACKOFF_SECONDS = 0.75


# --- error classification ---------------------------------------------------

_RATE_LIMIT_MARKERS = (
    "too many requests",
    "rate limited",
    "429",
)

# Yahoo's several ways of saying "this symbol isn't a thing I have data
# for". Retrying these is pure waste — the answer won't change today.
_MISSING_MARKERS = (
    "no fundamentals data found",
    "possibly delisted",
    "no price data found",
    "no data found",
    "symbol may be delisted",
    "404",
)


# Transport-level failures: the connection died, not the data. Yahoo
# drops connections under load (curl 35/28/56 via curl_cffi), and a plain
# second attempt almost always succeeds. Unlike a 429 these do not mean
# "you are going too fast", so they retry on a short local backoff without
# penalizing the whole process's rate.
_TRANSIENT_MARKERS = (
    "failed to perform",  # curl_cffi's prefix for every transport error
    "connection closed",
    "connection reset",
    "connection aborted",
    "connection refused",
    "connection error",
    "remote end closed",
    "incomplete read",
    "max retries exceeded",
    "ssl_connect",
    "ssl error",
    "sslerror",
    "handshake",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "bad gateway",
    "service unavailable",
    "gateway time-out",
    "502",
    "503",
    "504",
)


def _is_rate_limited(exc: BaseException) -> bool:
    from yfinance.exceptions import YFRateLimitError

    if isinstance(exc, YFRateLimitError):
        return True
    msg = str(exc).lower()
    return any(m in msg for m in _RATE_LIMIT_MARKERS)


def _is_missing_symbol(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _MISSING_MARKERS)


def _is_transient(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


# --- pacing -----------------------------------------------------------------


class _Pacer:
    """AIMD request pacer shared by every yfinance caller in the process.

    `acquire()` reserves the next send slot (so concurrent threads space
    themselves out rather than all sleeping the same interval and firing
    together) and blocks until it arrives, including any cooldown a
    recent 429 imposed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rate = float(RATE_LIMIT_PER_MIN)
        self._next_slot = 0.0
        self._cooldown_until = 0.0
        self._successes = 0
        self._consecutive_limits = 0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            interval = 60.0 / self._rate
            start = max(now, self._next_slot, self._cooldown_until)
            self._next_slot = start + interval
            wait = start - now
        if wait > 0:
            time.sleep(wait)

    def penalize(self) -> float:
        """Halve the rate and park every thread. Returns the cooldown.

        Concurrent workers hit the wall together, so a burst of N rate
        limits is one event, not N: while a cooldown is already running,
        further reports ride it out instead of halving the rate again
        (which would drop it to the floor on the first bad minute).
        """
        with self._lock:
            remaining = self._cooldown_until - time.monotonic()
            if remaining > 0:
                return remaining
            self._consecutive_limits += 1
            self._successes = 0
            self._rate = max(float(MIN_RATE_PER_MIN), self._rate / 2.0)
            cooldown = min(
                BASE_COOLDOWN_SECONDS * (2 ** (self._consecutive_limits - 1)),
                _MAX_COOLDOWN_SECONDS,
            )
            # Jitter so the blocked threads don't resume in lockstep and
            # re-trip the limit on the first burst.
            cooldown *= 1.0 + random.random() * 0.25
            resume_at = time.monotonic() + cooldown
            self._cooldown_until = max(self._cooldown_until, resume_at)
            self._next_slot = max(self._next_slot, self._cooldown_until)
            rate = self._rate
        logger.warning(
            "yfinance rate limited — pausing all fetches %.0fs, rate now %.0f/min",
            cooldown,
            rate,
        )
        return cooldown

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_limits = 0
            self._successes += 1
            if self._successes >= _RECOVERY_AFTER_SUCCESSES and self._rate < RATE_LIMIT_PER_MIN:
                self._rate = min(float(RATE_LIMIT_PER_MIN), self._rate + _RECOVERY_STEP_PER_MIN)
                self._successes = 0
                logger.info("yfinance recovered — rate now %.0f/min", self._rate)

    @property
    def rate_per_min(self) -> float:
        with self._lock:
            return self._rate

    def reset(self) -> None:
        with self._lock:
            self._rate = float(RATE_LIMIT_PER_MIN)
            self._next_slot = 0.0
            self._cooldown_until = 0.0
            self._successes = 0
            self._consecutive_limits = 0


_pacer = _Pacer()
_semaphore = threading.BoundedSemaphore(MAX_CONCURRENCY)


def reload_from_env() -> None:
    """Re-read the tunables and rebuild the pacer.

    Module import happens before the CLI's `load_dotenv()`, so without
    this the knobs would only ever pick up variables exported in the
    shell and everything in `.env` would be silently ignored. Entry
    points call this right after loading the environment.
    """
    global MAX_CONCURRENCY, RATE_LIMIT_PER_MIN, MIN_RATE_PER_MIN, MAX_ATTEMPTS
    global BASE_COOLDOWN_SECONDS, BATCH_WORKERS, TICKER_CACHE_TTL, _pacer, _semaphore

    MAX_CONCURRENCY = _env_int("YF_MAX_CONCURRENCY", _DEFAULTS["YF_MAX_CONCURRENCY"])
    RATE_LIMIT_PER_MIN = _env_int("YF_RATE_LIMIT_PER_MIN", _DEFAULTS["YF_RATE_LIMIT_PER_MIN"])
    MIN_RATE_PER_MIN = min(
        _env_int("YF_MIN_RATE_PER_MIN", _DEFAULTS["YF_MIN_RATE_PER_MIN"]), RATE_LIMIT_PER_MIN
    )
    MAX_ATTEMPTS = _env_int("YF_MAX_ATTEMPTS", _DEFAULTS["YF_MAX_ATTEMPTS"])
    BASE_COOLDOWN_SECONDS = _env_int("YF_COOLDOWN_SECONDS", _DEFAULTS["YF_COOLDOWN_SECONDS"])
    BATCH_WORKERS = _env_int("YF_BATCH_WORKERS", _DEFAULTS["YF_BATCH_WORKERS"])
    TICKER_CACHE_TTL = _env_int("YF_TICKER_CACHE_TTL", _DEFAULTS["YF_TICKER_CACHE_TTL"])

    _pacer = _Pacer()
    _semaphore = threading.BoundedSemaphore(MAX_CONCURRENCY)
    logger.debug(
        "yfinance pacing: %d concurrent, %d/min, %d attempts, %ds cooldown",
        MAX_CONCURRENCY,
        RATE_LIMIT_PER_MIN,
        MAX_ATTEMPTS,
        BASE_COOLDOWN_SECONDS,
    )


# --- caches -----------------------------------------------------------------

_ticker_lock = threading.Lock()
_tickers: dict[str, tuple[float, Any]] = {}

_missing_lock = threading.Lock()
_missing: dict[str, str] = {}

_stats_lock = threading.Lock()
_stats = {
    "calls": 0,
    "retries": 0,
    "rate_limited": 0,
    "transient": 0,
    "missing": 0,
    "failed": 0,
}


def _bump(key: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[key] += n


def stats() -> dict[str, int | float]:
    """Counters for an end-of-run summary line."""
    with _stats_lock:
        out: dict[str, int | float] = dict(_stats)
    out["rate_per_min"] = round(_pacer.rate_per_min, 1)
    out["cached_symbols"] = len(_tickers)
    return out


def log_stats(context: str = "") -> None:
    s = stats()
    logger.info(
        "yfinance%s: %d call(s), %d retried, %d rate-limit pause(s), "
        "%d dropped connection(s), %d unavailable symbol(s), %d failure(s); "
        "ending rate %.0f/min",
        f" [{context}]" if context else "",
        s["calls"],
        s["retries"],
        s["rate_limited"],
        s["transient"],
        s["missing"],
        s["failed"],
        s["rate_per_min"],
    )


def mark_unavailable(symbol: str, reason: str) -> None:
    """Record that Yahoo has no data for `symbol` — skip it from now on."""
    key = (symbol or "").upper()
    if not key:
        return
    with _missing_lock:
        first_time = key not in _missing
        if first_time:
            _missing[key] = reason
    if first_time:
        _bump("missing")
        logger.info(
            "yfinance has no data for %s (%s); skipping it for the rest of the run",
            key,
            reason,
        )


def is_unavailable(symbol: str) -> bool:
    with _missing_lock:
        return (symbol or "").upper() in _missing


def unavailable_symbols() -> dict[str, str]:
    with _missing_lock:
        return dict(_missing)


def get_ticker(symbol: str):
    """A process-wide cached `yf.Ticker`.

    yfinance memoizes `info`, `fast_info` and friends on the instance, so
    sharing one instance per symbol is what turns four modules asking for
    `.info` into a single round trip.
    """
    key = (symbol or "").upper()
    now = time.monotonic()
    with _ticker_lock:
        entry = _tickers.get(key)
        if entry is not None and now - entry[0] < TICKER_CACHE_TTL:
            return entry[1]
        ticker = yf.Ticker(symbol)
        _tickers[key] = (now, ticker)
        return ticker


def reset() -> None:
    """Drop caches and pacing state. For tests and long-lived processes."""
    with _ticker_lock:
        _tickers.clear()
    with _bars_lock:
        _bars.clear()
    with _missing_lock:
        _missing.clear()
    with _stats_lock:
        for k in _stats:
            _stats[k] = 0
    _pacer.reset()


# --- the call wrapper -------------------------------------------------------


def call[T](
    fn: Callable[..., T],
    *args: Any,
    symbol: str | None = None,
    what: str = "fetch",
    default: T | None = None,
    attempts: int | None = None,
    **kwargs: Any,
) -> T | None:
    """Run one yfinance operation under the global pacer + concurrency cap.

    Retries rate limits (with the whole process backing off behind it),
    never retries "symbol doesn't exist", and returns `default` instead of
    raising so callers keep their "missing data is fine" shape.
    """
    if symbol and is_unavailable(symbol):
        logger.debug("skipping %s for %s — known unavailable", what, symbol)
        return default

    max_attempts = attempts or MAX_ATTEMPTS
    for attempt in range(1, max_attempts + 1):
        penalized = False
        backoff = 0.0
        with _semaphore:
            _pacer.acquire()
            _bump("calls")
            try:
                result = fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — classified below
                if _is_rate_limited(e):
                    _bump("rate_limited")
                    if attempt >= max_attempts:
                        _bump("failed")
                        logger.warning(
                            "%s failed for %s after %d attempts: rate limited",
                            what,
                            symbol or "?",
                            attempt,
                        )
                        return default
                    _bump("retries")
                    penalized = True
                elif _is_missing_symbol(e):
                    if symbol:
                        mark_unavailable(symbol, str(e)[:120])
                    else:
                        logger.debug("%s: no data (%s)", what, e)
                    return default
                elif _is_transient(e):
                    _bump("transient")
                    if attempt >= max_attempts:
                        _bump("failed")
                        logger.warning(
                            "%s failed for %s after %d attempts: %s",
                            what,
                            symbol or "?",
                            attempt,
                            e,
                        )
                        return default
                    _bump("retries")
                    backoff = _TRANSIENT_BACKOFF_SECONDS * attempt + random.uniform(0, 0.25)
                    logger.debug(
                        "%s for %s: dropped connection (%s) — retry %d in %.1fs",
                        what,
                        symbol or "?",
                        e,
                        attempt + 1,
                        backoff,
                    )
                else:
                    _bump("failed")
                    logger.warning("%s failed for %s: %s", what, symbol or "?", e)
                    return default
            else:
                _pacer.record_success()
                return result
        # Outside the semaphore so a cooldown doesn't hold a slot hostage.
        if penalized:
            _pacer.penalize()
        elif backoff:
            time.sleep(backoff)
    return default


def ticker_call[T](
    symbol: str,
    what: str,
    fn: Callable[[Any], T],
    *,
    default: T | None = None,
    attempts: int | None = None,
) -> T | None:
    """`call` for the common shape: do something with a symbol's Ticker.

    info = ticker_call("AAPL", "info", lambda t: t.info, default={})
    """
    if is_unavailable(symbol):
        return default

    def _run() -> T:
        return fn(get_ticker(symbol))

    return call(_run, symbol=symbol, what=what, default=default, attempts=attempts)


def history(symbol: str, *, what: str = "history", **kwargs: Any):
    """Paced `Ticker.history`. Returns None on failure or empty result."""

    def _run(t: Any):
        return t.history(**kwargs)

    df = ticker_call(symbol, what, _run)
    if df is None or getattr(df, "empty", False):
        return None
    return df


# One download per symbol per run covers at least this much history, which
# is the longest window any routine caller asks for (technicals: 2y).
BARS_MIN_DAYS = 730

_bars_lock = threading.Lock()
# symbol -> (fetched at, first date the download asked for, frame)
_bars: dict[str, tuple[float, date, Any]] = {}
_bars_fetch_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)


def daily_bars(symbol: str, *, start: date, end: date | None = None, what: str = "daily_bars"):
    """Split- and dividend-adjusted daily bars from `start` through `end`
    (inclusive; default today), or None when Yahoo has nothing.

    The technicals, realized-vol, track-record and daily-email steps each
    wanted their own slice of the same history and fetched it separately.
    Now the first ask downloads at least `BARS_MIN_DAYS` and the rest are
    served from memory for `TICKER_CACHE_TTL`; an ask reaching further back
    than the cached download replaces it with a longer one.
    """
    key = (symbol or "").upper()
    today = date.today()
    fetch_from = min(start, today - timedelta(days=BARS_MIN_DAYS))
    # One download per symbol even when two steps ask at once.
    with _bars_lock:
        fetch_lock = _bars_fetch_locks[key]
    with fetch_lock:
        with _bars_lock:
            entry = _bars.get(key)
        fresh = entry is not None and time.monotonic() - entry[0] < TICKER_CACHE_TTL
        if fresh and entry is not None and entry[1] <= start:
            frame = entry[2]
        else:
            frame = history(symbol, what=what, start=fetch_from.isoformat(), auto_adjust=True)
            if frame is None:
                return None
            with _bars_lock:
                _bars[key] = (time.monotonic(), fetch_from, frame)
    return _slice_days(frame, start, end)


def _slice_days(frame: Any, start: date, end: date | None) -> Any:
    import pandas as pd

    if not isinstance(frame.index, pd.DatetimeIndex):
        return frame  # nothing to slice by
    tz = frame.index.tz
    mask = frame.index >= pd.Timestamp(start).tz_localize(tz)
    if end is not None:
        mask &= frame.index < pd.Timestamp(end + timedelta(days=1)).tz_localize(tz)
    out = frame[mask]
    return None if out.empty else out


def download(symbols: Iterable[str], *, what: str = "download", **kwargs: Any):
    """Paced `yf.download` — counts as one request against the pacer."""
    kwargs.setdefault("progress", False)
    return call(yf.download, list(symbols), what=what, **kwargs)


# --- bounded fan-out --------------------------------------------------------


def map_symbols[T](
    fn: Callable[[str], T],
    symbols: Iterable[str],
    *,
    workers: int | None = None,
) -> Iterator[tuple[str, T | None]]:
    """Apply `fn` across symbols with a bounded pool, yielding (symbol, result).

    Use this instead of a module-local ThreadPoolExecutor: real
    concurrency is capped by the global semaphore regardless of how many
    pipeline steps fan out at once, and the pool itself stays small so
    thousands of threads don't pile up behind it.

    One symbol's unexpected failure yields None for that symbol rather
    than aborting the batch — losing 500 names' data to one bad frame is
    not a trade worth making.
    """
    items = list(symbols)
    if not items:
        return

    def _guarded(symbol: str) -> T | None:
        try:
            return fn(symbol)
        except Exception as e:  # noqa: BLE001 — one bad symbol must not kill the batch
            logger.warning("batch fetch failed for %s: %s", symbol, e)
            return None

    n_workers = max(1, min(workers or BATCH_WORKERS, len(items)))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        yield from zip(items, ex.map(_guarded, items), strict=False)


# --- yfinance's own logging -------------------------------------------------


class _DemoteYFinanceNoise(logging.Handler):
    """Forward yfinance's own records into our logger at DEBUG.

    yfinance logs `HTTP Error 404: {"quoteSummary": ...}` at ERROR for
    every symbol it has no fundamentals for, and those land on logging's
    last-resort handler as bare unprefixed lines. They are not actionable
    — the gateway already reports an unavailable symbol once — so they
    belong in the log file at DEBUG, not on the operator's terminal.
    """

    def emit(self, record: logging.LogRecord) -> None:
        with contextlib.suppress(Exception):  # logging must never raise
            logger.debug("yfinance: %s", record.getMessage())


def _quiet_yfinance_logging() -> None:
    yf_logger = logging.getLogger("yfinance")
    if any(isinstance(h, _DemoteYFinanceNoise) for h in yf_logger.handlers):
        return
    yf_logger.handlers.clear()
    yf_logger.addHandler(_DemoteYFinanceNoise())
    yf_logger.propagate = False


_quiet_yfinance_logging()

# yfinance's built-in retry is off by default; leave it off — retrying
# inside the library would bypass this module's pacing and cooldown.
yf.config.network.retries = 0


__all__ = [
    "call",
    "download",
    "get_ticker",
    "history",
    "is_unavailable",
    "log_stats",
    "map_symbols",
    "mark_unavailable",
    "reload_from_env",
    "reset",
    "stats",
    "ticker_call",
    "unavailable_symbols",
]
