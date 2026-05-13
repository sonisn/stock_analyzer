"""Track-record measurement — close the feedback loop.

Reads past PICKs out of `discover.db`, fetches forward prices via
yfinance, computes:

  - per-pick realized return from pick date to today (or to a 90-day
    cap, whichever is shorter)
  - SPY benchmark return over the same window
  - alpha vs SPY (return - benchmark)
  - aggregate win rate (% of mature picks that beat SPY)

Mature pick = at least `_MIN_AGE_DAYS` old. Newer picks are listed
separately as "pending" with their live return so the user sees them
without their noise being counted in the stats.

Surfaced two ways:
  1. As a header section in the email + PDF report ("Track record:
     23 mature picks, mean +6.4% vs SPY +2.1%, 13 winners / 10 losers")
  2. As context in the Opus ranker / rebalancer prompts so the LLM
     can reason about its own historical accuracy
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import yfinance as yf
from pydantic import BaseModel, ConfigDict

from ..logging import get_logger

logger = get_logger(__name__)

# Picks newer than this aren't counted in aggregate stats — their realized
# return is dominated by noise rather than signal. Listed as "pending"
# separately.
_MIN_AGE_DAYS = 14
# Cap how far forward we measure: 90 days matches the user's chosen
# evaluation window. Picks older than 90 days are measured to their 90d
# anniversary, not to today, so all mature picks are on the same yardstick.
_MEASUREMENT_WINDOW_DAYS = 90
# yfinance batch size for parallel ticker fetches.
_MAX_WORKERS = 6


class PickReturn(BaseModel):
    """One scored pick — its realized return and how it compared to SPY."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    pick_date: str          # ISO yyyy-mm-dd
    age_days: int
    pick_price: float | None
    measured_price: float | None
    pick_return_pct: float | None
    spy_return_pct: float | None
    alpha_pct: float | None  # pick_return - spy_return
    is_mature: bool          # >= _MIN_AGE_DAYS old


class TrackRecord(BaseModel):
    """Aggregate summary of mature picks over the lookback window."""

    model_config = ConfigDict(frozen=True)

    n_picks_total: int
    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int       # mature picks where alpha > 0
    losers: int        # mature picks where alpha < 0
    flats: int         # mature picks where alpha ≈ 0
    picks: list[PickReturn]
    pending: list[PickReturn]


# --- DB read ---------------------------------------------------------------


def _fetch_recent_picks(
    conn: sqlite3.Connection, *, lookback_days: int
) -> list[tuple[str, str, int]]:
    """Return [(ticker, pick_date, age_days), ...] for every PICK from any
    run in the last `lookback_days`, deduplicated to the OLDEST occurrence
    per ticker (so re-picks don't double-count)."""
    cur = conn.execute(
        "SELECT runs.run_at, picks.ticker FROM picks "
        "JOIN runs ON runs.id = picks.run_id "
        "WHERE runs.run_at >= ? "
        "ORDER BY runs.run_at ASC",
        ((datetime.now() - timedelta(days=lookback_days)).isoformat(),),
    )
    rows = list(cur.fetchall())
    oldest_by_ticker: dict[str, str] = {}
    for run_at, ticker in rows:
        if ticker not in oldest_by_ticker:
            oldest_by_ticker[ticker] = run_at
    today = date.today()
    out: list[tuple[str, str, int]] = []
    for ticker, run_at in oldest_by_ticker.items():
        try:
            pick_date = datetime.fromisoformat(run_at).date()
        except ValueError:
            continue
        age = (today - pick_date).days
        out.append((ticker, pick_date.isoformat(), age))
    return sorted(out, key=lambda x: x[1])  # oldest first


# --- yfinance price fetch --------------------------------------------------


@dataclass
class _Quote:
    pick_price: float | None
    measured_price: float | None


def _fetch_quote(
    ticker: str, pick_date: str, age_days: int
) -> _Quote:
    """Pick-date close and measurement-date close from yfinance.

    Measurement date = min(pick_date + 90d, today). For picks younger than
    90d we measure to today, so the "pending" entries show live returns;
    for picks older than 90d we cap at +90d so the metric stays
    apples-to-apples across vintages.
    """
    try:
        start = datetime.fromisoformat(pick_date).date()
        end = min(start + timedelta(days=_MEASUREMENT_WINDOW_DAYS), date.today())
        if end <= start:
            return _Quote(None, None)
        hist = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
        if hist.empty:
            return _Quote(None, None)
        # First and last close in the window.
        return _Quote(
            pick_price=float(hist["Close"].iloc[0]),
            measured_price=float(hist["Close"].iloc[-1]),
        )
    except Exception as e:
        logger.debug("yfinance fetch failed for %s: %s", ticker, e)
        return _Quote(None, None)


def _fetch_spy_quote(pick_date: str) -> _Quote:
    return _fetch_quote("SPY", pick_date, age_days=0)


# --- aggregation ----------------------------------------------------------


def _pct_change(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * 100


def _score_pick(
    ticker: str, pick_date: str, age_days: int, spy_quote: _Quote
) -> PickReturn:
    quote = _fetch_quote(ticker, pick_date, age_days)
    pick_ret = _pct_change(quote.pick_price, quote.measured_price)
    spy_ret = _pct_change(spy_quote.pick_price, spy_quote.measured_price)
    alpha = (pick_ret - spy_ret) if (pick_ret is not None and spy_ret is not None) else None
    return PickReturn(
        ticker=ticker,
        pick_date=pick_date,
        age_days=age_days,
        pick_price=quote.pick_price,
        measured_price=quote.measured_price,
        pick_return_pct=pick_ret,
        spy_return_pct=spy_ret,
        alpha_pct=alpha,
        is_mature=age_days >= _MIN_AGE_DAYS,
    )


def measure_track_record(
    db_path: str, *, lookback_days: int = 180
) -> TrackRecord:
    """Top-level entry — query past picks, fetch prices, summarize.

    `lookback_days` bounds how far back we look in the DB; defaults to
    180 so we always have enough mature picks to compute meaningful
    stats once the system has been running a while. Empty TrackRecord
    is returned if the DB has no picks yet.
    """
    from .persistence import connect

    try:
        with connect(db_path) as conn:
            raw = _fetch_recent_picks(conn, lookback_days=lookback_days)
    except Exception as e:
        logger.warning("track-record fetch failed (%s) — returning empty", e)
        return _empty_record(0)
    if not raw:
        return _empty_record(0)

    # SPY benchmarks — one per distinct pick_date so we don't refetch.
    distinct_dates = sorted({pick_date for _, pick_date, _ in raw})
    spy_cache: dict[str, _Quote] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for d, q in zip(distinct_dates, ex.map(_fetch_spy_quote, distinct_dates)):
            spy_cache[d] = q

    picks: list[PickReturn] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = [
            ex.submit(_score_pick, t, d, a, spy_cache.get(d, _Quote(None, None)))
            for t, d, a in raw
        ]
        for fut in futures:
            try:
                picks.append(fut.result())
            except Exception as e:
                logger.debug("pick scoring failed: %s", e)

    mature = [p for p in picks if p.is_mature and p.alpha_pct is not None]
    pending = [p for p in picks if not p.is_mature or p.alpha_pct is None]

    if mature:
        mean_ret = sum(p.pick_return_pct or 0 for p in mature) / len(mature)
        mean_spy = sum(p.spy_return_pct or 0 for p in mature) / len(mature)
        mean_alpha = sum(p.alpha_pct or 0 for p in mature) / len(mature)
        winners = sum(1 for p in mature if (p.alpha_pct or 0) > 0.5)
        losers = sum(1 for p in mature if (p.alpha_pct or 0) < -0.5)
        flats = len(mature) - winners - losers
    else:
        mean_ret = mean_spy = mean_alpha = None
        winners = losers = flats = 0

    record = TrackRecord(
        n_picks_total=len(picks),
        n_mature=len(mature),
        n_pending=len(pending),
        mean_return_pct=mean_ret,
        mean_spy_return_pct=mean_spy,
        mean_alpha_pct=mean_alpha,
        winners=winners,
        losers=losers,
        flats=flats,
        picks=mature,
        pending=pending,
    )
    logger.info(
        "Track record: %d mature picks, mean %s%% vs SPY %s%% "
        "(%d winners / %d losers / %d flat), %d pending",
        record.n_mature,
        f"{mean_ret:+.1f}" if mean_ret is not None else "—",
        f"{mean_spy:+.1f}" if mean_spy is not None else "—",
        winners, losers, flats, record.n_pending,
    )
    return record


def _empty_record(_) -> TrackRecord:
    return TrackRecord(
        n_picks_total=0, n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, picks=[], pending=[],
    )


# --- formatters -----------------------------------------------------------


def format_track_record_summary(record: TrackRecord) -> str:
    """One-line summary suitable for the LLM prompt header."""
    if record.n_mature == 0:
        if record.n_pending:
            return (
                f"Track record: 0 mature picks yet ({record.n_pending} too "
                f"young to score; min age {_MIN_AGE_DAYS}d)."
            )
        return "Track record: no prior picks in the lookback window."
    return (
        f"Track record: {record.n_mature} mature picks over the last "
        f"~{_MEASUREMENT_WINDOW_DAYS}d, mean "
        f"{record.mean_return_pct:+.1f}% vs SPY "
        f"{record.mean_spy_return_pct:+.1f}% "
        f"(alpha {record.mean_alpha_pct:+.1f}%, "
        f"{record.winners}W / {record.losers}L / {record.flats}F). "
        f"{record.n_pending} pending."
    )


def format_track_record_lines(record: TrackRecord, *, limit: int = 15) -> list[str]:
    """Per-pick lines for the report body. Mature picks first (sorted by
    alpha, biggest winners + losers shown), then a small pending tail."""
    lines: list[str] = []
    sorted_mature = sorted(
        record.picks, key=lambda p: p.alpha_pct or 0, reverse=True
    )
    for p in sorted_mature[:limit]:
        lines.append(
            f"  {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  "
            f"return {p.pick_return_pct:+.1f}%  "
            f"SPY {p.spy_return_pct:+.1f}%  "
            f"alpha {p.alpha_pct:+.1f}%"
        )
    if record.pending:
        lines.append("  -- pending (too young to score) --")
        for p in sorted(record.pending, key=lambda p: p.pick_date, reverse=True)[:5]:
            live_ret = (
                f"live {p.pick_return_pct:+.1f}%"
                if p.pick_return_pct is not None else "—"
            )
            lines.append(
                f"  {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  {live_ret}"
            )
    return lines


def format_track_record_block(record: TrackRecord) -> str:
    """Multi-line block suitable for prepending to the ranker / rebalancer
    prompt as historical context."""
    if record.n_picks_total == 0:
        return ""
    head = format_track_record_summary(record)
    body = format_track_record_lines(record, limit=10)
    return head + "\n" + "\n".join(body) if body else head


__all__ = [
    "PickReturn",
    "TrackRecord",
    "measure_track_record",
    "format_track_record_summary",
    "format_track_record_lines",
    "format_track_record_block",
]


def _ages_for_test() -> tuple[int, int]:
    return _MIN_AGE_DAYS, _MEASUREMENT_WINDOW_DAYS
