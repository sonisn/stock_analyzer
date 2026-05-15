"""Track-record measurement — close the feedback loop.

Reads two kinds of past decisions out of `discover.db`, fetches forward
prices via yfinance, scores both:

  BUY decisions  (`picks` table)    — ranker top picks from discover runs
  SELL decisions (`holdings_reviews`)— SELL or TRIM verdicts from rebalance
                                       runs (TRIM is a softer SELL but
                                       still says "reduce exposure")

Alpha sign convention: positive alpha always means "the call was right":
  - BUY:  alpha = stock_ret - spy_ret  (stock beat SPY → wise buy)
  - SELL: alpha = spy_ret - stock_ret  (stock underperformed SPY after
                                        we said sell → wise sell)

Mature decision = at least `_MIN_AGE_DAYS` old. Newer ones are listed
separately as "pending" so their noise doesn't pollute the stats.

Surfaced two ways:
  1. As a header section in the email + PDF report ("Buy track record:
     23 mature, mean +6.4% vs SPY +2.1% — Sell track record: 8 mature,
     alpha +3.1%, 5W/3L")
  2. As context in the Opus ranker prompt so the LLM can reason about
     its own historical accuracy on BOTH directions
"""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any

from ..logging import get_logger
from ..models.track_record import (
    Direction,
    DirectionStats,
    PickReturn,
    Quote,
    TrackRecord,
)

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


# --- DB read ---------------------------------------------------------------


def _fetch_recent_picks(
    conn: sqlite3.Connection, *, lookback_days: int
) -> list[tuple[str, str, int]]:
    """Return [(ticker, pick_date, age_days), ...] for every BUY pick in
    the last `lookback_days`, deduplicated to the OLDEST occurrence per
    ticker (so re-picks don't double-count)."""
    cur = conn.execute(
        "SELECT runs.run_at, picks.ticker FROM picks "
        "JOIN runs ON runs.id = picks.run_id "
        "WHERE runs.run_at >= ? "
        "ORDER BY runs.run_at ASC",
        ((datetime.now() - timedelta(days=lookback_days)).isoformat(),),
    )
    return _dedup_oldest(cur.fetchall())


def _fetch_recent_sells(
    conn: sqlite3.Connection, *, lookback_days: int
) -> list[tuple[str, str, int]]:
    """Return [(ticker, sell_date, age_days), ...] for every SELL or TRIM
    verdict in the last `lookback_days`, deduplicated to the OLDEST
    occurrence per ticker.

    SELL and TRIM both count: TRIM is a softer SELL but still a
    directional 'reduce exposure' call we should be held accountable for.
    The holdings_reviews.verdict column stores parsed verdicts from the
    Reviewer; rows where verdict is NULL or HOLD are skipped here.
    """
    cur = conn.execute(
        "SELECT runs.run_at, holdings_reviews.ticker FROM holdings_reviews "
        "JOIN runs ON runs.id = holdings_reviews.run_id "
        "WHERE runs.run_at >= ? "
        "AND UPPER(COALESCE(holdings_reviews.verdict, '')) IN ('SELL', 'TRIM') "
        "ORDER BY runs.run_at ASC",
        ((datetime.now() - timedelta(days=lookback_days)).isoformat(),),
    )
    return _dedup_oldest(cur.fetchall())


def _dedup_oldest(
    rows: list[tuple[str, str]],
) -> list[tuple[str, str, int]]:
    """Shared post-processing: keep the OLDEST (ticker, run_at) per ticker,
    compute age_days, return sorted-oldest-first."""
    oldest_by_ticker: dict[str, str] = {}
    for run_at, ticker in rows:
        if ticker not in oldest_by_ticker:
            oldest_by_ticker[ticker] = run_at
    today = date.today()
    out: list[tuple[str, str, int]] = []
    for ticker, run_at in oldest_by_ticker.items():
        try:
            decision_date = datetime.fromisoformat(run_at).date()
        except ValueError:
            continue
        age = (today - decision_date).days
        out.append((ticker, decision_date.isoformat(), age))
    return sorted(out, key=lambda x: x[1])  # oldest first


# --- yfinance price fetch --------------------------------------------------


def _fetch_quote(
    ticker: str, pick_date: str, age_days: int
) -> Quote:
    """Pick-date close and measurement-date close from yfinance.

    Measurement date = min(pick_date + 90d, today). For picks younger than
    90d we measure to today, so the "pending" entries show live returns;
    for picks older than 90d we cap at +90d so the metric stays
    apples-to-apples across vintages.
    """
    try:
        import yfinance as yf
        start = datetime.fromisoformat(pick_date).date()
        end = min(start + timedelta(days=_MEASUREMENT_WINDOW_DAYS), date.today())
        if end <= start:
            return Quote(pick_price=None, measured_price=None)
        hist = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
        if hist.empty:
            return Quote(pick_price=None, measured_price=None)
        # First and last close in the window.
        return Quote(
            pick_price=float(hist["Close"].iloc[0]),
            measured_price=float(hist["Close"].iloc[-1]),
        )
    except Exception as e:
        logger.debug("yfinance fetch failed for %s: %s", ticker, e)
        return Quote(pick_price=None, measured_price=None)


def _fetch_spy_quote(pick_date: str) -> Quote:
    return _fetch_quote("SPY", pick_date, age_days=0)


# --- aggregation ----------------------------------------------------------


def _pct_change(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * 100


def _score_pick(
    ticker: str,
    pick_date: str,
    age_days: int,
    spy_quote: Quote,
    direction: Direction = "buy",
) -> PickReturn:
    quote = _fetch_quote(ticker, pick_date, age_days)
    pick_ret = _pct_change(quote.pick_price, quote.measured_price)
    spy_ret = _pct_change(spy_quote.pick_price, spy_quote.measured_price)
    if pick_ret is not None and spy_ret is not None:
        raw_alpha = pick_ret - spy_ret
        # For sell calls, the call is "right" when the stock UNDERPERFORMS
        # SPY (you correctly identified the relative loser). Flip the sign
        # so positive alpha always means "wise call" across directions.
        alpha = raw_alpha if direction == "buy" else -raw_alpha
    else:
        alpha = None
    return PickReturn(
        ticker=ticker,
        pick_date=pick_date,
        age_days=age_days,
        direction=direction,
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
    """Top-level entry — query past buy picks AND sell calls, fetch prices,
    summarize per direction.

    `lookback_days` bounds how far back we look; defaults to 180 so the
    system has enough mature decisions to compute meaningful stats once
    it's been running a while. Empty TrackRecord is returned if the DB
    has neither buys nor sells yet.
    """
    from .persistence import connect

    try:
        with connect(db_path) as conn:
            raw_buys = _fetch_recent_picks(conn, lookback_days=lookback_days)
            raw_sells = _fetch_recent_sells(conn, lookback_days=lookback_days)
    except Exception as e:
        logger.warning("track-record fetch failed (%s) — returning empty", e)
        return _empty_record()
    if not raw_buys and not raw_sells:
        return _empty_record()

    # SPY benchmarks — one per distinct decision date so we don't refetch.
    distinct_dates = sorted({d for _, d, _ in raw_buys + raw_sells})
    spy_cache: dict[str, Quote] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for d, q in zip(distinct_dates, ex.map(_fetch_spy_quote, distinct_dates), strict=False):
            spy_cache[d] = q

    decisions: list[PickReturn] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = []
        for t, d, a in raw_buys:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "buy",
            ))
        for t, d, a in raw_sells:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "sell",
            ))
        for fut in futures:
            try:
                decisions.append(fut.result())
            except Exception as e:
                logger.debug("decision scoring failed: %s", e)

    # Drop decisions where yfinance returned no price data — these are
    # delisted tickers from old picks/sells that still live in the DB.
    # Including them as "pending" inflates the count with zombie entries
    # that will never mature; better to silently skip and log the count.
    delisted = [p for p in decisions if p.pick_price is None]
    if delisted:
        logger.info(
            "Track record: skipped %d delisted/unmeasurable ticker(s): %s",
            len(delisted),
            ", ".join(sorted({p.ticker for p in delisted})[:10]),
        )
    decisions = [p for p in decisions if p.pick_price is not None]

    mature = [p for p in decisions if p.is_mature and p.alpha_pct is not None]
    pending = [p for p in decisions if not p.is_mature or p.alpha_pct is None]

    overall = _aggregate(mature)
    buy_stats = _aggregate(
        [p for p in mature if p.direction == "buy"],
        pending_count=sum(1 for p in pending if p.direction == "buy"),
    )
    sell_stats = _aggregate(
        [p for p in mature if p.direction == "sell"],
        pending_count=sum(1 for p in pending if p.direction == "sell"),
    )

    record = TrackRecord(
        n_picks_total=len(decisions),
        n_mature=len(mature),
        n_pending=len(pending),
        mean_return_pct=overall.mean_return_pct,
        mean_spy_return_pct=overall.mean_spy_return_pct,
        mean_alpha_pct=overall.mean_alpha_pct,
        winners=overall.winners,
        losers=overall.losers,
        flats=overall.flats,
        buy_stats=buy_stats,
        sell_stats=sell_stats,
        picks=mature,
        pending=pending,
    )
    logger.info(
        "Track record: buys %d mature alpha %s%% (%dW/%dL), "
        "sells %d mature alpha %s%% (%dW/%dL), %d pending",
        buy_stats.n_mature,
        f"{buy_stats.mean_alpha_pct:+.1f}" if buy_stats.mean_alpha_pct is not None else "—",
        buy_stats.winners, buy_stats.losers,
        sell_stats.n_mature,
        f"{sell_stats.mean_alpha_pct:+.1f}" if sell_stats.mean_alpha_pct is not None else "—",
        sell_stats.winners, sell_stats.losers,
        record.n_pending,
    )
    return record


def _aggregate(
    mature: list[PickReturn], *, pending_count: int = 0
) -> DirectionStats:
    """Compute mean returns + win/loss/flat counts for a list of mature
    decisions. Alpha is already direction-aware (positive = right call),
    so we threshold ±0.5% on alpha regardless of direction."""
    if not mature:
        return DirectionStats(
            n_mature=0, n_pending=pending_count,
            mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
            winners=0, losers=0, flats=0,
        )
    mean_ret = sum(p.pick_return_pct or 0 for p in mature) / len(mature)
    mean_spy = sum(p.spy_return_pct or 0 for p in mature) / len(mature)
    mean_alpha = sum(p.alpha_pct or 0 for p in mature) / len(mature)
    winners = sum(1 for p in mature if (p.alpha_pct or 0) > 0.5)
    losers = sum(1 for p in mature if (p.alpha_pct or 0) < -0.5)
    flats = len(mature) - winners - losers
    return DirectionStats(
        n_mature=len(mature),
        n_pending=pending_count,
        mean_return_pct=mean_ret,
        mean_spy_return_pct=mean_spy,
        mean_alpha_pct=mean_alpha,
        winners=winners,
        losers=losers,
        flats=flats,
    )


def _empty_record() -> TrackRecord:
    empty_dir = DirectionStats(
        n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0,
    )
    return TrackRecord(
        n_picks_total=0, n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0,
        buy_stats=empty_dir, sell_stats=empty_dir,
        picks=[], pending=[],
    )


# --- formatters -----------------------------------------------------------


def format_track_record_summary(record: TrackRecord) -> str:
    """One-line summary suitable for the LLM prompt header. Renders BOTH
    directions when each has at least one mature decision; falls back to
    a single combined line when only one direction exists."""
    if record.n_mature == 0:
        if record.n_pending:
            return (
                f"Track record: 0 mature decisions yet ({record.n_pending} "
                f"too young to score; min age {_MIN_AGE_DAYS}d)."
            )
        return "Track record: no prior decisions in the lookback window."

    parts: list[str] = []
    parts.append(f"Track record (last ~{_MEASUREMENT_WINDOW_DAYS}d):")
    if record.buy_stats.n_mature:
        bs = record.buy_stats
        parts.append(
            f" Buy picks {bs.n_mature} mature, "
            f"return {bs.mean_return_pct:+.1f}% vs SPY {bs.mean_spy_return_pct:+.1f}% "
            f"(alpha {bs.mean_alpha_pct:+.1f}%, "
            f"{bs.winners}W/{bs.losers}L/{bs.flats}F)."
        )
    if record.sell_stats.n_mature:
        ss = record.sell_stats
        # For sells, "stock return" is the unflipped raw return — what the
        # ticker did. Alpha is sign-flipped so positive means the sell call
        # was right. Report both.
        parts.append(
            f" Sell calls {ss.n_mature} mature, "
            f"stock {ss.mean_return_pct:+.1f}% vs SPY {ss.mean_spy_return_pct:+.1f}% "
            f"(call-alpha {ss.mean_alpha_pct:+.1f}%, "
            f"{ss.winners}W/{ss.losers}L/{ss.flats}F)."
        )
    if record.n_pending:
        parts.append(f" {record.n_pending} pending.")
    return "".join(parts)


def _format_decision_line(p: PickReturn) -> str:
    """Render one mature decision; sells get a [SELL] tag so the user can
    tell directions apart in the listing."""
    tag = "[SELL]" if p.direction == "sell" else "[BUY] "
    return (
        f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  "
        f"return {p.pick_return_pct:+.1f}%  "
        f"SPY {p.spy_return_pct:+.1f}%  "
        f"alpha {p.alpha_pct:+.1f}%"
    )


def format_track_record_lines(record: TrackRecord, *, limit: int = 15) -> list[str]:
    """Per-decision lines for the report body. Buy and sell mature
    sections rendered separately so the user can see how each direction
    performed; biggest winners + losers shown per section."""
    lines: list[str] = []
    buy_mature = sorted(
        [p for p in record.picks if p.direction == "buy"],
        key=lambda p: p.alpha_pct or 0, reverse=True,
    )
    sell_mature = sorted(
        [p for p in record.picks if p.direction == "sell"],
        key=lambda p: p.alpha_pct or 0, reverse=True,
    )

    if buy_mature:
        lines.append("  -- BUY picks (mature) --")
        for p in buy_mature[:limit]:
            lines.append(_format_decision_line(p))
    if sell_mature:
        lines.append("  -- SELL calls (mature) --")
        for p in sell_mature[:limit]:
            lines.append(_format_decision_line(p))

    if record.pending:
        lines.append("  -- pending (too young to score) --")
        for p in sorted(record.pending, key=lambda p: p.pick_date, reverse=True)[:5]:
            tag = "[SELL]" if p.direction == "sell" else "[BUY] "
            live_ret = (
                f"live {p.pick_return_pct:+.1f}%"
                if p.pick_return_pct is not None else "—"
            )
            lines.append(
                f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  {live_ret}"
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
    "Direction",
    "Quote",
    "PickReturn",
    "DirectionStats",
    "TrackRecord",
    "measure_track_record",
    "score_covered_call",
    "format_track_record_block",
    "format_track_record_summary",
    "format_track_record_lines",
]


def _ages_for_test() -> tuple[int, int]:
    return _MIN_AGE_DAYS, _MEASUREMENT_WINDOW_DAYS


# --- Covered call scoring --------------------------------------------------


def _spot_at(ticker: str, on: str) -> float | None:
    """Look up historical spot for `ticker` on ISO date `on`.

    Default impl uses yfinance; patched in tests. Returns None when the
    lookup fails so the caller can mark the outcome UNKNOWN rather than
    crash the track-record block.
    """
    try:
        import yfinance as yf
        end = date.fromisoformat(on)
        start = end - timedelta(days=7)
        df = yf.Ticker(ticker).history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:
        from ..logging import get_logger as _gl
        _gl(__name__).info(
            "score_covered_call: spot lookup failed for %s on %s (%s); "
            "outcome will be UNKNOWN.", ticker, on, e,
        )
        return None


def score_covered_call(
    *,
    ticker: str,
    strike: float,
    expiry: str,
    contracts: int,
    est_premium_per_share: float,
) -> dict[str, Any]:
    """Score one WRITE_CALL after `expiry` has passed.

    Returns:
      {
        "outcome": "EXPIRED_OTM" | "ASSIGNED" | "UNKNOWN",
        "spot_at_expiry": float | None,
        "pnl_usd": float | None,            # net of opportunity cost
        "premium_collected_usd": float,
        "opportunity_cost_usd": float,
      }
    """
    spot = _spot_at(ticker, expiry)
    premium = contracts * est_premium_per_share * 100.0
    if spot is None:
        return {
            "outcome": "UNKNOWN",
            "spot_at_expiry": None,
            "pnl_usd": None,
            "premium_collected_usd": premium,
            "opportunity_cost_usd": 0.0,
        }
    if spot < strike:
        return {
            "outcome": "EXPIRED_OTM",
            "spot_at_expiry": spot,
            "pnl_usd": premium,
            "premium_collected_usd": premium,
            "opportunity_cost_usd": 0.0,
        }
    opportunity_cost = (spot - strike) * contracts * 100.0
    return {
        "outcome": "ASSIGNED",
        "spot_at_expiry": spot,
        "pnl_usd": premium - opportunity_cost,
        "premium_collected_usd": premium,
        "opportunity_cost_usd": opportunity_cost,
    }
