"""Track-record measurement — close the feedback loop.

Reads four kinds of past decisions out of `discover.db`, fetches forward
prices via yfinance, and scores each:

  BUY  decisions (`picks` table)            — discover-run top picks
  HOLD decisions (`holdings_reviews` HOLD)  — rebalance "keep it" verdicts
  TRIM decisions (`holdings_reviews` TRIM)  — rebalance "reduce by X%" verdicts
  SELL decisions (`holdings_reviews` SELL)  — rebalance "exit" verdicts

Alpha sign convention: positive alpha always means "the call was right".
  - BUY  / HOLD: alpha = stock_ret - spy_ret  (vindicated when stock beats SPY)
  - TRIM / SELL: alpha = spy_ret - stock_ret  (vindicated when stock lags SPY)

Mature decision = at least `_MIN_AGE_DAYS` old. Newer ones are listed
separately as "pending" so their noise doesn't pollute the stats.

Surfaced two ways:
  1. As a header section in the email + PDF report (one line per direction
     with mean alpha + per-decision Sharpe, plus an opus-model breakdown row)
  2. As context in the Opus ranker prompt so the LLM can reason about
     its own historical accuracy by direction AND by which model picked
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any

from ..db.session import get_session
from ..db.track_record import (
    fetch_recent_pick_runs_with_model,
    fetch_recent_verdict_runs,
)
from ..logging import get_logger
from ..models.track_record import (
    Direction,
    DirectionStats,
    ModelStats,
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
        # BUY and HOLD: vindicated when the stock outperforms SPY (positive raw_alpha).
        # TRIM and SELL: vindicated when the stock underperforms SPY — sign-flip so
        # positive always means "the call was right" across all four directions.
        alpha = raw_alpha if direction in ("buy", "hold") else -raw_alpha
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


def _sharpe(alphas: list[float]) -> float | None:
    """Per-decision Sharpe = mean(alpha) / stdev(alpha). Returns None when
    the sample is too small (< 5) or effectively flat (stdev < 0.001).
    Unannualized — matches the unit of the underlying alpha (per-decision
    measurement over the 90-day window)."""
    if len(alphas) < 5:
        return None
    stdev = statistics.stdev(alphas)
    if stdev < 0.001:
        return None
    return statistics.mean(alphas) / stdev


def measure_track_record(
    db_path: str, *, lookback_days: int = 180
) -> TrackRecord:
    """Top-level entry — query buy picks AND hold/trim/sell verdicts, fetch
    forward prices, summarize per direction and per Opus model.

    `lookback_days` bounds how far back we look; defaults to 180 so the
    system has enough mature decisions to compute meaningful stats once
    it's been running a while. Empty TrackRecord is returned if the DB
    has no decisions yet.
    """
    try:
        with get_session(db_path) as session:
            pick_rows = fetch_recent_pick_runs_with_model(session, lookback_days=lookback_days)
            hold_rows = fetch_recent_verdict_runs(session, "HOLD", lookback_days=lookback_days)
            trim_rows = fetch_recent_verdict_runs(session, "TRIM", lookback_days=lookback_days)
            sell_rows = fetch_recent_verdict_runs(session, "SELL", lookback_days=lookback_days)
        buy_pairs = [(r, t) for r, t, _m in pick_rows]
        raw_buys = _dedup_oldest(buy_pairs)
        raw_holds = _dedup_oldest(hold_rows)
        raw_trims = _dedup_oldest(trim_rows)
        raw_sells = _dedup_oldest(sell_rows)
        ticker_model: dict[str, str | None] = {}
        for _r, ticker, model in pick_rows:
            ticker_model.setdefault(ticker, model)
    except Exception as e:
        logger.warning("track-record fetch failed (%s) — returning empty", e)
        return _empty_record()
    if not (raw_buys or raw_holds or raw_trims or raw_sells):
        return _empty_record()

    distinct_dates = sorted({
        d for _, d, _ in raw_buys + raw_holds + raw_trims + raw_sells
    })
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
        for t, d, a in raw_holds:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "hold",
            ))
        for t, d, a in raw_trims:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "trim",
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
    hold_stats = _aggregate(
        [p for p in mature if p.direction == "hold"],
        pending_count=sum(1 for p in pending if p.direction == "hold"),
    )
    trim_stats = _aggregate(
        [p for p in mature if p.direction == "trim"],
        pending_count=sum(1 for p in pending if p.direction == "trim"),
    )
    sell_stats = _aggregate(
        [p for p in mature if p.direction == "sell"],
        pending_count=sum(1 for p in pending if p.direction == "sell"),
    )

    model_breakdown = _compute_model_breakdown(
        [p for p in mature if p.direction == "buy"], ticker_model,
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
        overall_sharpe=overall.sharpe,
        buy_stats=buy_stats,
        hold_stats=hold_stats,
        trim_stats=trim_stats,
        sell_stats=sell_stats,
        model_breakdown=model_breakdown,
        picks=mature,
        pending=pending,
    )
    logger.info(
        "Track record: buys=%d hold=%d trim=%d sell=%d mature; %d pending; "
        "overall_sharpe=%s",
        buy_stats.n_mature, hold_stats.n_mature,
        trim_stats.n_mature, sell_stats.n_mature,
        record.n_pending,
        f"{record.overall_sharpe:.2f}" if record.overall_sharpe is not None else "n/a",
    )
    return record


def _compute_model_breakdown(
    buy_mature: list[PickReturn],
    ticker_model: Mapping[str, str | None],
) -> list[ModelStats]:
    """Group mature BUY decisions by their originating opus_model. Models
    with n_mature < 3 are dropped — their stats are too noisy to report
    individually (the decisions still appear in the overall buy aggregate).
    Picks whose opus_model is None are grouped under 'unknown'.

    `n_mature` in the returned `ModelStats` counts only picks with a
    measurable alpha (not the raw bucket size), so the reported mean is
    always derived from exactly `n_mature` data points. Today's caller
    pre-filters to alpha-bearing picks, but this guarantee makes the
    function safe under wider future use."""
    by_model: dict[str, list[PickReturn]] = defaultdict(list)
    for p in buy_mature:
        model = ticker_model.get(p.ticker) or "unknown"
        by_model[model].append(p)
    out: list[ModelStats] = []
    for model, picks in by_model.items():
        if len(picks) < 3:
            continue
        alphas = [p.alpha_pct for p in picks if p.alpha_pct is not None]
        if len(alphas) < 3:
            continue
        out.append(ModelStats(
            opus_model=model,
            n_mature=len(alphas),
            mean_alpha_pct=sum(alphas) / len(alphas),
            sharpe=_sharpe(alphas),
        ))
    # mean_alpha_pct is always non-None for surviving rows (we just computed it
    # from a non-empty alphas list); the `or 0.0` is a typing-narrowing fallback.
    return sorted(
        out, key=lambda m: m.mean_alpha_pct if m.mean_alpha_pct is not None else 0.0,
        reverse=True,
    )


def _aggregate(
    mature: list[PickReturn], *, pending_count: int = 0
) -> DirectionStats:
    """Compute mean returns + win/loss/flat counts + Sharpe for a list of
    mature decisions. Alpha is already direction-aware (positive = right
    call), so we threshold ±0.5% on alpha regardless of direction."""
    if not mature:
        return DirectionStats(
            n_mature=0, n_pending=pending_count,
            mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
            winners=0, losers=0, flats=0, sharpe=None,
        )
    mean_ret = sum(p.pick_return_pct or 0 for p in mature) / len(mature)
    mean_spy = sum(p.spy_return_pct or 0 for p in mature) / len(mature)
    mean_alpha = sum(p.alpha_pct or 0 for p in mature) / len(mature)
    winners = sum(1 for p in mature if (p.alpha_pct or 0) > 0.5)
    losers = sum(1 for p in mature if (p.alpha_pct or 0) < -0.5)
    flats = len(mature) - winners - losers
    alphas = [p.alpha_pct for p in mature if p.alpha_pct is not None]
    return DirectionStats(
        n_mature=len(mature),
        n_pending=pending_count,
        mean_return_pct=mean_ret,
        mean_spy_return_pct=mean_spy,
        mean_alpha_pct=mean_alpha,
        winners=winners,
        losers=losers,
        flats=flats,
        sharpe=_sharpe(alphas),
    )


def _empty_record() -> TrackRecord:
    empty_dir = DirectionStats(
        n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, sharpe=None,
    )
    return TrackRecord(
        n_picks_total=0, n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, overall_sharpe=None,
        buy_stats=empty_dir,
        hold_stats=empty_dir,
        trim_stats=empty_dir,
        sell_stats=empty_dir,
        model_breakdown=[],
        picks=[], pending=[],
    )


# --- formatters -----------------------------------------------------------


def _sharpe_text(sharpe: float | None, n_mature: int) -> str:
    """Render Sharpe as either '0.42' or 'n/a (n<5)' / 'n/a (flat)'."""
    if sharpe is not None:
        return f"{sharpe:.2f}"
    if n_mature < 5:
        return "n/a (n<5)"
    return "n/a (flat)"


_DIRECTION_TAGS: dict[str, str] = {
    "buy": "[BUY] ",
    "hold": "[HOLD]",
    "trim": "[TRIM]",
    "sell": "[SELL]",
}
_UNKNOWN_DIRECTION_TAG = "[?]   "


def _alpha_text(mean_alpha_pct: float | None) -> str:
    """Render mean alpha as ``+8.0%`` or ``n/a`` — shared between the summary
    and block formatters so DirectionStats with None alpha don't crash the
    f-string."""
    if mean_alpha_pct is None:
        return "n/a"
    return f"{mean_alpha_pct:+.1f}%"


def format_track_record_summary(record: TrackRecord) -> str:
    """One-line summary suitable for the dashboard / short prompt context.
    Renders the OVERALL aggregate plus per-direction sub-totals when each
    direction has at least one mature decision."""
    if record.n_mature == 0:
        if record.n_pending:
            return (
                f"Track record: 0 mature decisions yet ({record.n_pending} "
                f"too young to score; min age {_MIN_AGE_DAYS}d)."
            )
        return "Track record: no prior decisions in the lookback window."

    parts: list[str] = [f"Track record (last ~{_MEASUREMENT_WINDOW_DAYS}d):"]
    for label, stats in (
        ("Buy", record.buy_stats),
        ("Hold", record.hold_stats),
        ("Trim", record.trim_stats),
        ("Sell", record.sell_stats),
    ):
        if stats.n_mature:
            parts.append(
                f" {label} {stats.n_mature} mature, "
                f"alpha {_alpha_text(stats.mean_alpha_pct)} "
                f"({stats.winners}W/{stats.losers}L/{stats.flats}F)."
            )
    if record.n_pending:
        parts.append(f" {record.n_pending} pending.")
    return "".join(parts)


def _format_decision_line(p: PickReturn) -> str:
    """Render one mature decision; non-buy decisions get a [VERDICT] tag
    so the user can tell directions apart in the listing."""
    tag = _DIRECTION_TAGS.get(p.direction, _UNKNOWN_DIRECTION_TAG)
    return (
        f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  "
        f"return {p.pick_return_pct:+.1f}%  "
        f"SPY {p.spy_return_pct:+.1f}%  "
        f"alpha {p.alpha_pct:+.1f}%"
    )


def format_track_record_lines(record: TrackRecord, *, limit: int = 15) -> list[str]:
    """Per-decision lines for the report body. One section per direction
    with mature data; each section shows the top decisions by alpha."""
    lines: list[str] = []
    by_dir: dict[str, list[PickReturn]] = {
        "buy": [], "hold": [], "trim": [], "sell": [],
    }
    for p in record.picks:
        # pre-populated above; KeyError on an unknown direction is the right
        # failure mode (signals the Direction literal grew without updating
        # this dispatch table).
        by_dir[p.direction].append(p)

    section_headers = [
        ("buy", "  -- BUY picks (mature) --"),
        ("hold", "  -- HOLD verdicts (mature) --"),
        ("trim", "  -- TRIM verdicts (mature) --"),
        ("sell", "  -- SELL verdicts (mature) --"),
    ]
    for direction, header in section_headers:
        picks = sorted(
            by_dir[direction], key=lambda p: p.alpha_pct or 0, reverse=True,
        )
        if not picks:
            continue
        lines.append(header)
        for p in picks[:limit]:
            lines.append(_format_decision_line(p))

    if record.pending:
        lines.append("  -- pending (too young to score) --")
        for p in sorted(record.pending, key=lambda p: p.pick_date, reverse=True)[:5]:
            tag = _DIRECTION_TAGS.get(p.direction, _UNKNOWN_DIRECTION_TAG)
            live_ret = (
                f"live {p.pick_return_pct:+.1f}%"
                if p.pick_return_pct is not None else "—"
            )
            lines.append(
                f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  {live_ret}"
            )
    return lines


def _format_direction_block_line(
    label: str, stats: DirectionStats, *, is_first: bool,
) -> str:
    """One line per direction in the multi-line block — sample size, alpha,
    Sharpe. The FIRST direction line (``is_first=True``) spells out
    "Sharpe (per-decision)" to communicate the unit; subsequent lines just
    say "Sharpe"."""
    sharpe_label = "Sharpe (per-decision)" if is_first else "Sharpe"
    return (
        f"{label} track record: {stats.n_mature} mature, "
        f"alpha {_alpha_text(stats.mean_alpha_pct)}, "
        f"{sharpe_label} {_sharpe_text(stats.sharpe, stats.n_mature)}"
    )


def format_track_record_block(record: TrackRecord) -> str:
    """Multi-line block suitable for prepending to the ranker / rebalancer
    prompt as historical context. Direction lines emitted for every
    direction with at least one mature decision; model_breakdown rendered
    on the last header line when non-empty; per-decision detail follows."""
    if record.n_picks_total == 0:
        return ""
    lines: list[str] = []
    is_first = True
    for label, stats in [
        ("Buy", record.buy_stats),
        ("Hold", record.hold_stats),
        ("Trim", record.trim_stats),
        ("Sell", record.sell_stats),
    ]:
        if stats.n_mature:
            lines.append(_format_direction_block_line(
                label, stats, is_first=is_first,
            ))
            is_first = False
    if record.model_breakdown:
        model_parts = [
            f"{m.opus_model} ({m.n_mature} picks, {m.mean_alpha_pct:+.1f}%)"
            for m in record.model_breakdown
        ]
        lines.append("Model breakdown: " + " | ".join(model_parts))
    head = "\n".join(lines) if lines else format_track_record_summary(record)
    body = format_track_record_lines(record, limit=10)
    return head + "\n" + "\n".join(body) if body else head


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
