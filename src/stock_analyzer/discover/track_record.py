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

Three properties this module is careful about, because the number it
produces is fed back into the ranker prompt as the model's own accuracy:

1. ONE HORIZON PER NUMBER. Every decision is measured over a *completed*
   fixed window (`_HORIZONS`) and aggregated only with other decisions
   measured over the same window. Mixing a 15-day outcome into the same
   mean as a 90-day one made the statistic track how recently the
   pipeline had run rather than how good the calls were.

2. NOTHING IS SILENTLY DROPPED. A decision whose forward price lookup
   comes back empty while SPY has data is usually a delisting — the worst
   outcome a BUY can have — or a bad symbol out of the news-regex
   universe. Removing those from the sample biases the mean upward by
   exactly the left tail, so they are counted and reported instead.

3. BETA IS NOT SKILL. The screen selects high-beta momentum leaders by
   construction, so raw excess return over SPY flatters the system in a
   rising tape. Every row also carries `beta_adjusted_alpha_pct`, with
   beta estimated strictly on pre-decision data.

Surfaced two ways:
  1. As a header section in the email + PDF report (one block per
     horizon, one line per direction, with the beta-adjusted figure and
     per-decision Sharpe alongside the raw alpha)
  2. As context in the Opus ranker prompt so the LLM can reason about
     its own historical accuracy by direction AND by which model picked
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ..db.session import get_session
from ..db.track_record import (
    fetch_recent_pick_runs_with_model,
    fetch_recent_verdict_runs,
)
from ..logging import get_logger
from ..models.track_record import (
    Direction,
    DirectionStats,
    HorizonStats,
    ModelStats,
    PickReturn,
    ProviderStats,
    TrackRecord,
    UnmeasurableDecision,
)

logger = get_logger(__name__)

# Completed measurement windows, shortest first. A decision is scored at
# every horizon it is old enough to have finished, and the resulting rows
# are only ever aggregated with other rows at the SAME horizon. 90 days is
# the evaluation window the reports headline; 30 gives an earlier read
# while a fresh database fills up.
_HORIZONS: tuple[int, ...] = (30, 90)
# The horizon the top-level TrackRecord fields describe when it has data.
_PRIMARY_HORIZON = 90
# Anything younger than the shortest horizon cannot be scored yet; it is
# listed as "pending" with a live mark so the user sees it exists.
_MIN_AGE_DAYS = min(_HORIZONS)

# Trailing window for the beta estimate, measured BACKWARD from the
# decision date so the estimate never sees the outcome it adjusts.
_BETA_LOOKBACK_DAYS = 180
# Below this many overlapping daily observations the covariance estimate is
# too noisy to use; the row keeps a None beta and sits out the
# beta-adjusted mean rather than polluting it.
_BETA_MIN_OBS = 60

# yfinance batch size for parallel ticker fetches.
_MAX_WORKERS = 6


@dataclass(frozen=True)
class _Decision:
    """One deduplicated decision awaiting measurement."""

    ticker: str
    pick_date: str
    age_days: int
    direction: Direction


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


# --- price history ---------------------------------------------------------


def _fetch_history(ticker: str, start: date, end: date) -> pd.DataFrame | None:
    """Daily OHLCV for [start, end], date-indexed and timezone-naive.

    One fetch per (ticker, decision) serves every horizon plus the beta
    estimate, so widening the window costs no extra requests. Returns None
    when the provider has nothing — the caller decides whether that means
    "too young" or "delisted / bad symbol".
    """
    try:
        from ..data import yf_gateway

        df = yf_gateway.history(
            ticker,
            what="track_record.history",
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
        if df is None or df.empty:
            return None
        df = df.copy()
        idx = pd.to_datetime(df.index)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        df.index = idx.normalize()
        return df
    except Exception as e:
        logger.debug("yfinance history fetch failed for %s: %s", ticker, e)
        return None


def _close_on_or_after(closes: pd.Series, on: date) -> tuple[float, date] | None:
    """First close at or after `on` — the entry fill a reader could get."""
    window = closes[closes.index >= pd.Timestamp(on)]
    if window.empty:
        return None
    value = float(window.iloc[0])
    return (value, window.index[0].date()) if pd.notna(value) else None


def _close_on_or_before(closes: pd.Series, on: date) -> tuple[float, date] | None:
    """Last close at or before `on` — the horizon's measurement price."""
    window = closes[closes.index <= pd.Timestamp(on)]
    if window.empty:
        return None
    value = float(window.iloc[-1])
    return (value, window.index[-1].date()) if pd.notna(value) else None


def _compute_beta(ticker_closes: pd.Series, spy_closes: pd.Series, pick_date: date) -> float | None:
    """Trailing beta vs SPY on daily returns strictly BEFORE `pick_date`.

    Using only pre-decision data is the point: a beta fitted over the
    measurement window itself would absorb the very move being scored.
    """
    start = pd.Timestamp(pick_date - timedelta(days=_BETA_LOOKBACK_DAYS))
    cutoff = pd.Timestamp(pick_date)
    joined = pd.DataFrame(
        {
            "t": ticker_closes[(ticker_closes.index >= start) & (ticker_closes.index < cutoff)],
            "s": spy_closes[(spy_closes.index >= start) & (spy_closes.index < cutoff)],
        }
    ).dropna()
    if len(joined) < _BETA_MIN_OBS + 1:
        return None
    returns = joined.pct_change().dropna()
    if len(returns) < _BETA_MIN_OBS:
        return None
    spy_var = float(returns["s"].var())
    if not spy_var or spy_var <= 0:
        return None
    beta = float(returns["t"].cov(returns["s"]) / spy_var)
    if pd.isna(beta):
        return None
    return beta


# --- aggregation ----------------------------------------------------------


def _pct_change(start: float | None, end: float | None) -> float | None:
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1) * 100


def _directional(value: float | None, direction: Direction) -> float | None:
    """Flip sign for TRIM/SELL so positive always means 'call was right'."""
    if value is None:
        return None
    return value if direction in ("buy", "hold") else -value


def _score_decision(
    decision: _Decision,
    ticker_df: pd.DataFrame | None,
    spy_df: pd.DataFrame | None,
) -> tuple[list[PickReturn], UnmeasurableDecision | None]:
    """Score one decision at every horizon it has finished.

    Returns (rows, unmeasurable). A decision younger than the shortest
    horizon yields exactly one pending row (live mark, `horizon_days=0`).
    """
    pick_date = date.fromisoformat(decision.pick_date)
    too_young = decision.age_days < _MIN_AGE_DAYS

    if ticker_df is None or ticker_df.empty or spy_df is None or spy_df.empty:
        return [], UnmeasurableDecision(
            ticker=decision.ticker,
            pick_date=decision.pick_date,
            direction=decision.direction,
            age_days=decision.age_days,
            reason="too_young" if too_young else "no_price_data",
        )

    t_closes = ticker_df["Close"].dropna()
    s_closes = spy_df["Close"].dropna()
    t_entry = _close_on_or_after(t_closes, pick_date)
    s_entry = _close_on_or_after(s_closes, pick_date)
    if t_entry is None or s_entry is None:
        return [], UnmeasurableDecision(
            ticker=decision.ticker,
            pick_date=decision.pick_date,
            direction=decision.direction,
            age_days=decision.age_days,
            reason="too_young" if too_young else "no_price_data",
        )

    beta = _compute_beta(t_closes, s_closes, pick_date)

    def _row(horizon: int, measure_on: date, *, mature: bool) -> PickReturn | None:
        t_exit = _close_on_or_before(t_closes, measure_on)
        s_exit = _close_on_or_before(s_closes, measure_on)
        if t_exit is None or s_exit is None:
            return None
        if t_exit[1] <= t_entry[1]:
            return None  # no elapsed trading time yet
        pick_ret = _pct_change(t_entry[0], t_exit[0])
        spy_ret = _pct_change(s_entry[0], s_exit[0])
        alpha = (
            _directional(pick_ret - spy_ret, decision.direction)
            if pick_ret is not None and spy_ret is not None
            else None
        )
        beta_alpha = (
            _directional(pick_ret - beta * spy_ret, decision.direction)
            if (pick_ret is not None and spy_ret is not None and beta is not None)
            else None
        )
        return PickReturn(
            ticker=decision.ticker,
            pick_date=decision.pick_date,
            age_days=decision.age_days,
            direction=decision.direction,
            horizon_days=horizon,
            measured_date=t_exit[1].isoformat(),
            pick_price=t_entry[0],
            measured_price=t_exit[0],
            pick_return_pct=pick_ret,
            spy_return_pct=spy_ret,
            alpha_pct=alpha,
            beta=beta,
            beta_adjusted_alpha_pct=beta_alpha,
            is_mature=mature,
        )

    if too_young:
        # Live mark to the latest available bar, clearly not a finished
        # measurement (horizon_days=0, is_mature=False).
        pending = _row(0, date.today(), mature=False)
        if pending is None:
            return [], UnmeasurableDecision(
                ticker=decision.ticker,
                pick_date=decision.pick_date,
                direction=decision.direction,
                age_days=decision.age_days,
                reason="too_young",
            )
        return [pending], None

    rows: list[PickReturn] = []
    for horizon in _HORIZONS:
        if decision.age_days < horizon:
            continue
        row = _row(horizon, pick_date + timedelta(days=horizon), mature=True)
        if row is not None:
            rows.append(row)
    if not rows:
        return [], UnmeasurableDecision(
            ticker=decision.ticker,
            pick_date=decision.pick_date,
            direction=decision.direction,
            age_days=decision.age_days,
            reason="no_price_data",
        )
    return rows, None


def _sharpe(alphas: list[float]) -> float | None:
    """Per-decision Sharpe = mean(alpha) / stdev(alpha). Returns None when
    the sample is too small (< 5) or effectively flat (stdev < 0.001).
    Unannualized, and only ever computed within a single horizon — the
    unit is "one decision measured over that horizon"."""
    if len(alphas) < 5:
        return None
    stdev = statistics.stdev(alphas)
    if stdev < 0.001:
        return None
    return statistics.mean(alphas) / stdev


def _collect_decisions(
    db_path: str, lookback_days: int
) -> tuple[list[_Decision], dict[str, str | None], dict[str, list[str] | None]] | None:
    """Read + dedup every decision. None signals a failed DB read."""
    try:
        with get_session(db_path) as session:
            pick_rows = fetch_recent_pick_runs_with_model(session, lookback_days=lookback_days)
            verdict_rows = {
                verdict: fetch_recent_verdict_runs(session, verdict, lookback_days=lookback_days)
                for verdict in ("HOLD", "TRIM", "SELL")
            }
    except Exception as e:
        logger.warning("track-record fetch failed (%s) — returning empty", e)
        return None

    ticker_model: dict[str, str | None] = {}
    ticker_providers: dict[str, list[str] | None] = {}
    for _run_at, ticker, model, voting_providers in pick_rows:
        ticker_model.setdefault(ticker, model)
        ticker_providers.setdefault(
            ticker, voting_providers.split(",") if voting_providers else None
        )

    decisions: list[_Decision] = []
    buy_pairs = [(run_at, ticker) for run_at, ticker, _m, _vp in pick_rows]
    for source, direction in (
        (buy_pairs, "buy"),
        (verdict_rows["HOLD"], "hold"),
        (verdict_rows["TRIM"], "trim"),
        (verdict_rows["SELL"], "sell"),
    ):
        for ticker, pick_date, age in _dedup_oldest(source):
            decisions.append(
                _Decision(
                    ticker=ticker,
                    pick_date=pick_date,
                    age_days=age,
                    direction=direction,  # type: ignore[arg-type]
                )
            )
    return decisions, ticker_model, ticker_providers


def measure_track_record(db_path: str, *, lookback_days: int = 180) -> TrackRecord:
    """Top-level entry — query buy picks AND hold/trim/sell verdicts, fetch
    forward prices, and summarize per horizon, per direction and per Opus
    model.

    `lookback_days` bounds how far back we look; defaults to 180 so the
    system has enough finished decisions to compute meaningful stats once
    it's been running a while. Empty TrackRecord is returned if the DB
    has no decisions yet.
    """
    collected = _collect_decisions(db_path, lookback_days)
    if collected is None:
        return _empty_record()
    decisions, ticker_model, ticker_providers = collected
    if not decisions:
        return _empty_record()

    # One history fetch per (ticker, decision date) covers every horizon and
    # the beta window; SPY is fetched once per distinct decision date.
    max_horizon = max(_HORIZONS)
    fetch_keys = {(d.ticker, d.pick_date) for d in decisions}
    distinct_dates = sorted({d.pick_date for d in decisions})

    def _window(pick_date: str) -> tuple[date, date]:
        start = date.fromisoformat(pick_date)
        return (
            start - timedelta(days=_BETA_LOOKBACK_DAYS + 10),
            min(start + timedelta(days=max_horizon), date.today()),
        )

    spy_frames: dict[str, pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {d: ex.submit(_fetch_history, "SPY", *_window(d)) for d in distinct_dates}
        for d, fut in futures.items():
            try:
                spy_frames[d] = fut.result()
            except Exception as e:
                logger.debug("SPY history fetch failed for %s: %s", d, e)
                spy_frames[d] = None

    ticker_frames: dict[tuple[str, str], pd.DataFrame | None] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures2 = {key: ex.submit(_fetch_history, key[0], *_window(key[1])) for key in fetch_keys}
        for key, fut in futures2.items():
            try:
                ticker_frames[key] = fut.result()
            except Exception as e:
                logger.debug("history fetch failed for %s: %s", key, e)
                ticker_frames[key] = None

    all_rows: list[PickReturn] = []
    unmeasurable: list[UnmeasurableDecision] = []
    for decision in decisions:
        rows, bad = _score_decision(
            decision,
            ticker_frames.get((decision.ticker, decision.pick_date)),
            spy_frames.get(decision.pick_date),
        )
        all_rows.extend(rows)
        if bad is not None:
            unmeasurable.append(bad)

    no_data = [u for u in unmeasurable if u.reason == "no_price_data"]
    if no_data:
        logger.info(
            "Track record: %d decision(s) had no forward price data "
            "(delisted or bad symbol) and are reported, not dropped: %s",
            len(no_data),
            ", ".join(sorted({u.ticker for u in no_data})[:10]),
        )

    pending = [r for r in all_rows if not r.is_mature]
    horizons: list[HorizonStats] = []
    for horizon in _HORIZONS:
        rows = [r for r in all_rows if r.is_mature and r.horizon_days == horizon]
        if not rows:
            continue
        scored = [r for r in rows if r.alpha_pct is not None]
        horizons.append(
            HorizonStats(
                horizon_days=horizon,
                overall=_aggregate(_dedup_for_overall(scored)),
                buy_stats=_aggregate(
                    [r for r in scored if r.direction == "buy"],
                    pending_count=sum(1 for p in pending if p.direction == "buy"),
                ),
                hold_stats=_aggregate(
                    [r for r in scored if r.direction == "hold"],
                    pending_count=sum(1 for p in pending if p.direction == "hold"),
                ),
                trim_stats=_aggregate(
                    [r for r in scored if r.direction == "trim"],
                    pending_count=sum(1 for p in pending if p.direction == "trim"),
                ),
                sell_stats=_aggregate(
                    [r for r in scored if r.direction == "sell"],
                    pending_count=sum(1 for p in pending if p.direction == "sell"),
                ),
                model_breakdown=_compute_model_breakdown(
                    [r for r in scored if r.direction == "buy"],
                    ticker_model,
                ),
                provider_breakdown=_compute_provider_breakdown(
                    [r for r in scored if r.direction == "buy"],
                    ticker_providers,
                ),
                decisions=rows,
            )
        )

    reported = _pick_reported_horizon(horizons)
    n_distinct = len({(d.ticker, d.pick_date, d.direction) for d in decisions})

    if reported is None:
        record = TrackRecord(
            n_picks_total=n_distinct,
            n_mature=0,
            n_pending=len(pending),
            reported_horizon_days=0,
            horizons=[],
            n_unmeasurable=len(no_data),
            unmeasurable=unmeasurable,
            mean_return_pct=None,
            mean_spy_return_pct=None,
            mean_alpha_pct=None,
            winners=0,
            losers=0,
            flats=0,
            overall_sharpe=None,
            buy_stats=_empty_direction(),
            hold_stats=_empty_direction(),
            trim_stats=_empty_direction(),
            sell_stats=_empty_direction(),
            model_breakdown=[],
            picks=[],
            pending=pending,
        )
    else:
        overall = reported.overall
        record = TrackRecord(
            n_picks_total=n_distinct,
            n_mature=len(reported.decisions),
            n_pending=len(pending),
            reported_horizon_days=reported.horizon_days,
            horizons=horizons,
            n_unmeasurable=len(no_data),
            unmeasurable=unmeasurable,
            mean_return_pct=overall.mean_return_pct,
            mean_spy_return_pct=overall.mean_spy_return_pct,
            mean_alpha_pct=overall.mean_alpha_pct,
            winners=overall.winners,
            losers=overall.losers,
            flats=overall.flats,
            overall_sharpe=overall.sharpe,
            buy_stats=reported.buy_stats,
            hold_stats=reported.hold_stats,
            trim_stats=reported.trim_stats,
            sell_stats=reported.sell_stats,
            model_breakdown=reported.model_breakdown,
            provider_breakdown=reported.provider_breakdown,
            picks=reported.decisions,
            pending=pending,
        )

    logger.info(
        "Track record: reported horizon=%sd; buys=%d hold=%d trim=%d sell=%d; "
        "%d pending; %d unmeasurable; overall_sharpe=%s",
        record.reported_horizon_days,
        record.buy_stats.n_mature,
        record.hold_stats.n_mature,
        record.trim_stats.n_mature,
        record.sell_stats.n_mature,
        record.n_pending,
        record.n_unmeasurable,
        f"{record.overall_sharpe:.2f}" if record.overall_sharpe is not None else "n/a",
    )
    return record


def _pick_reported_horizon(
    horizons: list[HorizonStats],
) -> HorizonStats | None:
    """The horizon the headline numbers describe: the primary window when
    it has data, otherwise the longest window that does. Never a blend."""
    if not horizons:
        return None
    for h in horizons:
        if h.horizon_days == _PRIMARY_HORIZON:
            return h
    return max(horizons, key=lambda h: h.horizon_days)


def _dedup_for_overall(rows: list[PickReturn]) -> list[PickReturn]:
    """One row per ticker for the cross-direction headline number.

    A name can legitimately be both a BUY pick (discover) and a HOLD
    verdict (rebalance). Both belong in their own direction stats, but
    counting the same ticker twice in the overall aggregate double-weights
    whatever you happen to hold. Keep the earliest decision per ticker.
    """
    best: dict[str, PickReturn] = {}
    for row in rows:
        current = best.get(row.ticker)
        if current is None or row.pick_date < current.pick_date:
            best[row.ticker] = row
    return list(best.values())


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
        out.append(
            ModelStats(
                opus_model=model,
                n_mature=len(alphas),
                mean_alpha_pct=sum(alphas) / len(alphas),
                sharpe=_sharpe(alphas),
            )
        )
    # mean_alpha_pct is always non-None for surviving rows (we just computed it
    # from a non-empty alphas list); the `or 0.0` is a typing-narrowing fallback.
    return sorted(
        out,
        key=lambda m: m.mean_alpha_pct if m.mean_alpha_pct is not None else 0.0,
        reverse=True,
    )


def _compute_provider_breakdown(
    buy_mature: list[PickReturn],
    ticker_providers: Mapping[str, list[str] | None],
) -> list[ProviderStats]:
    """Group mature BUY decisions by which provider(s) voted for them in
    the Ranker's multi-provider consensus. A pick with voting_providers =
    ['claude', 'openai'] counts toward BOTH buckets — this measures "when
    provider X was one of the ones that agreed, how did the pick do",
    not a disjoint partition the way `_compute_model_breakdown` groups by
    a single opus_model.

    Picks with no recorded voting_providers (single-round runs, or picks
    made before multi-provider consensus existed) are excluded entirely
    rather than bucketed as 'unknown' — there's no provider attribution
    to report for them, unlike `_compute_model_breakdown`'s 'unknown'
    opus_model bucket which is a real (if incomplete) data point.

    Same n_mature/threshold rules as `_compute_model_breakdown`: providers
    with fewer than 3 alpha-bearing picks are dropped as too noisy."""
    by_provider: dict[str, list[PickReturn]] = defaultdict(list)
    for p in buy_mature:
        for provider in ticker_providers.get(p.ticker) or []:
            by_provider[provider].append(p)
    out: list[ProviderStats] = []
    for provider, picks in by_provider.items():
        alphas = [p.alpha_pct for p in picks if p.alpha_pct is not None]
        if len(alphas) < 3:
            continue
        out.append(
            ProviderStats(
                provider=provider,
                n_mature=len(alphas),
                mean_alpha_pct=sum(alphas) / len(alphas),
                sharpe=_sharpe(alphas),
            )
        )
    return sorted(
        out,
        key=lambda m: m.mean_alpha_pct if m.mean_alpha_pct is not None else 0.0,
        reverse=True,
    )


def _aggregate(mature: list[PickReturn], *, pending_count: int = 0) -> DirectionStats:
    """Compute mean returns + win/loss/flat counts + Sharpe for a list of
    decisions measured over the SAME horizon. Alpha is already
    direction-aware (positive = right call), so we threshold ±0.5% on
    alpha regardless of direction."""
    if not mature:
        return DirectionStats(
            n_mature=0,
            n_pending=pending_count,
            mean_return_pct=None,
            mean_spy_return_pct=None,
            mean_alpha_pct=None,
            mean_beta_adjusted_alpha_pct=None,
            n_beta_adjusted=0,
            winners=0,
            losers=0,
            flats=0,
            sharpe=None,
        )
    mean_ret = sum(p.pick_return_pct or 0 for p in mature) / len(mature)
    mean_spy = sum(p.spy_return_pct or 0 for p in mature) / len(mature)
    mean_alpha = sum(p.alpha_pct or 0 for p in mature) / len(mature)
    winners = sum(1 for p in mature if (p.alpha_pct or 0) > 0.5)
    losers = sum(1 for p in mature if (p.alpha_pct or 0) < -0.5)
    flats = len(mature) - winners - losers
    alphas = [p.alpha_pct for p in mature if p.alpha_pct is not None]
    beta_alphas = [
        p.beta_adjusted_alpha_pct for p in mature if p.beta_adjusted_alpha_pct is not None
    ]
    return DirectionStats(
        n_mature=len(mature),
        n_pending=pending_count,
        mean_return_pct=mean_ret,
        mean_spy_return_pct=mean_spy,
        mean_alpha_pct=mean_alpha,
        mean_beta_adjusted_alpha_pct=(sum(beta_alphas) / len(beta_alphas) if beta_alphas else None),
        n_beta_adjusted=len(beta_alphas),
        winners=winners,
        losers=losers,
        flats=flats,
        sharpe=_sharpe(alphas),
    )


def _empty_direction() -> DirectionStats:
    return DirectionStats(
        n_mature=0,
        n_pending=0,
        mean_return_pct=None,
        mean_spy_return_pct=None,
        mean_alpha_pct=None,
        mean_beta_adjusted_alpha_pct=None,
        n_beta_adjusted=0,
        winners=0,
        losers=0,
        flats=0,
        sharpe=None,
    )


def _empty_record() -> TrackRecord:
    empty_dir = _empty_direction()
    return TrackRecord(
        n_picks_total=0,
        n_mature=0,
        n_pending=0,
        reported_horizon_days=0,
        horizons=[],
        n_unmeasurable=0,
        unmeasurable=[],
        mean_return_pct=None,
        mean_spy_return_pct=None,
        mean_alpha_pct=None,
        winners=0,
        losers=0,
        flats=0,
        overall_sharpe=None,
        buy_stats=empty_dir,
        hold_stats=empty_dir,
        trim_stats=empty_dir,
        sell_stats=empty_dir,
        model_breakdown=[],
        picks=[],
        pending=[],
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


def _beta_alpha_text(stats: DirectionStats) -> str:
    """Beta-adjusted alpha, or a note on why it's absent.

    Shown next to raw alpha everywhere, because the screen selects for
    high beta: raw alpha minus beta exposure is the part that is actually
    attributable to picking.
    """
    if stats.mean_beta_adjusted_alpha_pct is None:
        return "beta-adj n/a"
    return f"beta-adj {stats.mean_beta_adjusted_alpha_pct:+.1f}% (n={stats.n_beta_adjusted})"


def format_track_record_summary(record: TrackRecord) -> str:
    """One-line summary suitable for the dashboard / short prompt context.
    Renders the per-direction sub-totals for the reported horizon, which is
    always named so the window is never implicit."""
    if record.n_mature == 0:
        bits: list[str] = []
        if record.n_pending:
            bits.append(f"{record.n_pending} too young to score (min horizon {_MIN_AGE_DAYS}d)")
        if record.n_unmeasurable:
            bits.append(f"{record.n_unmeasurable} unmeasurable")
        if bits:
            return "Track record: 0 finished decisions yet (" + "; ".join(bits) + ")."
        return "Track record: no prior decisions in the lookback window."

    parts: list[str] = [f"Track record ({record.reported_horizon_days}d horizon):"]
    for label, stats in (
        ("Buy", record.buy_stats),
        ("Hold", record.hold_stats),
        ("Trim", record.trim_stats),
        ("Sell", record.sell_stats),
    ):
        if stats.n_mature:
            parts.append(
                f" {label} {stats.n_mature} scored, "
                f"alpha {_alpha_text(stats.mean_alpha_pct)} "
                f"[{_beta_alpha_text(stats)}] "
                f"({stats.winners}W/{stats.losers}L/{stats.flats}F)."
            )
    if record.n_pending:
        parts.append(f" {record.n_pending} pending.")
    if record.n_unmeasurable:
        parts.append(f" {record.n_unmeasurable} unmeasurable.")
    return "".join(parts)


def _format_decision_line(p: PickReturn) -> str:
    """Render one scored decision; non-buy decisions get a [VERDICT] tag
    so the user can tell directions apart in the listing."""
    tag = _DIRECTION_TAGS.get(p.direction, _UNKNOWN_DIRECTION_TAG)
    beta_bit = (
        f"  beta {p.beta:.2f} -> {p.beta_adjusted_alpha_pct:+.1f}%"
        if p.beta is not None and p.beta_adjusted_alpha_pct is not None
        else ""
    )
    return (
        f"  {tag} {p.ticker:6s}  {p.pick_date}  {p.horizon_days}d  "
        f"return {p.pick_return_pct:+.1f}%  "
        f"SPY {p.spy_return_pct:+.1f}%  "
        f"alpha {p.alpha_pct:+.1f}%{beta_bit}"
    )


def format_track_record_lines(record: TrackRecord, *, limit: int = 15) -> list[str]:
    """Per-decision lines for the report body. One section per direction
    with data at the reported horizon; each shows the top decisions by
    alpha."""
    lines: list[str] = []
    by_dir: dict[str, list[PickReturn]] = {
        "buy": [],
        "hold": [],
        "trim": [],
        "sell": [],
    }
    for p in record.picks:
        # pre-populated above; KeyError on an unknown direction is the right
        # failure mode (signals the Direction literal grew without updating
        # this dispatch table).
        by_dir[p.direction].append(p)

    section_headers = [
        ("buy", "  -- BUY picks --"),
        ("hold", "  -- HOLD verdicts --"),
        ("trim", "  -- TRIM verdicts --"),
        ("sell", "  -- SELL verdicts --"),
    ]
    for direction, header in section_headers:
        picks = sorted(
            by_dir[direction],
            key=lambda p: p.alpha_pct or 0,
            reverse=True,
        )
        if not picks:
            continue
        lines.append(f"{header} measured over {record.reported_horizon_days}d")
        for p in picks[:limit]:
            lines.append(_format_decision_line(p))

    if record.pending:
        lines.append(f"  -- pending (younger than {_MIN_AGE_DAYS}d; live mark, not scored) --")
        for p in sorted(record.pending, key=lambda p: p.pick_date, reverse=True)[:5]:
            tag = _DIRECTION_TAGS.get(p.direction, _UNKNOWN_DIRECTION_TAG)
            live_ret = f"live {p.pick_return_pct:+.1f}%" if p.pick_return_pct is not None else "—"
            lines.append(f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  {live_ret}")

    if record.n_unmeasurable:
        no_data = [u for u in record.unmeasurable if u.reason == "no_price_data"]
        if no_data:
            lines.append(
                "  -- unmeasurable (no forward price: delisted or bad symbol; "
                "NOT excluded from the count, and a likely loss) --"
            )
            for u in sorted(no_data, key=lambda u: u.pick_date)[:5]:
                tag = _DIRECTION_TAGS.get(u.direction, _UNKNOWN_DIRECTION_TAG)
                lines.append(f"  {tag} {u.ticker:6s}  {u.pick_date}  age {u.age_days}d")
    return lines


def _format_direction_block_line(
    label: str,
    stats: DirectionStats,
    *,
    is_first: bool,
) -> str:
    """One line per direction in the multi-line block — sample size, alpha,
    beta-adjusted alpha, Sharpe. The FIRST direction line
    (``is_first=True``) spells out "Sharpe (per-decision)" to communicate
    the unit; subsequent lines just say "Sharpe"."""
    sharpe_label = "Sharpe (per-decision)" if is_first else "Sharpe"
    return (
        f"{label}: {stats.n_mature} scored, "
        f"alpha {_alpha_text(stats.mean_alpha_pct)}, "
        f"{_beta_alpha_text(stats)}, "
        f"{sharpe_label} {_sharpe_text(stats.sharpe, stats.n_mature)}"
    )


def format_track_record_block(record: TrackRecord) -> str:
    """Multi-line block suitable for prepending to the ranker / rebalancer
    prompt as historical context.

    One block per measurement horizon, each labeled with its window —
    numbers are never averaged across horizons. Direction lines are
    emitted for every direction with at least one scored decision;
    model_breakdown is rendered per horizon when non-empty; per-decision
    detail for the reported horizon follows.
    """
    if record.n_picks_total == 0:
        return ""
    lines: list[str] = []
    for horizon in record.horizons:
        dir_lines: list[str] = []
        is_first = True
        for label, stats in [
            ("Buy", horizon.buy_stats),
            ("Hold", horizon.hold_stats),
            ("Trim", horizon.trim_stats),
            ("Sell", horizon.sell_stats),
        ]:
            if stats.n_mature:
                dir_lines.append(
                    _format_direction_block_line(
                        label,
                        stats,
                        is_first=is_first,
                    )
                )
                is_first = False
        if not dir_lines:
            continue
        lines.append(
            f"=== Measured over {horizon.horizon_days} days "
            f"(alpha vs SPY; beta-adj removes market exposure) ==="
        )
        lines.extend(dir_lines)
        if horizon.model_breakdown:
            model_parts = [
                f"{m.opus_model} ({m.n_mature} picks, {m.mean_alpha_pct:+.1f}%)"
                for m in horizon.model_breakdown
            ]
            lines.append("Model breakdown: " + " | ".join(model_parts))
        if horizon.provider_breakdown:
            provider_parts = [
                f"{p.provider} ({p.n_mature} picks, {p.mean_alpha_pct:+.1f}%)"
                for p in horizon.provider_breakdown
            ]
            lines.append("Provider breakdown: " + " | ".join(provider_parts))
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
        from ..data import yf_gateway

        end = date.fromisoformat(on)
        start = end - timedelta(days=7)
        df = yf_gateway.history(
            ticker,
            what="track_record.spot",
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
            "score_covered_call: spot lookup failed for %s on %s (%s); outcome will be UNKNOWN.",
            ticker,
            on,
            e,
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
