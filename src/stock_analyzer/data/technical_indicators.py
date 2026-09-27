"""Technical indicators for mid-to-long term holds.

Pure-math wrappers over yfinance OHLCV. No LLM, no external services.
Indicator choice is deliberate for 6-12 month holds:
  - 50/200 SMA: trend health (Stage 2 base = 50>200 + price>200)
  - RS vs SPY: institutional accumulation signal
  - 52w high distance: entry-zone vs extended
  - Volume trend: accumulation confirmation
  - Weekly RSI: momentum without exhaustion
Day-trading indicators (MACD, Bollinger, intraday RSI) are intentionally omitted.
"""

from __future__ import annotations

import threading
import time
from datetime import date, timedelta
from typing import Any

import polars as pl

from ..logging import get_logger
from . import yf_gateway
from .frames import DATE

logger = get_logger(__name__)

# See fundamentals._MAX_WORKERS: fan-out width only — yf_gateway caps the
# actual concurrent-request count for the whole process.
_MAX_WORKERS = 8
_TRADING_DAYS_PER_MONTH = 21
# SPY history is the denominator for every RS calculation, so it is cached
# once per batch rather than refetched per ticker. The TTL keeps a
# long-lived process (a service, a loop) from comparing today's tickers
# against a days-old SPY series; the lock keeps the parallel fetches in
# batch_technicals from stampeding the same refresh.
_SPY_CACHE_TTL_SECONDS = 3600.0
_SPY_HISTORY: pl.DataFrame | None = None
_SPY_FETCHED_AT: float = 0.0
_SPY_LOCK = threading.Lock()


def _two_years_ago() -> date:
    return date.today() - timedelta(days=730)


def _spy_history() -> pl.DataFrame | None:
    global _SPY_HISTORY, _SPY_FETCHED_AT
    with _SPY_LOCK:
        age = time.monotonic() - _SPY_FETCHED_AT
        if _SPY_HISTORY is None or age >= _SPY_CACHE_TTL_SECONDS:
            fetched = yf_gateway.daily_bars("SPY", start=_two_years_ago(), what="technicals.spy")
            if fetched is None:
                logger.warning("Failed to fetch SPY history — relative strength unavailable")
            _SPY_HISTORY = fetched if fetched is not None else pl.DataFrame()
            _SPY_FETCHED_AT = time.monotonic()
        cached = _SPY_HISTORY
    return cached if cached is not None and not cached.is_empty() else None


def _float(value: Any) -> float | None:
    return None if value is None or value != value else float(value)


def _sma(series: pl.Series, window: int) -> float | None:
    if len(series) < window:
        return None
    return _float(series.rolling_mean(window)[-1])


def _weekly_closes(history: pl.DataFrame) -> pl.Series:
    """Each week's last close (weeks ending Sunday, as pandas' "W"), weeks
    with no close left out."""
    return (
        history.select(DATE, pl.col("Close").fill_nan(None))
        .drop_nulls("Close")
        .with_columns((pl.col(DATE) + pl.duration(days=7 - pl.col(DATE).dt.weekday())).alias("_wk"))
        .group_by("_wk")
        .agg(pl.col("Close").last())
        .sort("_wk")["Close"]
    )


def _rsi_weekly(history: pl.DataFrame, period: int = 14) -> float | None:
    if history.is_empty() or history.height < period * 5 + 5:
        return None
    weekly = _weekly_closes(history)
    if len(weekly) < period + 1:
        return None
    delta = weekly.diff().drop_nulls()
    if len(delta) < period:
        return None
    gains = delta.clip(lower_bound=0.0).to_list()
    losses = (-delta).clip(lower_bound=0.0).to_list()

    # Wilder's smoothing, not a simple rolling mean: seed with the simple
    # average of the first `period` deltas, then carry it forward with
    # avg = (avg * (period - 1) + new) / period. This is what "RSI" means
    # by convention, and the screen scores hard bands at 40 / 65 / 80 —
    # a simple mean puts enough of a shift on those edges to flip points.
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + float(gain)) / period
        avg_loss = (avg_loss * (period - 1) + float(loss)) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _rs_vs_spy(history: pl.DataFrame, months: int) -> float | None:
    """Ticker return minus SPY return over `months`, aligned by DATE.

    Positional alignment on two independently fetched frames assumes the
    ticker and SPY have identical bar counts. That holds for a liquid US
    name on the NYSE calendar and breaks silently otherwise — a trading
    halt, an ADR on a different holiday calendar, or a stale final bar
    shifts one series relative to the other and the comparison spans two
    different windows. Since `rs_6mo > 0` is a hard filter, that
    arithmetic decides whether a name is screened out at all, so we
    intersect the two calendars first and take both endpoints off the
    joined frame.
    """
    spy = _spy_history()
    if spy is None or history.is_empty():
        return None
    joined = (
        history.select(DATE, pl.col("Close").alias("ticker"))
        .join(spy.select(DATE, pl.col("Close").alias("spy")), on=DATE)
        .drop_nulls()
        .sort(DATE)
    )
    days = months * _TRADING_DAYS_PER_MONTH
    if joined.height <= days:
        return None
    t_now, t_then = float(joined["ticker"][-1]), float(joined["ticker"][-1 - days])
    s_now, s_then = float(joined["spy"][-1]), float(joined["spy"][-1 - days])
    if t_then == 0 or s_then == 0:
        return None
    return (t_now / t_then - 1) - (s_now / s_then - 1)


def _distance_from_52w_high(history: pl.DataFrame) -> float | None:
    """Current close vs the true 52-week high (intraday highs, not closes).

    The highest close understates the real high, so every name reads as
    less extended than it is — and this number drives both a hard
    rejection at -30% and the triangular entry-zone score peaked at -10%,
    where small shifts change points. Falls back to closes only if the
    High column is missing/empty.
    """
    if history.is_empty():
        return None
    window = history.tail(252)
    highs = window["High"] if "High" in window.columns else None
    if highs is not None and highs.drop_nulls().len() > 0:
        high = float(highs.max())  # ty: ignore[invalid-argument-type]
    else:
        high = float(window["Close"].max())  # ty: ignore[invalid-argument-type]
    current = _float(history["Close"][-1])
    if high == 0 or current is None:
        return None
    return (current - high) / high


def _volume_trend(history: pl.DataFrame) -> float | None:
    if history.is_empty() or history.height < 60 or "Volume" not in history.columns:
        return None
    short = _float(history["Volume"].tail(20).mean())
    long_ = _float(history["Volume"].tail(60).mean())
    if not long_ or short is None:
        return None
    return (short / long_) - 1


def fetch_technicals(ticker: str) -> dict[str, Any] | None:
    hist = yf_gateway.daily_bars(ticker, start=_two_years_ago(), what="technicals")
    if hist is None or hist.is_empty():
        return None

    close = hist["Close"]
    price = _float(close[-1])
    if price is None:
        return None
    sma50 = _sma(close, 50)
    sma200 = _sma(close, 200)

    return {
        "ticker": ticker,
        "price": price,
        "sma_50": sma50,
        "sma_200": sma200,
        "above_200dma": sma200 is not None and price > sma200,
        "ma_alignment_50_200": (sma50 is not None and sma200 is not None and sma50 > sma200),
        "rs_3mo": _rs_vs_spy(hist, 3),
        "rs_6mo": _rs_vs_spy(hist, 6),
        "dist_from_52w_high": _distance_from_52w_high(hist),
        "volume_trend_20_60": _volume_trend(hist),
        "weekly_rsi": _rsi_weekly(hist),
        "model_features": _model_features(hist),
    }


def _model_features(hist: pl.DataFrame) -> dict[str, float | None]:
    """Inputs for the forward-return model (model/features.py), from the
    history already fetched here — no extra request."""
    from ..model.features import ticker_features

    spy = _spy_history()
    if spy is None:
        return {}
    try:
        return ticker_features(hist, spy.select(DATE, "Close"))
    except Exception as e:  # noqa: BLE001 — a feature bug must not drop the ticker
        logger.warning("model features failed: %s", e)
        return {}


def batch_technicals(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch technicals for many tickers in parallel. Warms SPY cache once."""
    _spy_history()
    results: dict[str, dict[str, Any]] = {}
    for ticker, r in yf_gateway.map_symbols(fetch_technicals, tickers, workers=_MAX_WORKERS):
        if r:
            results[ticker] = r
    return results
