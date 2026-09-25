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

import pandas as pd

from ..logging import get_logger
from . import yf_gateway

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
_SPY_HISTORY: pd.DataFrame | None = None
_SPY_FETCHED_AT: float = 0.0
_SPY_LOCK = threading.Lock()


def _two_years_ago() -> date:
    return date.today() - timedelta(days=730)


def _spy_history() -> pd.DataFrame | None:
    global _SPY_HISTORY, _SPY_FETCHED_AT
    with _SPY_LOCK:
        age = time.monotonic() - _SPY_FETCHED_AT
        if _SPY_HISTORY is None or age >= _SPY_CACHE_TTL_SECONDS:
            fetched = yf_gateway.daily_bars("SPY", start=_two_years_ago(), what="technicals.spy")
            if fetched is None:
                logger.warning("Failed to fetch SPY history — relative strength unavailable")
            _SPY_HISTORY = fetched if fetched is not None else pd.DataFrame()
            _SPY_FETCHED_AT = time.monotonic()
        cached = _SPY_HISTORY
    return cached if cached is not None and not cached.empty else None


def _sma(series: pd.Series, window: int) -> float | None:
    if len(series) < window:
        return None
    val = series.rolling(window).mean().iloc[-1]
    return float(val) if pd.notna(val) else None


def _rsi_weekly(history: pd.DataFrame, period: int = 14) -> float | None:
    if history.empty or len(history) < period * 5 + 5:
        return None
    weekly = history["Close"].resample("W").last().dropna()
    if len(weekly) < period + 1:
        return None
    delta = weekly.diff().dropna()
    if len(delta) < period:
        return None
    gains = delta.clip(lower=0.0)
    losses = (-delta).clip(lower=0.0)

    # Wilder's smoothing, not a simple rolling mean: seed with the simple
    # average of the first `period` deltas, then carry it forward with
    # avg = (avg * (period - 1) + new) / period. This is what "RSI" means
    # by convention, and the screen scores hard bands at 40 / 65 / 80 —
    # a simple mean puts enough of a shift on those edges to flip points.
    avg_gain = float(gains.iloc[:period].mean())
    avg_loss = float(losses.iloc[:period].mean())
    if pd.isna(avg_gain) or pd.isna(avg_loss):
        return None
    for gain, loss in zip(gains.iloc[period:], losses.iloc[period:], strict=True):
        avg_gain = (avg_gain * (period - 1) + float(gain)) / period
        avg_loss = (avg_loss * (period - 1) + float(loss)) / period

    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return float(100 - (100 / (1 + rs)))


def _rs_vs_spy(history: pd.DataFrame, months: int) -> float | None:
    """Ticker return minus SPY return over `months`, aligned by DATE.

    Positional alignment (`iloc[-days]` on two independently fetched
    frames) assumes the ticker and SPY have identical bar counts. That
    holds for a liquid US name on the NYSE calendar and breaks silently
    otherwise — a trading halt, an ADR on a different holiday calendar, or
    a stale final bar shifts one series relative to the other and the
    comparison spans two different windows. Since `rs_6mo > 0` is a hard
    filter, that arithmetic decides whether a name is screened out at all,
    so we intersect the two calendars first and take both endpoints off
    the joined frame.
    """
    spy = _spy_history()
    if spy is None or history.empty:
        return None
    joined = pd.DataFrame({"ticker": history["Close"], "spy": spy["Close"]}).dropna()
    days = months * _TRADING_DAYS_PER_MONTH
    if len(joined) <= days:
        return None
    t_now = float(joined["ticker"].iloc[-1])
    t_then = float(joined["ticker"].iloc[-1 - days])
    s_now = float(joined["spy"].iloc[-1])
    s_then = float(joined["spy"].iloc[-1 - days])
    if t_then == 0 or s_then == 0:
        return None
    return (t_now / t_then - 1) - (s_now / s_then - 1)


def _distance_from_52w_high(history: pd.DataFrame) -> float | None:
    """Current close vs the true 52-week high (intraday highs, not closes).

    The highest close understates the real high, so every name reads as
    less extended than it is — and this number drives both a hard
    rejection at -30% and the triangular entry-zone score peaked at -10%,
    where small shifts change points. Falls back to closes only if the
    High column is missing/empty.
    """
    if history.empty:
        return None
    window = history.tail(252)
    highs = window["High"] if "High" in window.columns else None
    if highs is not None and highs.notna().any():
        high = float(highs.max())
    else:
        high = float(window["Close"].max())
    current = float(history["Close"].iloc[-1])
    if high == 0:
        return None
    return (current - high) / high


def _volume_trend(history: pd.DataFrame) -> float | None:
    if history.empty or len(history) < 60:
        return None
    short = float(history["Volume"].tail(20).mean())
    long_ = float(history["Volume"].tail(60).mean())
    if long_ == 0:
        return None
    return (short / long_) - 1


def fetch_technicals(ticker: str) -> dict[str, Any] | None:
    hist = yf_gateway.daily_bars(ticker, start=_two_years_ago(), what="technicals")
    if hist is None or hist.empty:
        return None

    close = hist["Close"]
    price = float(close.iloc[-1])
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


def _model_features(hist: pd.DataFrame) -> dict[str, float | None]:
    """Inputs for the forward-return model (model/features.py), from the
    history already fetched here — no extra request."""
    from ..model.features import ticker_features

    spy = _spy_history()
    if spy is None:
        return {}
    try:
        return ticker_features(hist, spy["Close"])
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
