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

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 5
_TRADING_DAYS_PER_MONTH = 21
_SPY_HISTORY: pd.DataFrame | None = None


def _spy_history() -> pd.DataFrame | None:
    global _SPY_HISTORY
    if _SPY_HISTORY is None:
        try:
            _SPY_HISTORY = yf.Ticker("SPY").history(period="2y", auto_adjust=True)
        except Exception as e:
            logger.warning("Failed to fetch SPY history: %s", e)
            _SPY_HISTORY = pd.DataFrame()
    return _SPY_HISTORY if _SPY_HISTORY is not None and not _SPY_HISTORY.empty else None


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
    gains = delta.where(delta > 0, 0.0).rolling(period).mean()
    losses = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    last_gain = gains.iloc[-1]
    last_loss = losses.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss):
        return None
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _rs_vs_spy(history: pd.DataFrame, months: int) -> float | None:
    spy = _spy_history()
    if spy is None or history.empty:
        return None
    days = months * _TRADING_DAYS_PER_MONTH
    if len(history) < days or len(spy) < days:
        return None
    t_now = float(history["Close"].iloc[-1])
    t_then = float(history["Close"].iloc[-days])
    s_now = float(spy["Close"].iloc[-1])
    s_then = float(spy["Close"].iloc[-days])
    if t_then == 0 or s_then == 0:
        return None
    return (t_now / t_then - 1) - (s_now / s_then - 1)


def _distance_from_52w_high(history: pd.DataFrame) -> float | None:
    if history.empty:
        return None
    high = float(history["Close"].tail(252).max())
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
    try:
        hist = yf.Ticker(ticker).history(period="2y", auto_adjust=True)
    except Exception as e:
        logger.warning("technicals fetch failed for %s: %s", ticker, e)
        return None
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
        "ma_alignment_50_200": (
            sma50 is not None and sma200 is not None and sma50 > sma200
        ),
        "rs_3mo": _rs_vs_spy(hist, 3),
        "rs_6mo": _rs_vs_spy(hist, 6),
        "dist_from_52w_high": _distance_from_52w_high(hist),
        "volume_trend_20_60": _volume_trend(hist),
        "weekly_rsi": _rsi_weekly(hist),
    }


def batch_technicals(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch technicals for many tickers in parallel. Warms SPY cache once."""
    _spy_history()
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, r in zip(tickers, ex.map(fetch_technicals, tickers)):
            if r:
                results[ticker] = r
    return results
