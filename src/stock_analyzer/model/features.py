"""Price-only features for the forward-return model, computed two ways.

`panel_features` computes every feature for every ticker on every date at
once (training); `ticker_features` computes the same numbers for one
ticker on its latest bar (live scoring inside the technicals fetch). The
definitions mirror `data/technical_indicators.py` where the screen already
uses a feature, and a test pins the two implementations to each other.

Only prices and volumes go in. They are the one input that is safe to
reconstruct after the fact — a historical close is the same number today
as it was then — so a multi-year training set can be rebuilt without
leaking the outcome into the features. Fundamentals from yfinance are a
live snapshot with no history and are deliberately excluded.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21
RSI_PERIOD = 14

FEATURES: tuple[str, ...] = (
    "rs_1mo",  # short-term reversal
    "rs_3mo",
    "rs_6mo",
    "rs_12_1",  # 12-month excess return, skipping the latest month
    "dist_from_52w_high",
    "px_vs_sma200",
    "sma50_vs_sma200",
    "volume_trend_20_60",
    "weekly_rsi",
    "vol_60d",
    "beta_252",
)


def _excess(close: pd.DataFrame | pd.Series, spy: pd.Series, back: int, skip: int = 0):
    """Return from `back` bars ago to `skip` bars ago, minus SPY's."""
    t = close.shift(skip) / close.shift(back) - 1
    s = spy.shift(skip) / spy.shift(back) - 1
    if isinstance(t, pd.DataFrame):
        return t.sub(s, axis=0)
    return t - s


def _weekly_rsi(close: pd.DataFrame) -> pd.DataFrame:
    """Wilder weekly RSI as seen on each DAILY bar, matching
    technical_indicators._rsi_weekly: completed weeks use their last close,
    and the week in progress counts as one more (partial) week ending on
    that bar. The Wilder state is carried per completed week, so each
    daily value is one extra smoothing step on top of it."""
    weekly = close.resample("W").last()
    wk = weekly.to_numpy(dtype=float)
    delta = np.diff(wk, axis=0, prepend=np.nan)
    n_w, n_c = wk.shape
    gain_state = np.full((n_w, n_c), np.nan)
    loss_state = np.full((n_w, n_c), np.nan)
    for j in range(n_c):
        d = delta[:, j]
        valid = np.flatnonzero(~np.isnan(d))
        if len(valid) < RSI_PERIOD:
            continue
        start = valid[0]
        seed = d[start : start + RSI_PERIOD]
        if np.isnan(seed).any():
            continue
        g = float(np.clip(seed, 0, None).mean())
        lo = float(np.clip(-seed, 0, None).mean())
        gain_state[start + RSI_PERIOD - 1, j] = g
        loss_state[start + RSI_PERIOD - 1, j] = lo
        for i in range(start + RSI_PERIOD, n_w):
            if not np.isnan(d[i]):
                g = (g * (RSI_PERIOD - 1) + max(d[i], 0.0)) / RSI_PERIOD
                lo = (lo * (RSI_PERIOD - 1) + max(-d[i], 0.0)) / RSI_PERIOD
            gain_state[i, j] = g
            loss_state[i, j] = lo

    # Week each daily bar belongs to, and the completed week before it.
    prev = weekly.index.searchsorted(close.index) - 1
    ok = prev >= 0
    safe = np.where(ok, prev, 0)
    base = np.where(ok[:, None], wk[safe], np.nan)
    g0 = np.where(ok[:, None], gain_state[safe], np.nan)
    l0 = np.where(ok[:, None], loss_state[safe], np.nan)
    d = close.to_numpy(dtype=float) - base
    g = (g0 * (RSI_PERIOD - 1) + np.clip(d, 0, None)) / RSI_PERIOD
    lo = (l0 * (RSI_PERIOD - 1) + np.clip(-d, 0, None)) / RSI_PERIOD
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + g / lo)
    rsi = np.where(lo == 0, np.where(g > 0, 100.0, 50.0), rsi)
    rsi = np.where(np.isnan(g) | np.isnan(lo), np.nan, rsi)
    return pd.DataFrame(rsi, index=close.index, columns=close.columns)


def panel_features(
    close: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    spy_close: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Every feature as a (date x ticker) frame on SPY's trading calendar.
    Inputs are dividend-adjusted daily bars; tickers are reindexed onto
    SPY's calendar so every lookback is a count of market days."""
    cal = spy_close.dropna().index
    close = close.reindex(cal)
    high = high.reindex(cal)
    volume = volume.reindex(cal)
    spy = spy_close.reindex(cal)
    m = TRADING_DAYS_PER_MONTH

    rets = close.pct_change(fill_method=None)
    spy_rets = spy.pct_change(fill_method=None)
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    # Intraday highs over the trailing year, falling back to closes where a
    # High column is missing — as in technical_indicators.
    high_52w = high.rolling(252, min_periods=1).max()
    high_52w = high_52w.fillna(close.rolling(252, min_periods=1).max())

    mean_xy = rets.mul(spy_rets, axis=0).rolling(252, min_periods=200).mean()
    mean_x = rets.rolling(252, min_periods=200).mean()
    mean_y = spy_rets.rolling(252, min_periods=200).mean()
    var_y = spy_rets.rolling(252, min_periods=200).var(ddof=0)
    beta = (mean_xy - mean_x.mul(mean_y, axis=0)).div(var_y, axis=0)

    return {
        "rs_1mo": _excess(close, spy, m),
        "rs_3mo": _excess(close, spy, 3 * m),
        "rs_6mo": _excess(close, spy, 6 * m),
        "rs_12_1": _excess(close, spy, 12 * m, skip=m),
        "dist_from_52w_high": (close - high_52w) / high_52w,
        "px_vs_sma200": close / sma200 - 1,
        "sma50_vs_sma200": sma50 / sma200 - 1,
        "volume_trend_20_60": volume.rolling(20).mean() / volume.rolling(60).mean() - 1,
        "weekly_rsi": _weekly_rsi(close),
        "vol_60d": rets.rolling(60).std() * math.sqrt(252),
        "beta_252": beta,
    }


def _as_dates(index: pd.Index) -> pd.DatetimeIndex:
    """yfinance returns tz-aware bars from Ticker.history and naive ones
    from download(); compare them as plain calendar dates."""
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


def ticker_features(history: pd.DataFrame, spy_close: pd.Series) -> dict[str, float | None]:
    """The same features for one ticker at its latest bar. `history` is a
    yfinance frame (Close/High/Volume); `spy_close` is SPY's closes."""
    if history is None or history.empty:
        return {}
    hist = history.copy()
    hist.index = _as_dates(hist.index)
    spy = spy_close.copy()
    spy.index = _as_dates(spy.index)
    x = {"Close": hist["Close"]}
    x["High"] = hist["High"] if "High" in hist else hist["Close"]
    x["Volume"] = hist["Volume"] if "Volume" in hist else pd.Series(np.nan, index=hist.index)
    frames = panel_features(
        x["Close"].to_frame("x"),
        x["High"].to_frame("x"),
        x["Volume"].to_frame("x"),
        spy.loc[spy.index <= hist.index[-1]],
    )
    out: dict[str, float | None] = {}
    for name in FEATURES:
        col = frames[name]["x"]
        val = col.iloc[-1] if len(col) else np.nan
        out[name] = None if pd.isna(val) else float(val)
    return out


def passes_trend_gate(f: dict[str, float | None]) -> bool:
    """The screen's four price rules (screen.passes_trend_gate) on this
    module's feature names: above the 200-day average, 50 above 200,
    positive 6-month relative strength, within 30% of the 52-week high."""
    px, ma, rs6, dist = (
        f.get("px_vs_sma200"),
        f.get("sma50_vs_sma200"),
        f.get("rs_6mo"),
        f.get("dist_from_52w_high"),
    )
    return (
        px is not None
        and px > 0
        and ma is not None
        and ma > 0
        and rs6 is not None
        and rs6 > 0
        and dist is not None
        and dist >= -0.30
    )
