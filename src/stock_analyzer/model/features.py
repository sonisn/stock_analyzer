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

Frames are Polars and wide: a `date` column plus one column per ticker.
A missing price is null, which rolling windows and ranks skip — the rule
pandas applied to NaN — and results come back with NaN where a value
could not be computed, so callers test with `is_nan`/`is_null` alike.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from ..data.frames import DATE

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


def tickers_of(frame: pl.DataFrame) -> list[str]:
    return [c for c in frame.columns if c != DATE]


def on_calendar(frame: pl.DataFrame, calendar: pl.Series) -> pl.DataFrame:
    """`frame` on exactly `calendar`'s dates (missing rows null), NaN as null."""
    cal = pl.DataFrame({DATE: calendar})
    out = cal.join(frame, on=DATE, how="left")
    return out.with_columns(pl.col(tickers_of(out)).cast(pl.Float64).fill_nan(None))


def _each(frame: pl.DataFrame, fn) -> pl.DataFrame:
    """`fn(col_expr)` applied to every ticker column."""
    return frame.select(pl.col(DATE), *(fn(pl.col(t)).alias(t) for t in tickers_of(frame)))


def _excess(close: pl.DataFrame, spy: pl.Series, back: int, skip: int = 0) -> pl.DataFrame:
    """Return from `back` bars ago to `skip` bars ago, minus SPY's."""
    s = spy.shift(skip) / spy.shift(back) - 1
    return _each(close, lambda c: c.shift(skip) / c.shift(back) - 1 - pl.lit(s))


def _week_end(days: np.ndarray, weekday: int) -> np.ndarray:
    """The `weekday` (Mon=0) ending each date's week, as datetime64[D]."""
    dow = (days.astype("datetime64[D]").view("int64") - 4) % 7  # 1970-01-01 was a Thursday
    return days + ((weekday - dow) % 7).astype("timedelta64[D]")


def _weekly_rsi(close: pl.DataFrame) -> pl.DataFrame:
    """Wilder weekly RSI as seen on each DAILY bar, matching
    technical_indicators._rsi_weekly: completed weeks use their last close,
    and the week in progress counts as one more (partial) week ending on
    that bar. The Wilder state is carried per completed week, so each
    daily value is one extra smoothing step on top of it."""
    days = close[DATE].to_numpy().astype("datetime64[D]")
    daily = close.select(tickers_of(close)).to_numpy().astype(float)  # null -> NaN
    # Sunday-ending weeks, every week in the range (as pandas' "W" bins).
    ends = _week_end(days, 6)
    weeks = np.arange(ends.min(), ends.max() + np.timedelta64(1, "D"), np.timedelta64(7, "D"))
    slot = np.searchsorted(weeks, ends)
    n_w, n_c = len(weeks), daily.shape[1]
    wk = np.full((n_w, n_c), np.nan)
    for i, row in enumerate(daily):  # the week's last present close
        present = ~np.isnan(row)
        wk[slot[i], present] = row[present]

    delta = np.diff(wk, axis=0, prepend=np.nan)
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
    prev = np.searchsorted(weeks, days) - 1
    ok = prev >= 0
    safe = np.where(ok, prev, 0)
    base = np.where(ok[:, None], wk[safe], np.nan)
    g0 = np.where(ok[:, None], gain_state[safe], np.nan)
    l0 = np.where(ok[:, None], loss_state[safe], np.nan)
    d = daily - base
    g = (g0 * (RSI_PERIOD - 1) + np.clip(d, 0, None)) / RSI_PERIOD
    lo = (l0 * (RSI_PERIOD - 1) + np.clip(-d, 0, None)) / RSI_PERIOD
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + g / lo)
    rsi = np.where(lo == 0, np.where(g > 0, 100.0, 50.0), rsi)
    rsi = np.where(np.isnan(g) | np.isnan(lo), np.nan, rsi)
    return pl.DataFrame(
        {DATE: close[DATE], **{t: rsi[:, j] for j, t in enumerate(tickers_of(close))}}
    )


def _nan_for_null(frame: pl.DataFrame) -> pl.DataFrame:
    cols = tickers_of(frame)
    return frame.with_columns(pl.col(cols).fill_null(float("nan")))


def panel_features(
    close: pl.DataFrame,
    high: pl.DataFrame,
    volume: pl.DataFrame,
    spy_close: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Every feature as a wide (date x ticker) frame on SPY's trading
    calendar. Inputs are dividend-adjusted daily bars; `spy_close` is a
    (date, SPY) frame. Tickers are reindexed onto SPY's calendar so every
    lookback is a count of market days."""
    spy_col = next(c for c in spy_close.columns if c != DATE)
    spy_frame = spy_close.select(DATE, pl.col(spy_col).cast(pl.Float64).fill_nan(None))
    cal = spy_frame.drop_nulls(spy_col)[DATE]
    close = on_calendar(close, cal)
    high = on_calendar(high, cal)
    volume = on_calendar(volume, cal)
    spy = spy_frame.join(pl.DataFrame({DATE: cal}), on=DATE, how="right")[spy_col]
    m = TRADING_DAYS_PER_MONTH

    rets = _each(close, lambda c: c / c.shift(1) - 1)
    spy_rets = spy / spy.shift(1) - 1
    sma50 = _each(close, lambda c: c.rolling_mean(50))
    sma200 = _each(close, lambda c: c.rolling_mean(200))
    # Intraday highs over the trailing year, falling back to closes where a
    # High column is missing — as in technical_indicators.
    high_max = _each(high, lambda c: c.rolling_max(252, min_samples=1))
    close_max = _each(close, lambda c: c.rolling_max(252, min_samples=1))
    high_52w = high_max.select(
        pl.col(DATE),
        *(pl.col(t).fill_null(close_max[t]).alias(t) for t in tickers_of(close)),
    )

    beta_cols = []
    mean_y = spy_rets.rolling_mean(252, min_samples=200)
    var_y = spy_rets.rolling_var(252, min_samples=200, ddof=0)
    for t in tickers_of(close):
        x = rets[t]
        # pandas aligned the product before rolling: a missing SPY return
        # makes that day's product missing too.
        mean_xy = (x * spy_rets).rolling_mean(252, min_samples=200)
        mean_x = x.rolling_mean(252, min_samples=200)
        beta_cols.append(((mean_xy - mean_x * mean_y) / var_y).alias(t))
    beta = pl.DataFrame([close[DATE], *beta_cols])

    def ratio(a: pl.DataFrame, b: pl.DataFrame, minus: float = 1.0) -> pl.DataFrame:
        return a.select(pl.col(DATE), *((pl.col(t) / b[t] - minus).alias(t) for t in tickers_of(a)))

    frames = {
        "rs_1mo": _excess(close, spy, m),
        "rs_3mo": _excess(close, spy, 3 * m),
        "rs_6mo": _excess(close, spy, 6 * m),
        "rs_12_1": _excess(close, spy, 12 * m, skip=m),
        "dist_from_52w_high": close.select(
            pl.col(DATE),
            *(((pl.col(t) - high_52w[t]) / high_52w[t]).alias(t) for t in tickers_of(close)),
        ),
        "px_vs_sma200": ratio(close, sma200),
        "sma50_vs_sma200": ratio(sma50, sma200),
        "volume_trend_20_60": ratio(
            _each(volume, lambda c: c.rolling_mean(20)),
            _each(volume, lambda c: c.rolling_mean(60)),
        ),
        "weekly_rsi": _weekly_rsi(close),
        "vol_60d": _each(rets, lambda c: c.rolling_std(60) * math.sqrt(252)),
        "beta_252": beta,
    }
    return {name: _nan_for_null(f) for name, f in frames.items()}


def ticker_features(
    history: pl.DataFrame | None, spy_close: pl.DataFrame
) -> dict[str, float | None]:
    """The same features for one ticker at its latest bar. `history` is a
    bar frame (Close/High/Volume); `spy_close` is SPY's (date, Close)."""
    if history is None or history.is_empty():
        return {}
    last_day = history[DATE].max()
    x = {
        "Close": history.select(DATE, pl.col("Close").alias("x")),
        "High": history.select(
            DATE, pl.col("High" if "High" in history.columns else "Close").alias("x")
        ),
        "Volume": (
            history.select(DATE, pl.col("Volume").alias("x"))
            if "Volume" in history.columns
            else history.select(DATE, pl.lit(float("nan")).alias("x"))
        ),
    }
    spy_col = next(c for c in spy_close.columns if c != DATE)
    spy = spy_close.filter(pl.col(DATE) <= last_day).select(DATE, pl.col(spy_col).alias("SPY"))
    frames = panel_features(x["Close"], x["High"], x["Volume"], spy)
    out: dict[str, float | None] = {}
    for name in FEATURES:
        col = frames[name]["x"]
        val = col[-1] if len(col) else float("nan")
        out[name] = None if val is None or math.isnan(val) else float(val)
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


__all__ = [
    "FEATURES",
    "on_calendar",
    "panel_features",
    "passes_trend_gate",
    "ticker_features",
    "tickers_of",
]
