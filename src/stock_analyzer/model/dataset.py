"""Historical training set: weekly cross-sections of price features with
forward excess-return labels.

Fundamentals are optional and, when present, strictly point-in-time: see
`model/fundamental_features.py`, which reads each filing as of the date
it was published rather than as it reads today.

The universe is today's S&P 500 list (`data/universe_base.py`), which is
the main known bias: names that were dropped from the index — often after
falling hard — are missing, so absolute returns in the backtest look
better than reality. The model is judged on RANKING within each week,
which the bias distorts much less than it distorts levels.

Labels enter at the NEXT trading day's close, not the feature bar's own
close: the pipeline reads prices during the day and a buy lands later, so
scoring the same bar the features were computed on would credit the model
with a price nobody could trade at.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import yf_gateway
from ..logging import get_logger
from .features import FEATURES, panel_features

logger = get_logger(__name__)

HORIZONS: tuple[int, ...] = (21, 63)  # trading days ≈ 1 and 3 months
_CHUNK = 100
_CACHE_MAX_AGE_HOURS = 20


@dataclass
class PricePanel:
    close: pd.DataFrame
    high: pd.DataFrame
    volume: pd.DataFrame
    spy: pd.Series


def _field(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    if isinstance(frame.columns, pd.MultiIndex):
        level = 0 if name in frame.columns.get_level_values(0) else 1
        return frame.xs(name, axis=1, level=level)
    return frame[[name]]


def download_panel(tickers: list[str], *, years: int = 6) -> PricePanel:
    """Dividend-adjusted daily bars for `tickers` + SPY, in chunks of 100
    per yf.download call (each call is one paced request)."""
    symbols = sorted({t.upper() for t in tickers} | {"SPY"})
    closes, highs, volumes = [], [], []
    for i in range(0, len(symbols), _CHUNK):
        chunk = symbols[i : i + _CHUNK]
        frame = yf_gateway.download(
            chunk,
            what="model.panel",
            period=f"{years}y",
            auto_adjust=True,
            group_by="column",
            threads=True,
        )
        if frame is None or frame.empty:
            logger.warning("Price download returned nothing for %d symbols", len(chunk))
            continue
        closes.append(_field(frame, "Close"))
        highs.append(_field(frame, "High"))
        volumes.append(_field(frame, "Volume"))
        logger.info("Downloaded %d/%d symbols", min(i + _CHUNK, len(symbols)), len(symbols))
    if not closes:
        raise RuntimeError("No price data downloaded")
    close = pd.concat(closes, axis=1)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    high = pd.concat(highs, axis=1).set_axis(close.index)
    volume = pd.concat(volumes, axis=1).set_axis(close.index)
    if "SPY" not in close:
        raise RuntimeError("SPY missing from the price download")
    spy = close.pop("SPY")
    high = high.drop(columns="SPY", errors="ignore")
    volume = volume.drop(columns="SPY", errors="ignore")
    close = close.dropna(axis=1, how="all")
    return PricePanel(close, high[close.columns], volume[close.columns], spy)


def load_panel(tickers: list[str], cache_dir: str, *, years: int = 6) -> PricePanel:
    """download_panel, cached as a pickle for `_CACHE_MAX_AGE_HOURS` so a
    retrain the same day doesn't re-download ~500 symbols."""
    path = Path(os.path.expanduser(cache_dir)) / f"price_panel_{years}y.pkl"
    if path.exists() and time.time() - path.stat().st_mtime < _CACHE_MAX_AGE_HOURS * 3600:
        cached: PricePanel = pd.read_pickle(path)
        if set(t.upper() for t in tickers) <= set(cached.close.columns) | {"SPY"}:
            logger.info("Using cached price panel %s", path)
            return cached
    panel = download_panel(tickers, years=years)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(panel, path)
    return panel


def forward_returns(panel: PricePanel, horizon: int) -> tuple[pd.DataFrame, pd.Series]:
    """(date x ticker) return from the next bar's close to `horizon` bars
    after it, and SPY's return over the same bars. NaN past the data."""
    spy = panel.spy.reindex(panel.close.index)
    ret = panel.close.shift(-1 - horizon) / panel.close.shift(-1) - 1
    return ret, spy.shift(-1 - horizon) / spy.shift(-1) - 1


def forward_excess(panel: PricePanel, horizon: int) -> pd.DataFrame:
    """Forward return minus SPY's over the same bars."""
    ret, spy_ret = forward_returns(panel, horizon)
    return ret.sub(spy_ret, axis=0)


def build_dataset(
    panel: PricePanel,
    *,
    freq: str = "W-FRI",
    fundamentals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Long frame: one row per (date, ticker) on the last trading day of
    each week, with every feature, the trend-gate flag and two labels per
    horizon: `fwd_{h}` (excess over SPY) and `fwd_{h}_badj` (beta-neutral). Rows missing any feature are dropped; rows whose
    label is still in the future keep NaN labels (used for scoring only)."""
    feats = panel_features(panel.close, panel.high, panel.volume, panel.spy)
    cal = feats[FEATURES[0]].index
    weekly_last = pd.Series(cal, index=cal).groupby(cal.to_period(freq)).max()
    dates = pd.DatetimeIndex(weekly_last.to_numpy())

    parts = {name: feats[name].loc[dates].stack(future_stack=True) for name in FEATURES}
    beta = feats["beta_252"].loc[dates]
    for h in HORIZONS:
        ret, spy_ret = forward_returns(panel, h)
        ret, spy_ret = ret.reindex(cal).loc[dates], spy_ret.reindex(cal).loc[dates]
        parts[f"fwd_{h}"] = ret.sub(spy_ret, axis=0).stack(future_stack=True)
        # Beta-neutral: remove the market move the stock's trailing beta
        # (known on the feature date) implies, so a high-beta name is not
        # credited with skill for simply riding a rising market.
        parts[f"fwd_{h}_badj"] = (
            ret - beta.reindex(columns=ret.columns).mul(spy_ret, axis=0)
        ).stack(future_stack=True)
    frame = pd.DataFrame(parts)
    frame.index.names = ["date", "ticker"]
    frame = frame.dropna(subset=list(FEATURES))
    if fundamentals is not None and not fundamentals.empty:
        # Point-in-time only: `fundamentals` holds what each filing said
        # by its own as-of date, carried forward (model/fundamental_features).
        frame = frame.join(fundamentals, how="left")
    frame["gated"] = (
        (frame["px_vs_sma200"] > 0)
        & (frame["sma50_vs_sma200"] > 0)
        & (frame["rs_6mo"] > 0)
        & (frame["dist_from_52w_high"] >= -0.30)
    )
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=list(FEATURES))
