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

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..data import yf_gateway
from ..logging import get_logger
from .features import FEATURES, panel_features

logger = get_logger(__name__)

HORIZONS: tuple[int, ...] = (21, 63)  # trading days ≈ 1 and 3 months

# What the training set carries labels for. A year is here because the
# portfolio is held for three to five: margins and leverage have no
# business predicting a quarter of price action, so a 63-day test is not
# a fair test of whether fundamentals matter at all. It is deliberately
# NOT in HORIZONS, which drives what `labels.py` writes to
# `candidate_outcomes` — a 252-day outcome row takes a year to mature and
# that is a separate decision from what the model may train on.
DATASET_HORIZONS: tuple[int, ...] = (*HORIZONS, 252)


@dataclass
class PricePanel:
    close: pd.DataFrame
    high: pd.DataFrame
    volume: pd.DataFrame
    spy: pd.Series


def download_panel(tickers: list[str], *, years: int = 6) -> PricePanel:
    """Dividend-adjusted daily bars for `tickers` + SPY.

    Served from the on-disk bar store (`data/bar_store.py`): the first run
    downloads each symbol's history, later runs only the days since.
    """
    symbols = sorted({t.upper() for t in tickers} | {"SPY"})
    start = date.today() - timedelta(days=round(years * 365.25))
    frames = yf_gateway.daily_bars_many(symbols, start=start, what="model.panel")
    logger.info("Price panel: %d/%d symbols", len(frames), len(symbols))
    if not frames:
        raise RuntimeError("No price data downloaded")

    def field(name: str) -> pd.DataFrame:
        return pd.DataFrame({sym: f[name] for sym, f in frames.items() if name in f})

    close = field("Close")
    # DatetimeIndex gets .normalize() by delegation, which type checkers can't see.
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()  # ty: ignore[unresolved-attribute]
    high = field("High").set_axis(close.index)
    volume = field("Volume").set_axis(close.index)
    if "SPY" not in close:
        raise RuntimeError("SPY missing from the price download")
    spy = close.pop("SPY")
    high = high.drop(columns="SPY", errors="ignore")
    volume = volume.drop(columns="SPY", errors="ignore")
    close = close.dropna(axis=1, how="all")
    return PricePanel(close, high[close.columns], volume[close.columns], spy)


def load_panel(tickers: list[str], cache_dir: str | None = None, *, years: int = 6) -> PricePanel:
    """download_panel. `cache_dir` is unused: the panel used to be pickled
    there, and is now assembled from the bar store instead."""
    del cache_dir
    return download_panel(tickers, years=years)


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
    cal = pd.DatetimeIndex(feats[FEATURES[0]].index)
    weekly_last = pd.Series(cal, index=cal).groupby(cal.to_period(freq)).max()
    dates = pd.DatetimeIndex(weekly_last.to_numpy())

    parts = {name: feats[name].loc[dates].stack(future_stack=True) for name in FEATURES}
    beta = feats["beta_252"].loc[dates]
    for h in DATASET_HORIZONS:
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
