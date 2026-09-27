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

import polars as pl

from ..data import yf_gateway
from ..data.frames import DATE
from ..logging import get_logger
from .features import FEATURES, panel_features, tickers_of

logger = get_logger(__name__)

HORIZONS: tuple[int, ...] = (21, 63, 126)  # trading days ≈ 1, 3 and 6 months

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
    """Wide Polars frames (a `date` column plus one column per ticker) on
    the union of the symbols' trading days, and SPY as a (date, SPY) frame
    on the same dates."""

    close: pl.DataFrame
    high: pl.DataFrame
    volume: pl.DataFrame
    spy: pl.DataFrame

    @property
    def tickers(self) -> list[str]:
        return tickers_of(self.close)

    def calendar(self) -> pl.Series:
        """SPY's trading days: the dates it has a close for."""
        spy = self.spy.with_columns(pl.col("SPY").fill_nan(None))
        return spy.drop_nulls("SPY")[DATE]


def _wide(frames: dict[str, pl.DataFrame], field: str) -> pl.DataFrame:
    """One column per symbol of `field`, on the union of their dates."""
    long = pl.concat(
        [
            f.select(DATE, pl.lit(sym).alias("ticker"), pl.col(field).cast(pl.Float64))
            for sym, f in sorted(frames.items())
            if field in f.columns
        ]
    )
    wide = long.pivot(on="ticker", index=DATE, values=field).sort(DATE)
    order = [s for s in sorted(frames) if s in wide.columns]
    return wide.select(DATE, *order).with_columns(pl.col(order).fill_nan(None))


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
    return panel_from_bars(frames)


def panel_from_bars(frames: dict[str, pl.DataFrame]) -> PricePanel:
    """A PricePanel from {symbol: bar frame}; SPY must be among them."""
    close = _wide(frames, "Close")
    if "SPY" not in close.columns:
        raise RuntimeError("SPY missing from the price download")
    spy = close.select(DATE, "SPY")
    close = close.drop("SPY")
    # A symbol with no close on any day carries nothing.
    keep = [t for t in tickers_of(close) if close[t].null_count() < close.height]
    close = close.select(DATE, *keep)
    dates = close.select(DATE)

    def aligned(field: str) -> pl.DataFrame:
        wide = _wide(frames, field) if any(field in f.columns for f in frames.values()) else dates
        out = dates.join(wide, on=DATE, how="left")
        missing = [t for t in keep if t not in out.columns]
        out = out.with_columns([pl.lit(None, dtype=pl.Float64).alias(t) for t in missing])
        return out.select(DATE, *keep)

    return PricePanel(close, aligned("High"), aligned("Volume"), spy)


def load_panel(tickers: list[str], cache_dir: str | None = None, *, years: int = 6) -> PricePanel:
    """download_panel. `cache_dir` is unused: the panel used to be pickled
    there, and is now assembled from the bar store instead."""
    del cache_dir
    return download_panel(tickers, years=years)


def forward_returns(panel: PricePanel, horizon: int) -> tuple[pl.DataFrame, pl.Series]:
    """Wide return from the next bar's close to `horizon` bars after it, on
    the panel's dates, and SPY's return over the same bars. Missing past
    the data."""
    spy = panel.close.select(DATE).join(panel.spy, on=DATE, how="left")["SPY"].fill_nan(None)
    ret = panel.close.select(
        pl.col(DATE),
        *(
            (pl.col(t).shift(-1 - horizon) / pl.col(t).shift(-1) - 1).alias(t)
            for t in panel.tickers
        ),
    )
    return ret, spy.shift(-1 - horizon) / spy.shift(-1) - 1


def forward_excess(panel: PricePanel, horizon: int) -> pl.DataFrame:
    """Forward return minus SPY's over the same bars."""
    ret, spy_ret = forward_returns(panel, horizon)
    return ret.select(pl.col(DATE), *((pl.col(t) - spy_ret).alias(t) for t in panel.tickers))


def _week_end(weekday: int) -> pl.Expr:
    """The `weekday` (ISO: Mon=1 .. Sun=7) ending each `date`'s week — the
    period pandas calls "W-FRI" for weekday 5."""
    return pl.col(DATE) + pl.duration(days=(weekday - pl.col(DATE).dt.weekday()) % 7)


def build_dataset(
    panel: PricePanel,
    *,
    freq: str = "W-FRI",
    fundamentals: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Long frame: one row per (date, ticker) on the last trading day of
    each week, with every feature, the trend-gate flag and two labels per
    horizon: `fwd_{h}` (excess over SPY) and `fwd_{h}_badj` (beta-neutral).
    Rows missing any feature are dropped; rows whose label is still in the
    future keep NaN labels (used for scoring only). Sorted by date, then the
    panel's ticker order."""
    if freq != "W-FRI":
        raise ValueError("weekly (W-FRI) sampling only")
    feats = panel_features(panel.close, panel.high, panel.volume, panel.spy)
    cal = feats[FEATURES[0]].select(DATE)
    weekly = (
        cal.with_columns(_week_end(5).alias("_wk"))
        .group_by("_wk")
        .agg(pl.col(DATE).max())
        .sort(DATE)
    )
    dates = weekly.select(DATE)
    tickers = panel.tickers
    order = pl.DataFrame({"ticker": tickers, "_pos": range(len(tickers))})

    def long(frame: pl.DataFrame, name: str) -> pl.DataFrame:
        at = dates.join(frame, on=DATE, how="left")
        return at.unpivot(index=DATE, on=tickers, variable_name="ticker", value_name=name)

    out = long(feats[FEATURES[0]], FEATURES[0])
    for name in FEATURES[1:]:
        out = out.join(long(feats[name], name), on=[DATE, "ticker"], how="left")
    beta_at = dates.join(feats["beta_252"], on=DATE, how="left")
    for h in DATASET_HORIZONS:
        ret, spy_ret = forward_returns(panel, h)
        ret = ret.with_columns(spy_ret.alias("_spy"))
        at = dates.join(ret, on=DATE, how="left")
        spy_at = at["_spy"]
        excess = at.select(pl.col(DATE), *((pl.col(t) - spy_at).alias(t) for t in tickers))
        # Beta-neutral: remove the market move the stock's trailing beta
        # (known on the feature date) implies, so a high-beta name is not
        # credited with skill for simply riding a rising market.
        badj = at.select(
            pl.col(DATE), *((pl.col(t) - beta_at[t] * spy_at).alias(t) for t in tickers)
        )
        out = out.join(long(excess, f"fwd_{h}"), on=[DATE, "ticker"], how="left")
        out = out.join(long(badj, f"fwd_{h}_badj"), on=[DATE, "ticker"], how="left")
    floats = [c for c in out.columns if c not in (DATE, "ticker")]
    out = out.with_columns(pl.col(floats).fill_null(float("nan")))
    out = out.join(order, on="ticker").sort([DATE, "_pos"]).drop("_pos")
    out = out.filter(pl.all_horizontal(pl.col(list(FEATURES)).is_not_nan()))
    if fundamentals is not None and not fundamentals.is_empty():
        # Point-in-time only: `fundamentals` holds what each filing said
        # by its own as-of date, carried forward (model/fundamental_features).
        out = out.join(fundamentals, on=[DATE, "ticker"], how="left")
    out = out.with_columns(
        (
            (pl.col("px_vs_sma200") > 0)
            & (pl.col("sma50_vs_sma200") > 0)
            & (pl.col("rs_6mo") > 0)
            & (pl.col("dist_from_52w_high") >= -0.30)
        ).alias("gated")
    )
    floats = [c for c, t in out.schema.items() if t == pl.Float64]
    out = out.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(float("nan")).otherwise(pl.col(c)).alias(c)
            for c in floats
        ]
    )
    return out.filter(pl.all_horizontal(pl.col(list(FEATURES)).is_not_nan()))


__all__ = [
    "DATASET_HORIZONS",
    "HORIZONS",
    "PricePanel",
    "build_dataset",
    "download_panel",
    "forward_excess",
    "forward_returns",
    "load_panel",
    "panel_from_bars",
]
