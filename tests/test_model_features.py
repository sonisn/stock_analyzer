"""The training-time (panel) and live (single ticker) feature code must
produce the same numbers, and agree with the screen's own indicators."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from stock_analyzer.data import technical_indicators as ti
from stock_analyzer.model.features import (
    FEATURES,
    panel_features,
    passes_trend_gate,
    ticker_features,
)
from tests.bars import bars, bdays


def _bars(seed: int, n: int = 600) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return bars(
        bdays(date(2023, 1, 2), periods=n),
        {
            "Close": close,
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float),
        },
    )


def _wide(tickers: dict[str, pl.DataFrame], col: str) -> pl.DataFrame:
    first = next(iter(tickers.values()))
    return pl.DataFrame({"date": first["date"], **{t: h[col] for t, h in tickers.items()}})


@pytest.fixture
def market():
    spy = _bars(0)
    tickers = {"AAA": _bars(1), "BBB": _bars(2)}
    return spy, tickers


@pytest.mark.parametrize("cut", [420, 457, 599])  # includes mid-week bars
def test_single_ticker_matches_panel(market, cut):
    spy, tickers = market
    spy_close = spy.select("date", pl.col("Close").alias("SPY"))
    panel = panel_features(
        _wide(tickers, "Close"), _wide(tickers, "High"), _wide(tickers, "Volume"), spy_close
    )
    day = spy["date"][cut]
    for t, hist in tickers.items():
        live = ticker_features(hist.filter(pl.col("date") <= day), spy_close)
        for name in FEATURES:
            expected = panel[name].filter(pl.col("date") == day)[t][0]
            assert live[name] == pytest.approx(expected, rel=1e-9, abs=1e-9), name


def test_matches_screen_indicators(market, monkeypatch):
    spy, tickers = market
    hist = tickers["AAA"]
    monkeypatch.setattr(ti, "_spy_history", lambda: spy)
    live = ticker_features(hist, spy.select("date", "Close"))
    assert live["rs_6mo"] == pytest.approx(ti._rs_vs_spy(hist, 6))
    assert live["rs_3mo"] == pytest.approx(ti._rs_vs_spy(hist, 3))
    assert live["dist_from_52w_high"] == pytest.approx(ti._distance_from_52w_high(hist))
    assert live["volume_trend_20_60"] == pytest.approx(ti._volume_trend(hist))
    assert live["weekly_rsi"] == pytest.approx(ti._rsi_weekly(hist), abs=1e-6)


def test_beta_of_a_levered_copy_is_the_leverage():
    spy = _bars(0)
    rets = (spy["Close"] / spy["Close"].shift(1) - 1).fill_null(0.0)
    levered = spy.with_columns((50 * (1 + 2 * rets).cum_prod()).alias("Close"))
    f = ticker_features(levered, spy.select("date", "Close"))
    assert f["beta_252"] == pytest.approx(2.0, rel=1e-6)


def test_trend_gate_matches_screen_rules():
    ok = {"px_vs_sma200": 0.1, "sma50_vs_sma200": 0.05, "rs_6mo": 0.02, "dist_from_52w_high": -0.1}
    assert passes_trend_gate(ok)
    assert not passes_trend_gate({**ok, "dist_from_52w_high": -0.35})
    assert not passes_trend_gate({**ok, "rs_6mo": None})
