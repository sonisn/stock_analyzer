"""The training-time (panel) and live (single ticker) feature code must
produce the same numbers, and agree with the screen's own indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analyzer.data import technical_indicators as ti
from stock_analyzer.model.features import (
    FEATURES,
    panel_features,
    passes_trend_gate,
    ticker_features,
)


def _bars(seed: int, n: int = 600, tz: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n, tz=tz)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return pd.DataFrame(
        {
            "Close": close,
            "High": close * (1 + rng.uniform(0, 0.01, n)),
            "Volume": rng.integers(1_000_000, 3_000_000, n).astype(float),
        },
        index=idx,
    )


@pytest.fixture
def market():
    spy = _bars(0)
    tickers = {"AAA": _bars(1), "BBB": _bars(2)}
    return spy, tickers


@pytest.mark.parametrize("cut", [420, 457, 599])  # includes mid-week bars
def test_single_ticker_matches_panel(market, cut):
    spy, tickers = market
    panel = panel_features(
        pd.DataFrame({t: h["Close"] for t, h in tickers.items()}),
        pd.DataFrame({t: h["High"] for t, h in tickers.items()}),
        pd.DataFrame({t: h["Volume"] for t, h in tickers.items()}),
        spy["Close"],
    )
    day = spy.index[cut]
    for t, hist in tickers.items():
        live = ticker_features(hist.loc[:day], spy["Close"])
        for name in FEATURES:
            assert live[name] == pytest.approx(panel[name].loc[day, t], rel=1e-9, abs=1e-9), name


def test_matches_screen_indicators_even_with_tz_aware_bars(market, monkeypatch):
    spy, tickers = market
    hist = _bars(1, tz="America/New_York")
    spy_tz = spy.tz_localize("America/New_York")
    monkeypatch.setattr(ti, "_spy_history", lambda: spy_tz)
    live = ticker_features(hist, spy_tz["Close"])
    assert live["rs_6mo"] == pytest.approx(ti._rs_vs_spy(hist, 6))
    assert live["rs_3mo"] == pytest.approx(ti._rs_vs_spy(hist, 3))
    assert live["dist_from_52w_high"] == pytest.approx(ti._distance_from_52w_high(hist))
    assert live["volume_trend_20_60"] == pytest.approx(ti._volume_trend(hist))
    assert live["weekly_rsi"] == pytest.approx(ti._rsi_weekly(hist), abs=1e-6)


def test_beta_of_a_levered_copy_is_the_leverage():
    spy = _bars(0)
    rets = spy["Close"].pct_change().fillna(0)
    levered = spy.copy()
    levered["Close"] = 50 * (1 + 2 * rets).cumprod()
    f = ticker_features(levered, spy["Close"])
    assert f["beta_252"] == pytest.approx(2.0, rel=1e-6)


def test_trend_gate_matches_screen_rules():
    ok = {"px_vs_sma200": 0.1, "sma50_vs_sma200": 0.05, "rs_6mo": 0.02, "dist_from_52w_high": -0.1}
    assert passes_trend_gate(ok)
    assert not passes_trend_gate({**ok, "dist_from_52w_high": -0.35})
    assert not passes_trend_gate({**ok, "rs_6mo": None})
