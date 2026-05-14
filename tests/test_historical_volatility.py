"""Tests for realized-volatility computation."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_annualized_vol_from_closes_matches_handcalc():
    """Stable returns of ~1% daily should produce ~16% annualized
    (since stdev ≈ 0.01, * sqrt(252) ≈ 0.159)."""
    from stock_analyzer.data.historical_volatility import (
        _annualized_vol_from_closes,
    )
    # 60 days of closes with a deliberate 1%-stdev log return.
    closes = [100.0]
    for i in range(60):
        # Alternate +1% / -1% to get exactly 1% absolute moves.
        closes.append(closes[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
    hv = _annualized_vol_from_closes(closes)
    assert hv is not None
    # 1% daily moves → annualized ~15.85%.
    assert 0.12 < hv < 0.20


def test_annualized_vol_from_closes_returns_none_for_short_input():
    from stock_analyzer.data.historical_volatility import (
        _annualized_vol_from_closes,
    )
    assert _annualized_vol_from_closes([100.0] * 5) is None


def test_annualized_vol_from_closes_returns_none_for_constant():
    from stock_analyzer.data.historical_volatility import (
        _annualized_vol_from_closes,
    )
    assert _annualized_vol_from_closes([100.0] * 40) is None


def test_fetch_realized_volatility_empty_input():
    from stock_analyzer.data.historical_volatility import (
        fetch_realized_volatility,
    )
    assert fetch_realized_volatility([]) == {}


def test_fetch_realized_volatility_happy_path():
    """Mock yfinance history; verify HV is computed for the ticker."""
    import pandas as pd

    from stock_analyzer.data.historical_volatility import (
        fetch_realized_volatility,
    )
    # 252 days of 1%-stdev returns.
    closes = [100.0]
    import random
    random.seed(42)
    for _ in range(260):
        closes.append(closes[-1] * (1 + random.gauss(0.0005, 0.01)))
    df = pd.DataFrame({"Close": closes})
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = df
    fake_yf = MagicMock()
    fake_yf.Ticker.return_value = fake_ticker
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        out = fetch_realized_volatility(["NVDA"])
    assert "NVDA" in out
    assert 0.10 < out["NVDA"].hv_annualized < 0.30  # ~15-20% with 1% daily


def test_fetch_realized_volatility_empty_df_returns_empty():
    import pandas as pd

    from stock_analyzer.data.historical_volatility import (
        fetch_realized_volatility,
    )
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = pd.DataFrame()
    fake_yf = MagicMock()
    fake_yf.Ticker.return_value = fake_ticker
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        out = fetch_realized_volatility(["NVDA"])
    assert out == {}


def test_fetch_realized_volatility_swallows_yf_errors():
    from stock_analyzer.data.historical_volatility import (
        fetch_realized_volatility,
    )
    fake_yf = MagicMock()
    fake_yf.Ticker.side_effect = RuntimeError("network down")
    with patch.dict("sys.modules", {"yfinance": fake_yf}):
        out = fetch_realized_volatility(["NVDA"])
    assert out == {}
