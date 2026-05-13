"""Tests for options_chain.py — providers, orchestrator, fallback."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from stock_analyzer.data.options_chain import (
    OptionChain,
    OptionQuote,
    TradierChain,
    YFinanceChain,
    fetch_chains,
)


def test_optionquote_frozen_and_typed():
    q = OptionQuote(
        strike=260.0, expiry=date(2026, 6, 20),
        bid=2.20, ask=2.40, iv=0.29, delta=0.36,
        open_interest=2890, volume=540,
    )
    assert q.strike == 260.0
    assert q.delta == 0.36


def test_optionchain_dataclass():
    chain = OptionChain(
        ticker="NVDA", spot=235.0, asof=datetime(2026, 5, 13, 16, 0, 0),
        calls=[], source="missing",
    )
    assert chain.ticker == "NVDA"
    assert chain.source == "missing"


def _fake_ticker(spot: float, expiries_to_calls: dict[str, pd.DataFrame]) -> MagicMock:
    """Build a MagicMock that mimics yfinance.Ticker."""
    t = MagicMock()
    t.fast_info = MagicMock(last_price=spot)
    t.options = tuple(expiries_to_calls.keys())
    t.option_chain.side_effect = lambda e: MagicMock(calls=expiries_to_calls[e])
    return t


def _calls_df(rows: list[tuple[float, float, float, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"],
    )


def test_yfinance_filters_to_dte_band_and_otm():
    today = date.today()
    e_in_band = (today + timedelta(days=35)).isoformat()
    e_too_close = (today + timedelta(days=10)).isoformat()
    e_too_far = (today + timedelta(days=120)).isoformat()
    chains = {
        e_in_band: _calls_df([
            (250.0, 3.10, 3.30, 0.31, 4210, 850),  # OTM
            (230.0, 8.00, 8.20, 0.33, 1000, 200),  # ITM — should be filtered
        ]),
        e_too_close: _calls_df([(260.0, 0.50, 0.60, 0.28, 100, 10)]),
        e_too_far: _calls_df([(260.0, 5.50, 5.60, 0.28, 100, 10)]),
    }
    fake = _fake_ticker(spot=235.0, expiries_to_calls=chains)
    with patch("stock_analyzer.data.options_chain.yf.Ticker", return_value=fake):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is not None
    assert chain.source == "yfinance"
    assert chain.spot == 235.0
    strikes = sorted(q.strike for q in chain.calls)
    assert strikes == [250.0]  # ITM 230 dropped, out-of-band expiries dropped


def test_yfinance_returns_none_on_error():
    with patch(
        "stock_analyzer.data.options_chain.yf.Ticker",
        side_effect=RuntimeError("network blew up"),
    ):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is None


def test_yfinance_no_expiries_returns_empty_chain_with_source_set():
    fake = _fake_ticker(spot=235.0, expiries_to_calls={})
    with patch("stock_analyzer.data.options_chain.yf.Ticker", return_value=fake):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is not None
    assert chain.calls == []
    assert chain.source == "yfinance"


_FIXTURES = Path(__file__).parent / "fixtures"


def test_tradier_returns_none_when_key_missing(monkeypatch):
    """TradierChain must degrade silently to None when the env var is unset."""
    monkeypatch.delenv("TRADIER_API_KEY", raising=False)
    from stock_analyzer.data.options_chain import TradierChain
    out = TradierChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert out is None


def test_tradier_parses_canned_chain(monkeypatch):
    """Happy path: expirations + chain fetch + Greeks extraction."""
    monkeypatch.setenv("TRADIER_API_KEY", "fake-token")
    monkeypatch.setenv("TRADIER_BASE_URL", "https://api.tradier.com/v1")

    today = date.today()
    in_band_str = (today + timedelta(days=35)).isoformat()
    out_of_band_str = (today + timedelta(days=120)).isoformat()

    def _fake_get(url, params=None, headers=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "expirations" in url:
            resp.json.return_value = {
                "expirations": {"date": [in_band_str, out_of_band_str]}
            }
        elif "chains" in url:
            resp.json.return_value = {
                "options": {
                    "option": [
                        {
                            "symbol": "NVDA260620C00260000",
                            "strike": 260, "bid": 2.2, "ask": 2.4,
                            "volume": 1105, "open_interest": 8249,
                            "option_type": "call",
                            "expiration_date": in_band_str,
                            "greeks": {"delta": 0.36, "mid_iv": 0.291},
                        },
                        {
                            "symbol": "NVDA260620P00260000",
                            "strike": 260, "bid": 1.0, "ask": 1.1,
                            "volume": 100, "open_interest": 50,
                            "option_type": "put",  # filtered out
                            "expiration_date": in_band_str,
                            "greeks": {"delta": -0.4, "mid_iv": 0.30},
                        },
                        {
                            "symbol": "NVDA260620C00230000",
                            "strike": 230, "bid": 8.0, "ask": 8.2,
                            "volume": 1000, "open_interest": 1000,
                            "option_type": "call",
                            "expiration_date": in_band_str,
                            "greeks": {"delta": 0.65, "mid_iv": 0.31},
                            # ITM (strike < spot=235), filtered out
                        },
                    ]
                }
            }
        elif "quotes" in url:
            resp.json.return_value = {
                "quotes": {"quote": {"symbol": "NVDA", "last": 235.0}}
            }
        else:
            resp.json.return_value = {}
        return resp

    with patch(
        "stock_analyzer.data.options_chain.requests.get",
        side_effect=_fake_get,
    ):
        chain = TradierChain().fetch("NVDA", dte_min=30, dte_max=45)

    assert chain is not None
    assert chain.source == "tradier"
    assert chain.spot == 235.0
    # Filtered to OTM calls only — strike 260 keeps, strike 230 (ITM) drops,
    # put row drops.
    strikes = sorted(q.strike for q in chain.calls)
    assert strikes == [260.0]
    q = chain.calls[0]
    assert q.delta == 0.36
    assert q.iv == 0.291
    assert q.open_interest == 8249


def test_tradier_handles_expirations_string_form(monkeypatch):
    """When only one expiry exists, Tradier returns `date` as a string,
    not a list. Must normalize."""
    monkeypatch.setenv("TRADIER_API_KEY", "fake")
    today = date.today()
    only_expiry = (today + timedelta(days=35)).isoformat()

    def _fake_get(url, params=None, headers=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        if "expirations" in url:
            resp.json.return_value = {"expirations": {"date": only_expiry}}
        elif "chains" in url:
            resp.json.return_value = {"options": {"option": []}}
        elif "quotes" in url:
            resp.json.return_value = {"quotes": {"quote": {"last": 100.0}}}
        else:
            resp.json.return_value = {}
        return resp

    with patch(
        "stock_analyzer.data.options_chain.requests.get",
        side_effect=_fake_get,
    ):
        chain = TradierChain().fetch("X", dte_min=30, dte_max=45)
    assert chain is not None
    # Empty chain (no options in fake response) but source set correctly.
    assert chain.source == "tradier"


def test_tradier_returns_none_on_network_error(monkeypatch):
    monkeypatch.setenv("TRADIER_API_KEY", "fake")
    with patch(
        "stock_analyzer.data.options_chain.requests.get",
        side_effect=RuntimeError("network down"),
    ):
        out = TradierChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert out is None


def test_fetch_chains_uses_tradier_when_available():
    fake_chain = OptionChain(
        ticker="NVDA", spot=235.0, asof=datetime.now(),
        calls=[], source="tradier",
    )
    with patch.object(TradierChain, "fetch", return_value=fake_chain) as tradier, \
         patch.object(YFinanceChain, "fetch") as yfin:
        out = fetch_chains(["NVDA"], dte_min=30, dte_max=45)
    tradier.assert_called_once()
    yfin.assert_not_called()
    assert out["NVDA"].source == "tradier"


def test_fetch_chains_falls_back_to_yfinance():
    fake = OptionChain(
        ticker="AAPL", spot=215.0, asof=datetime.now(),
        calls=[], source="yfinance",
    )
    with patch.object(TradierChain, "fetch", return_value=None), \
         patch.object(YFinanceChain, "fetch", return_value=fake):
        out = fetch_chains(["AAPL"], dte_min=30, dte_max=45)
    assert out["AAPL"].source == "yfinance"


def test_fetch_chains_marks_missing_when_all_fail():
    with patch.object(TradierChain, "fetch", return_value=None), \
         patch.object(YFinanceChain, "fetch", return_value=None):
        out = fetch_chains(["XYZ"], dte_min=30, dte_max=45)
    assert out["XYZ"].source == "missing"
    assert out["XYZ"].calls == []


def test_fetch_chains_empty_input():
    assert fetch_chains([], dte_min=30, dte_max=45) == {}


def test_yfinance_handles_nan_volume_and_open_interest():
    """Regression: yfinance returns NaN for low-volume strikes;
    int(NaN) raises ValueError. Must coerce to 0 cleanly."""
    today = date.today()
    e_in_band = (today + timedelta(days=35)).isoformat()
    nan = float("nan")
    df = pd.DataFrame(
        [(260.0, 2.20, 2.40, 0.29, nan, nan)],
        columns=["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"],
    )
    fake = _fake_ticker(spot=235.0, expiries_to_calls={e_in_band: df})
    with patch("stock_analyzer.data.options_chain.yf.Ticker", return_value=fake):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is not None
    assert len(chain.calls) == 1
    q = chain.calls[0]
    assert q.open_interest == 0
    assert q.volume == 0


def test_yfinance_handles_nan_bid_ask_iv():
    """NaN in bid/ask/iv must coerce to 0.0 / None, not propagate."""
    today = date.today()
    e_in_band = (today + timedelta(days=35)).isoformat()
    nan = float("nan")
    df = pd.DataFrame(
        [(260.0, nan, nan, nan, 100, 10)],
        columns=["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"],
    )
    fake = _fake_ticker(spot=235.0, expiries_to_calls={e_in_band: df})
    with patch("stock_analyzer.data.options_chain.yf.Ticker", return_value=fake):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is not None
    assert len(chain.calls) == 1
    q = chain.calls[0]
    assert q.bid == 0.0
    assert q.ask == 0.0
    assert q.iv is None  # None preserved for missing greeks/IV


