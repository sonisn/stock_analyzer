"""Tests for ORATS IV rank fetcher."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_fetch_iv_ranks_no_key_returns_empty(monkeypatch):
    monkeypatch.delenv("ORATS_API_KEY", raising=False)
    from stock_analyzer.data.orats import fetch_iv_ranks
    assert fetch_iv_ranks(["NVDA", "AAPL"]) == {}


def test_fetch_iv_ranks_empty_input_short_circuits(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "fake")
    from stock_analyzer.data.orats import fetch_iv_ranks
    # No network call should be made.
    with patch("stock_analyzer.data.orats.requests.get") as mock_get:
        out = fetch_iv_ranks([])
    assert out == {}
    mock_get.assert_not_called()


def test_fetch_iv_ranks_parses_list_payload(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "fake-token")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [
        {
            "ticker": "NVDA", "tradeDate": "2026-05-14",
            "iv": 0.3215, "ivRank1y": 42.5, "ivPct1y": 38.2,
            "ivRank1m": 50.0, "ivPct1m": 48.0,
        },
        {
            "ticker": "AAPL", "tradeDate": "2026-05-14",
            "iv": 0.2102, "ivRank1y": 18.3, "ivPct1y": 15.6,
            "ivRank1m": 22.0, "ivPct1m": 20.0,
        },
    ]
    from stock_analyzer.data.orats import fetch_iv_ranks
    with patch("stock_analyzer.data.orats.requests.get", return_value=resp):
        out = fetch_iv_ranks(["NVDA", "AAPL"])
    assert "NVDA" in out and "AAPL" in out
    assert out["NVDA"].iv == 0.3215
    assert out["NVDA"].iv_rank_1y == 42.5
    assert out["AAPL"].iv_rank_1y == 18.3


def test_fetch_iv_ranks_parses_dict_data_envelope(monkeypatch):
    """ORATS sometimes wraps the list in {"data": [...]}; handle both."""
    monkeypatch.setenv("ORATS_API_KEY", "fake")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"data": [
        {"ticker": "NVDA", "iv": 0.30, "ivRank1y": 35.0,
         "ivPct1y": 30.0, "ivRank1m": 40.0, "ivPct1m": 38.0},
    ]}
    from stock_analyzer.data.orats import fetch_iv_ranks
    with patch("stock_analyzer.data.orats.requests.get", return_value=resp):
        out = fetch_iv_ranks(["NVDA"])
    assert out["NVDA"].iv == 0.30
    assert out["NVDA"].iv_rank_1y == 35.0


def test_fetch_iv_ranks_returns_empty_on_network_error(monkeypatch):
    monkeypatch.setenv("ORATS_API_KEY", "fake")
    from stock_analyzer.data.orats import fetch_iv_ranks
    with patch(
        "stock_analyzer.data.orats.requests.get",
        side_effect=RuntimeError("network down"),
    ):
        out = fetch_iv_ranks(["NVDA"])
    assert out == {}


def test_fetch_iv_ranks_skips_malformed_rows(monkeypatch):
    """Rows missing ticker or with non-numeric iv get silently skipped."""
    monkeypatch.setenv("ORATS_API_KEY", "fake")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = [
        {"iv": 0.30, "ivRank1y": 35.0},          # missing ticker
        {"ticker": "NVDA", "iv": "garbage"},     # bad iv
        {"ticker": "AAPL", "iv": 0.21, "ivRank1y": 18.0,
         "ivPct1y": 15.0, "ivRank1m": 22.0, "ivPct1m": 20.0},  # valid
    ]
    from stock_analyzer.data.orats import fetch_iv_ranks
    with patch("stock_analyzer.data.orats.requests.get", return_value=resp):
        out = fetch_iv_ranks(["NVDA", "AAPL"])
    assert "AAPL" in out
    assert "NVDA" not in out  # bad iv skipped
