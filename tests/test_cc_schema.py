"""Tests for CC config + RebalancePlan/OptionWrite schema."""
from __future__ import annotations

from stock_analyzer.config import Settings


def test_cc_defaults():
    s = Settings()  # type: ignore[call-arg]
    assert s.cc_enabled is True
    assert s.cc_target_delta_min == 0.35
    assert s.cc_target_delta_max == 0.45
    assert s.cc_dte_min == 30
    assert s.cc_dte_max == 45
    assert s.cc_denylist == ()
    assert s.cc_min_premium_usd == 500
    assert s.cc_slippage_buffer == 0.10
    assert s.cc_stub_optimization is True
    assert s.cc_min_stub_usd == 1000


def test_cc_denylist_parses_csv(monkeypatch):
    monkeypatch.setenv("CC_DENYLIST", "TSLA, AAPL ,nvda")
    s = Settings()  # type: ignore[call-arg]
    assert s.cc_denylist == ("TSLA", "AAPL", "NVDA")
