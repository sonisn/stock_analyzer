"""Cross-source price-target reconciliation (yfinance vs Finnhub)."""

from __future__ import annotations

from stock_analyzer.discover.data_reconciliation import reconcile_price_targets


def test_agreeing_targets_produce_no_warning():
    fundamentals = {"analyst_target_mean": 100.0}
    finnhub = {"mean": 105.0}  # 5% apart, within default 20% tolerance
    assert reconcile_price_targets(fundamentals, finnhub) is None


def test_disagreeing_targets_produce_a_warning():
    fundamentals = {"analyst_target_mean": 100.0}
    finnhub = {"mean": 150.0}  # 50% apart
    warning = reconcile_price_targets(fundamentals, finnhub)
    assert warning is not None
    assert "100" in warning and "150" in warning


def test_missing_either_source_produces_no_warning():
    assert reconcile_price_targets({"analyst_target_mean": 100.0}, {}) is None
    assert reconcile_price_targets({}, {"mean": 100.0}) is None
    assert reconcile_price_targets(None, None) is None


def test_custom_tolerance_is_respected():
    fundamentals = {"analyst_target_mean": 100.0}
    finnhub = {"mean": 110.0}  # 10% apart
    assert reconcile_price_targets(fundamentals, finnhub, tolerance_pct=20.0) is None
    assert reconcile_price_targets(fundamentals, finnhub, tolerance_pct=5.0) is not None
