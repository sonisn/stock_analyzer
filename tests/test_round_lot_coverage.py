"""Tests for round-lot coverage math (stub consolidation context)."""
from __future__ import annotations

from stock_analyzer.discover.cc_eligibility import (
    RoundLotCoverage,
    round_lot_coverage,
)


def test_basic_split_and_stub():
    positions = {
        "TSLA": {"units": 335, "avg_buy_price": 250},
        "AAPL": {"units": 215, "avg_buy_price": 150},
        "NVDA": {"units": 100, "avg_buy_price": 235},  # exactly a round lot, no stub
        "GOOG": {"units": 50,  "avg_buy_price": 170},  # all stub
    }
    spots = {"TSLA": 300.0, "AAPL": 215.0, "NVDA": 235.0, "GOOG": 175.0}
    out = round_lot_coverage(positions, spots=spots)

    tsla = out["TSLA"]
    assert tsla.round_lots == 3
    assert tsla.stub_shares == 35
    assert tsla.stub_dollar_value == 35 * 300.0
    assert tsla.to_next_lot_shares == 65
    assert tsla.to_next_lot_cost == 65 * 300.0

    aapl = out["AAPL"]
    assert aapl.round_lots == 2
    assert aapl.stub_shares == 15

    nvda = out["NVDA"]
    assert nvda.round_lots == 1
    assert nvda.stub_shares == 0
    assert nvda.to_next_lot_shares == 0
    assert nvda.to_next_lot_cost == 0.0

    goog = out["GOOG"]
    assert goog.round_lots == 0
    assert goog.stub_shares == 50
    assert goog.to_next_lot_shares == 50
    assert goog.to_next_lot_cost == 50 * 175.0


def test_missing_spot_falls_back_to_zero_dollar_values():
    positions = {"FOO": {"units": 150}}
    out = round_lot_coverage(positions, spots={})
    rec = out["FOO"]
    assert rec.round_lots == 1
    assert rec.stub_shares == 50
    assert rec.stub_dollar_value == 0.0


def test_record_is_RoundLotCoverage_type():
    out = round_lot_coverage({"FOO": {"units": 100}}, spots={"FOO": 10.0})
    assert isinstance(out["FOO"], RoundLotCoverage)
