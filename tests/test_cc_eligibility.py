"""Tests for CC eligibility / round-lot / earnings / context-block builders."""
from __future__ import annotations

from stock_analyzer.discover.cc_eligibility import (
    EligibleHolding,
    eligible_holdings,
)


def _pos(units: int) -> dict[str, float | int]:
    return {"units": units, "avg_buy_price": 100.0, "cost_basis": units * 100.0}


def test_eligibility_excludes_under_100_shares():
    positions = {
        "AAPL": _pos(99),
        "TSLA": _pos(335),
    }
    out = eligible_holdings(positions, open_short_calls={}, denylist=())
    assert "AAPL" not in out
    assert "TSLA" in out


def test_eligibility_subtracts_open_short_calls():
    positions = {"NVDA": _pos(400)}
    out = eligible_holdings(
        positions, open_short_calls={"NVDA": 1}, denylist=(),
    )
    # 400 - 100 = 300 available, max_contracts = 3
    assert out["NVDA"].available_shares == 300
    assert out["NVDA"].max_contracts == 3


def test_eligibility_excludes_when_coverage_zero():
    positions = {"NVDA": _pos(150)}
    out = eligible_holdings(
        positions, open_short_calls={"NVDA": 1}, denylist=(),
    )
    # 150 - 100 = 50 < 100 → not eligible
    assert "NVDA" not in out


def test_eligibility_respects_denylist():
    positions = {"AAPL": _pos(200), "MSFT": _pos(200)}
    out = eligible_holdings(
        positions, open_short_calls={}, denylist=("AAPL",),
    )
    assert "AAPL" not in out
    assert "MSFT" in out


def test_eligibility_record_shape():
    out = eligible_holdings(
        {"NVDA": _pos(335)}, open_short_calls={}, denylist=(),
    )
    rec = out["NVDA"]
    assert isinstance(rec, EligibleHolding)
    assert rec.ticker == "NVDA"
    assert rec.shares_held == 335
    assert rec.available_shares == 335
    assert rec.max_contracts == 3
    assert rec.open_short_call_contracts == 0
