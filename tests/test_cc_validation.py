"""Tests for post-LLM validation of WRITE_CALL actions."""
from __future__ import annotations

from stock_analyzer.discover.cc_validation import validate_option_writes
from stock_analyzer.models.portfolio import EligibleHolding
from stock_analyzer.models.rebalance import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def _ow(ticker: str, contracts: int = 1) -> OptionWrite:
    return OptionWrite(
        ticker=ticker, strike=100.0, expiry="2026-06-20",
        contracts=contracts, est_premium_per_share=1.0,
        delta=0.4, assignment_probability=0.4,
    )


def _wc(ticker: str) -> RebalanceAction:
    return RebalanceAction(action="WRITE_CALL", ticker=ticker, sizing="x")


def _elig(ticker: str, max_contracts: int) -> EligibleHolding:
    return EligibleHolding(
        ticker=ticker, shares_held=max_contracts * 100,
        open_short_call_contracts=0,
        available_shares=max_contracts * 100,
        max_contracts=max_contracts,
    )


def test_orphan_write_call_gets_dropped():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("ORPHAN"), RebalanceAction(action="ADD", ticker="X", sizing="$100")],
        option_writes=[],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(
        plan, eligibility={"ORPHAN": _elig("ORPHAN", 3)},
    )
    types = [a.action for a in cleaned.actions]
    assert "WRITE_CALL" not in types
    assert "ADD" in types
    assert any("orphan" in w.lower() for w in warnings)


def test_oversized_contracts_get_clamped():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA")],
        option_writes=[_ow("NVDA", contracts=5)],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(
        plan, eligibility={"NVDA": _elig("NVDA", 3)},
    )
    ow = cleaned.option_writes[0]
    assert ow.contracts == 3
    assert any("clamp" in w.lower() for w in warnings)


def test_well_formed_plan_passes_through():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA")],
        option_writes=[_ow("NVDA", contracts=2)],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(
        plan, eligibility={"NVDA": _elig("NVDA", 3)},
    )
    assert warnings == []
    assert cleaned.option_writes[0].contracts == 2


def test_unknown_ticker_in_write_call_drops_it():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("MYSTERY")],
        option_writes=[_ow("MYSTERY")],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(plan, eligibility={})
    assert all(a.action != "WRITE_CALL" for a in cleaned.actions)
    assert cleaned.option_writes == []
    assert any("not eligible" in w.lower() for w in warnings)
