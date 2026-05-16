"""Tests for post-LLM validation of WRITE_CALL actions."""
from __future__ import annotations

from stock_analyzer.discover.cc_validation import validate_option_writes
from stock_analyzer.models.portfolio import EligibleHolding
from stock_analyzer.models.rebalance import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def _ow(ticker: str, contracts: int = 1, account: str = "Test Account") -> OptionWrite:
    return OptionWrite(
        ticker=ticker, account=account,
        strike=100.0, expiry="2026-06-20",
        contracts=contracts, est_premium_per_share=1.0,
        delta=0.4, assignment_probability=0.4,
    )


def _wc(ticker: str) -> RebalanceAction:
    return RebalanceAction(action="WRITE_CALL", ticker=ticker, sizing="x")


def _elig(ticker: str, max_contracts: int, account: str = "Test Account") -> EligibleHolding:
    return EligibleHolding(
        ticker=ticker, account=account,
        tax_status="taxable",
        shares_held=max_contracts * 100,
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
        plan, eligibility={"ORPHAN": [_elig("ORPHAN", 3)]},
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
        plan, eligibility={"NVDA": [_elig("NVDA", 3)]},
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
        plan, eligibility={"NVDA": [_elig("NVDA", 3)]},
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
    assert any("account" in w.lower() or "not eligible" in w.lower() for w in warnings)


def test_validation_clamps_per_account_max_contracts():
    """OptionWrite for 5 contracts in IRA, but IRA only allows 2 → clamp to 2."""
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA")],
        option_writes=[
            OptionWrite(
                ticker="NVDA", account="Fidelity IRA",
                strike=260.0, expiry="2026-06-20",
                contracts=5, est_premium_per_share=2.30,
                delta=0.36, assignment_probability=0.36,
            ),
        ],
        full_text="…",
    )
    eligibility = {"NVDA": [_elig("NVDA", 2, account="Fidelity IRA")]}
    cleaned, warnings = validate_option_writes(plan, eligibility=eligibility)
    assert cleaned.option_writes[0].contracts == 2
    assert any("clamp" in w.lower() for w in warnings)


def test_validation_allows_one_optionwrite_per_account():
    """NVDA IRA 2 + NVDA Taxable 1 both keep."""
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA"), _wc("NVDA")],
        option_writes=[
            OptionWrite(
                ticker="NVDA", account="Fidelity IRA",
                strike=260.0, expiry="2026-06-20",
                contracts=2, est_premium_per_share=2.30,
                delta=0.36, assignment_probability=0.36,
            ),
            OptionWrite(
                ticker="NVDA", account="Fidelity Taxable",
                strike=260.0, expiry="2026-06-20",
                contracts=1, est_premium_per_share=2.30,
                delta=0.36, assignment_probability=0.36,
            ),
        ],
        full_text="…",
    )
    eligibility = {"NVDA": [
        _elig("NVDA", 2, account="Fidelity IRA"),
        _elig("NVDA", 1, account="Fidelity Taxable"),
    ]}
    cleaned, warnings = validate_option_writes(plan, eligibility=eligibility)
    assert len(cleaned.option_writes) == 2
    accounts = sorted(ow.account for ow in cleaned.option_writes)
    assert accounts == ["Fidelity IRA", "Fidelity Taxable"]


def test_validation_drops_optionwrite_with_unknown_account():
    """Ticker is eligible somewhere, but not in this account → drop."""
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA")],
        option_writes=[
            OptionWrite(
                ticker="NVDA", account="Mystery Account",
                strike=260.0, expiry="2026-06-20",
                contracts=1, est_premium_per_share=2.30,
                delta=0.36, assignment_probability=0.36,
            ),
        ],
        full_text="…",
    )
    eligibility = {"NVDA": [_elig("NVDA", 2, account="Fidelity IRA")]}
    cleaned, warnings = validate_option_writes(plan, eligibility=eligibility)
    assert cleaned.option_writes == []
    assert any("account" in w.lower() for w in warnings)
