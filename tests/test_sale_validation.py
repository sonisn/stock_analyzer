"""You cannot sell a share you have already promised to someone else.

Run #35 proposed trimming 50 TSLA shares while 200 of 200.37 backed two
$400 calls — 0.37 free. The obligations were fetched in that same run,
for the tax-loss harvester, and never reached the rebalancer.
"""

from __future__ import annotations

from stock_analyzer.discover.sale_validation import (
    _requested_shares,
    covered_call_block,
    free_shares,
    validate_sales,
)
from stock_analyzer.models.rebalance import RebalanceAction, RebalancePlan

# The live book on 2026-09-20.
POSITIONS = {
    "TSLA": {"units": 200.37},
    "AVGO": {"units": 161.55},
    "GOOGL": {"units": 200.25},
    "LLY": {"units": 0.0},
}
OBLIGATIONS = {
    "TSLA": {
        "contracts": 2,
        "shares_committed": 200.0,
        "lowest_strike": 400.0,
        "next_expiry": "2026-12-18",
    },
    "AVGO": {"contracts": 1, "shares_committed": 100.0, "lowest_strike": 480.0},
    "GOOGL": {"contracts": 2, "shares_committed": 200.0, "lowest_strike": 450.0},
}


def _plan(*actions: tuple[str, str, str]) -> RebalancePlan:
    return RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[RebalanceAction(action=a, ticker=t, sizing=s) for a, t, s in actions],
        full_text="…",
    )


def test_free_shares_subtracts_what_is_promised():
    assert free_shares("TSLA", POSITIONS, OBLIGATIONS) == 200.37 - 200.0
    assert free_shares("AVGO", POSITIONS, OBLIGATIONS) == 161.55 - 100.0
    # No obligation at all: everything is free.
    assert free_shares("LLY", POSITIONS, OBLIGATIONS) == 0.0
    # A ticker that isn't held can't be reasoned about.
    assert free_shares("NVDA", POSITIONS, OBLIGATIONS) is None


def test_the_tsla_trim_that_shipped_is_refused():
    plan, warnings = validate_sales(
        _plan(("TRIM", "TSLA", "25% — 50 shares (~$18,214) in Traditional IRA")),
        positions=POSITIONS,
        obligations=OBLIGATIONS,
    )
    assert plan.actions == []
    assert len(warnings) == 1
    w = warnings[0]
    assert "only 0.37 are free" in w
    assert "2 short call(s)" in w and "$400.00" in w and "2026-12-18" in w
    assert "Buy the call back first" in w


def test_a_sale_inside_the_free_shares_survives():
    """AVGO's 61-share stub sits under its 61.55 free, so it stands."""
    plan, warnings = validate_sales(
        _plan(("TRIM", "AVGO", "61 shares — stub consolidation (~$21,814)")),
        positions=POSITIONS,
        obligations=OBLIGATIONS,
    )
    assert [a.ticker for a in plan.actions] == ["AVGO"]
    assert warnings == []


def test_buys_and_unencumbered_tickers_are_untouched():
    plan, warnings = validate_sales(
        _plan(
            ("BUY", "LLY", "~$26,000"),
            ("ADD", "GOOGL", "~$21,000 (60 shares)"),
            ("WRITE_CALL", "TSLA", "1 contract"),
        ),
        positions=POSITIONS,
        obligations=OBLIGATIONS,
    )
    assert len(plan.actions) == 3
    assert warnings == []


def test_an_unreadable_sizing_is_flagged_not_deleted():
    """Never silently drop a sale that merely failed to parse."""
    plan, warnings = validate_sales(
        _plan(("TRIM", "TSLA", "a meaningful amount")),
        positions=POSITIONS,
        obligations=OBLIGATIONS,
    )
    assert [a.ticker for a in plan.actions] == ["TSLA"]
    assert "could not be read as a share count" in warnings[0]
    assert "0.37 of 200.37" in warnings[0]


def test_sizing_forms_that_must_be_understood():
    assert _requested_shares("25% — 50 shares", 200.37) == 50.0
    assert _requested_shares("1,200 units", 5000.0) == 1200.0
    assert _requested_shares("25%", 200.0) == 50.0
    assert _requested_shares("full position", 200.37) == 200.37
    assert _requested_shares("~$18,214", 200.37) is None


def test_no_obligations_means_no_interference():
    plan = _plan(("TRIM", "TSLA", "50 shares"))
    out, warnings = validate_sales(plan, positions=POSITIONS, obligations={})
    assert out is plan
    assert warnings == []


def test_the_prompt_says_what_is_free():
    block = covered_call_block(POSITIONS, OBLIGATIONS)
    assert "TSLA: 200.37 held, 200 promised to 2 short call(s) at $400.00" in block
    assert "0.37 share(s) are free to sell" in block
    assert "turns that call naked" in block
    assert covered_call_block(POSITIONS, {}) == ""
