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


# Run #40, 2026-10-07: the TSLA trim was dropped, and the NVDA add it was
# funding still asked for ~$40,200 with $21,404 in the IRA.
OCT7_POSITIONS = {
    "TSLA": {"units": 200.37, "value": 200.37 * 450.0},
    "OKLO": {"units": 3.0, "value": 3 * 36.32},
}
OCT7_CASH = {"Traditional IRA": 21_404.0, "HSA": 31.0, "Robinhood": 1.0}
OCT7_PLAN = (
    ("TRIM", "TSLA", "25% — 50 shares (~$22,500) in Traditional IRA"),
    ("TRIM", "OKLO", "33% — 1 share (~$36) in Traditional IRA"),
    (
        "ADD",
        "NVDA",
        "~$40,200 less the TSLA call buy-to-close debit (~165-170 shares) in Traditional IRA",
    ),
)


def _oct7(*actions, cash=OCT7_CASH):
    plan = _plan(*actions).model_copy(update={"summary": "Trim TSLA, redeploy into NVDA."})
    return validate_sales(
        plan, positions=OCT7_POSITIONS, obligations=OBLIGATIONS, account_cash=cash
    )


def test_a_buy_funded_by_a_dropped_sale_shrinks_to_the_cash_there():
    plan, warnings = _oct7(*OCT7_PLAN)
    assert [a.ticker for a in plan.actions] == ["OKLO", "NVDA"]
    nvda = plan.actions[1].sizing
    # $21,404 cash + ~$36 from the OKLO share, rounded down to $100.
    assert nvda.startswith("~$21,400 in Traditional IRA")
    assert "TSLA trim it counted on was dropped" in nvda
    assert any("NVDA: ADD cut from ~$40,200 to ~$21,400" in w for w in warnings)
    # Neither the summary nor the plan text still reads as the old trade.
    assert plan.summary.startswith("Changed after planning:")
    assert plan.summary.endswith("Original plan: Trim TSLA, redeploy into NVDA.")
    assert plan.full_text.startswith("ADJUSTED AFTER PLANNING\n- TSLA: dropped TRIM")


def test_a_buy_the_cash_already_covers_is_left_alone():
    plan, warnings = _oct7(OCT7_PLAN[0], ("ADD", "NVDA", "~$15,000 in Traditional IRA"))
    assert plan.actions[0].sizing == "~$15,000 in Traditional IRA"
    assert len(warnings) == 1  # just the dropped trim


def test_a_buy_with_no_cash_left_is_dropped():
    plan, warnings = _oct7(OCT7_PLAN[0], ("BUY", "LLY", "~$5,000 in HSA"), cash={"HSA": 31.0})
    assert plan.actions == []
    assert any("LLY: dropped BUY of ~$5,000" in w for w in warnings)


def test_buys_split_the_cash_in_proportion_and_unnamed_ones_use_the_total():
    plan, _ = _oct7(
        OCT7_PLAN[0],
        ("BUY", "LLY", "~$30,000"),
        ("ADD", "NVDA", "~$10,000"),
        cash={"Traditional IRA": 20_000.0},
    )
    assert [a.sizing.split(" in ")[0] for a in plan.actions] == ["~$15,000", "~$5,000"]


def test_buys_are_untouched_when_no_sale_was_dropped_or_cash_is_unknown():
    over = ("ADD", "NVDA", "~$90,000 in Traditional IRA")
    plan, warnings = _oct7(over)
    assert plan.actions[0].sizing == over[2] and warnings == []
    plan, warnings = _oct7(OCT7_PLAN[0], over, cash=None)
    assert plan.actions[0].sizing == over[2] and len(warnings) == 1
