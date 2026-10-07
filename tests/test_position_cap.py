"""REBALANCE_MAX_POSITION_PCT was only in the prompt; cap_positions
enforces it on the plan the model returns."""

from __future__ import annotations

from stock_analyzer.discover.position_cap import cap_positions
from stock_analyzer.discover.sale_validation import note_adjustments
from stock_analyzer.models.rebalance import RebalanceAction, RebalancePlan

# A $500k book: $100k cash, NVDA at $150k (30%).
POSITIONS = {
    "NVDA": {"units": 600.0, "value": 150_000.0},
    "AVGO": {"units": 500.0, "value": 250_000.0},
}
CASH = 100_000.0


def _plan(*actions: tuple[str, str, str]) -> RebalancePlan:
    return RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[RebalanceAction(action=a, ticker=t, sizing=s) for a, t, s in actions],
        summary="Add NVDA.",
        full_text="plan",
    )


def _cap(plan, **kw):
    return cap_positions(
        plan,
        positions=POSITIONS,
        cash=CASH,
        max_pct=kw.pop("max_pct", 35.0),
        accounts=["Traditional IRA"],
        **kw,
    )


def test_a_buy_past_the_cap_is_cut_to_the_room_left():
    plan, warnings = _cap(_plan(("ADD", "NVDA", "~$40,000 in Traditional IRA")))
    # Cap 35% of $500k = $175k; NVDA holds $150k, so $25k of room.
    assert plan.actions[0].sizing.startswith("~$25,000 in Traditional IRA (cut from:")
    assert "NVDA: ADD cut from ~$40,000 to ~$25,000" in warnings[0]
    assert plan.summary.startswith("Changed after planning:")


def test_a_buy_inside_the_cap_and_a_new_name_by_share_count_pass():
    plan, warnings = _cap(
        _plan(("ADD", "NVDA", "~$20,000"), ("BUY", "LLY", "20 shares")),
        prices={"LLY": 800.0},
    )
    assert warnings == [] and plan.summary == "Add NVDA."


def test_a_trim_earlier_in_the_plan_makes_room():
    plan, warnings = _cap(_plan(("TRIM", "NVDA", "100 shares"), ("ADD", "NVDA", "~$40,000")))
    # 100 shares at $250 = $25k out, so $50k of room.
    assert warnings == []


def test_a_stock_already_over_the_cap_gets_no_buy():
    plan, warnings = _cap(_plan(("ADD", "AVGO", "~$5,000")))
    assert plan.actions == []
    assert "AVGO: dropped ADD of ~$5,000" in warnings[0] and "50%" in warnings[0]


def test_an_unreadable_buy_is_named_not_changed_and_no_cap_means_no_check():
    plan, warnings = _cap(_plan(("ADD", "NVDA", "a starter position")))
    assert plan.actions[0].sizing == "a starter position"
    assert "could not be sized" in warnings[0] and plan.summary == "Add NVDA."
    plan, warnings = _cap(_plan(("ADD", "NVDA", "~$900,000")), max_pct=100)
    assert warnings == []


def test_notes_from_two_checks_merge_instead_of_nesting():
    plan = note_adjustments(_plan(), ["TSLA: dropped TRIM."])
    plan = note_adjustments(plan, ["NVDA: ADD cut."])
    assert (
        plan.summary
        == "Changed after planning: NVDA: ADD cut. TSLA: dropped TRIM. Original plan: Add NVDA."
    )
    assert (
        plan.full_text == "ADJUSTED AFTER PLANNING\n- NVDA: ADD cut.\n- TSLA: dropped TRIM.\n\nplan"
    )
