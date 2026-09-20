"""Covered calls written to be kept.

The bands were 0.35-0.45 delta over 30-45 days — roughly a 40% chance of
losing the shares on a 3-5 year holding, every six weeks. They are now
0.10-0.25 over 60-120 days, with a hard floor under the strike and a
volatility gate, and the rules are enforced rather than prompted: the
call path checked eligibility only, so a 0.60-delta write a month out
would have passed validation.
"""

from __future__ import annotations

from datetime import date, timedelta

from stock_analyzer.discover.cc_validation import validate_option_writes
from stock_analyzer.discover.rebalance_cc import drop_cheap_premium
from stock_analyzer.models.portfolio import EligibleHolding, IvHvRegime
from stock_analyzer.models.rebalance import OptionWrite, RebalanceAction, RebalancePlan

TODAY = date(2026, 9, 20)
ELIGIBILITY = {
    "NVDA": [
        EligibleHolding(
            ticker="NVDA",
            account="Traditional IRA",
            tax_status="tax_advantaged",
            shares_held=400,
            open_short_call_contracts=0,
            available_shares=400,
            max_contracts=4,
        )
    ]
}


def _plan(strike: float, delta: float, days: int) -> RebalancePlan:
    return RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="NVDA", sizing="1 contract")],
        option_writes=[
            OptionWrite(
                ticker="NVDA",
                account="Traditional IRA",
                strike=strike,
                expiry=(TODAY + timedelta(days=days)).isoformat(),
                contracts=1,
                est_premium_per_share=4.0,
                delta=delta,
                assignment_probability=delta,
            )
        ],
        full_text="…",
    )


def _validate(plan):
    return validate_option_writes(
        plan,
        eligibility=ELIGIBILITY,
        spots={"NVDA": 222.27},
        delta_max=0.25,
        min_upside_pct=15.0,
        dte_min=60,
        dte_max=120,
        today=TODAY,
    )


def test_a_far_out_call_on_a_long_expiry_is_kept():
    plan, warnings = _validate(_plan(strike=285.0, delta=0.18, days=89))
    assert len(plan.option_writes) == 1 and warnings == []


def test_a_high_delta_write_is_dropped_however_rich():
    plan, warnings = _validate(_plan(strike=285.0, delta=0.42, days=89))
    assert plan.option_writes == []
    assert "delta 0.42 above the 0.25 ceiling" in warnings[0]
    assert "chance of losing the shares" in warnings[0]
    # the orphan action goes with it
    assert [a for a in plan.actions if a.action == "WRITE_CALL"] == []


def test_a_strike_too_close_to_spot_is_dropped():
    # $240 is only 8% above $222.27 — no room for the position to run.
    plan, warnings = _validate(_plan(strike=240.0, delta=0.20, days=89))
    assert plan.option_writes == []
    assert "inside the 15% floor" in warnings[0]


def test_a_near_dated_write_is_dropped():
    plan, warnings = _validate(_plan(strike=285.0, delta=0.18, days=30))
    assert plan.option_writes == []
    assert "30d to expiry, outside the 60-120d band" in warnings[0]


def test_an_over_long_write_is_dropped_too():
    # Longer is not unboundedly better: a LEAP caps the position for a year.
    plan, warnings = _validate(_plan(strike=285.0, delta=0.18, days=400))
    assert plan.option_writes == []
    assert "400d to expiry" in warnings[0]


def test_without_bands_nothing_new_is_enforced():
    # Older callers pass no bands; behaviour must be unchanged for them.
    plan, warnings = validate_option_writes(
        _plan(strike=240.0, delta=0.42, days=30), eligibility=ELIGIBILITY
    )
    assert len(plan.option_writes) == 1 and warnings == []


# --- write when the market is paying ---------------------------------------------


def _regime(ratio: float) -> IvHvRegime:
    return IvHvRegime(
        ticker="NVDA",
        current_iv=0.40,
        hv_annualized=0.40 / ratio,
        iv_hv_ratio=ratio,
        label="elevated" if ratio >= 1.2 else "average" if ratio >= 0.9 else "depressed",
    )


def test_cheap_premium_is_held_back_with_a_reason():
    kept, cheap = drop_cheap_premium(ELIGIBILITY, {"NVDA": _regime(0.80)}, min_ratio=1.0)
    assert kept == {}
    assert "below the 1.00x floor" in cheap["NVDA"]
    assert "Wait for a volatile session" in cheap["NVDA"]


def test_rich_premium_is_written():
    kept, cheap = drop_cheap_premium(ELIGIBILITY, {"NVDA": _regime(1.35)}, min_ratio=1.0)
    assert set(kept) == {"NVDA"} and cheap == {}


def test_an_unknown_volatility_reading_does_not_stop_writing():
    # A failed vol fetch is not evidence that premium is cheap.
    kept, cheap = drop_cheap_premium(ELIGIBILITY, {}, min_ratio=1.0)
    assert set(kept) == {"NVDA"} and cheap == {}


def test_the_gate_can_be_switched_off():
    kept, cheap = drop_cheap_premium(ELIGIBILITY, {"NVDA": _regime(0.5)}, min_ratio=0.0)
    assert set(kept) == {"NVDA"} and cheap == {}


def test_the_chain_row_shows_what_the_premium_pays_per_day():
    # "Further out pays more" is true per contract and false per day.
    from stock_analyzer.discover.cc_eligibility import _format_chain_row
    from stock_analyzer.models.market import OptionQuote

    def row(days, bid, ask):
        return _format_chain_row(
            OptionQuote(
                expiry=TODAY + timedelta(days=days),
                strike=285.0,
                bid=bid,
                ask=ask,
                delta=0.18,
                iv=0.42,
                open_interest=1200,
                volume=300,
            ),
            today=TODAY,
        )

    near = row(30, 3.90, 4.10)  # $400 over 30 days
    far = row(120, 9.90, 10.10)  # $1,000 over 120 days
    assert "(30d)" in near and "$13.33/day" in near
    assert "(120d)" in far and "$8.33/day" in far
