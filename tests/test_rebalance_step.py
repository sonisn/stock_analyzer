"""The rebalance step around the model call: a lost plan is reported as
lost, and a validator that crashes degrades the safe way."""

from __future__ import annotations

from contextlib import nullcontext
from unittest.mock import patch

import pytest

from stock_analyzer.cli.rebalance import RebalancePipeline
from stock_analyzer.cli.rebalance_steps import plan_steps
from stock_analyzer.config import Settings
from stock_analyzer.discover.rebalancer import RebalancePlanUnparseable
from stock_analyzer.models.rebalance import RebalanceAction, RebalancePlan

PLAN = RebalancePlan(
    status="ACTION",
    aggressiveness_applied="balanced",
    summary="sell",
    actions=[
        RebalanceAction(action="SELL", ticker="NVDA", sizing="50 shares"),
        RebalanceAction(action="SELL_PUT", ticker="AMD", sizing="1 contract"),
    ],
    full_text="plan prose",
)


def _run(outcome, **patched):
    class FakeRebalancer:
        def __init__(self, *a, **k):
            pass

        def decide(self, *a, **k):
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    p = RebalancePipeline(Settings())
    p.state.update(
        {
            "ranker_text": "#1 AMD — pick",
            "holdings_positions": {"NVDA": {"units": 100.0}},
            "harvest_candidates_obj": [],
        }
    )
    with (
        patch.object(plan_steps, "Rebalancer", FakeRebalancer),
        patch.object(plan_steps, "_build_history_block", return_value=""),
        patch.multiple(plan_steps, **patched) if patched else nullcontext(),
    ):
        out = p.step_rebalance(None)
    return p, out


def _boom(*a, **k):
    raise RuntimeError("validator crashed")


@pytest.mark.parametrize(
    ("error", "note"),
    [
        (
            RebalancePlanUnparseable("cut", raw_text='{"status":', truncated=True),
            "cut off before it finished",
        ),
        (RebalancePlanUnparseable("bad", raw_text="{"), "could not be read"),
        (ValueError("max_tokens too high"), "did not return a plan (ValueError)"),
    ],
)
def test_a_lost_plan_is_recorded_as_lost_not_as_no_action(error, note):
    p, out = _run(error)
    assert out.content.startswith("rebalance: PLAN LOST")
    assert p.state["rebalance_plan"] is None
    assert note in p.state["rebalance_failed"]


def test_the_raw_text_of_a_cut_off_plan_is_kept():
    p, _ = _run(RebalancePlanUnparseable("cut", raw_text='{"status":', truncated=True))
    assert p.state["rebalance_text"] == '{"status":'


def test_a_crashed_put_validator_drops_every_put():
    p, out = _run(PLAN, apply_csp_plan_validation=_boom)
    plan = p.state["rebalance_plan"]
    assert [a.action for a in plan.actions] == ["SELL"]
    assert "all puts dropped" in p.state["csp_warnings"][0]
    assert out.content.startswith("Rebalance plan generated")


def test_a_crashed_call_validator_keeps_the_plan_and_says_so():
    p, _ = _run(PLAN, apply_cc_plan_validation=_boom)
    assert p.state["cc_warnings"] == ["validation crashed: validator crashed"]
    assert p.state["rebalance_plan"] is not None
