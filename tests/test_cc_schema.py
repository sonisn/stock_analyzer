"""Tests for CC config + RebalancePlan/OptionWrite schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analyzer.config import Settings
from stock_analyzer.discover.rebalance_schema import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def test_cc_defaults():
    s = Settings()  # type: ignore[call-arg]
    assert s.cc_enabled is True
    assert s.cc_target_delta_min == 0.35
    assert s.cc_target_delta_max == 0.45
    assert s.cc_dte_min == 30
    assert s.cc_dte_max == 45
    assert s.cc_denylist == ()
    assert s.cc_min_premium_usd == 500
    assert s.cc_slippage_buffer == 0.10
    assert s.cc_stub_optimization is True
    assert s.cc_min_stub_usd == 1000


def test_cc_denylist_parses_csv(monkeypatch):
    monkeypatch.setenv("CC_DENYLIST", "TSLA, AAPL ,nvda")
    s = Settings()  # type: ignore[call-arg]
    assert s.cc_denylist == ("TSLA", "AAPL", "NVDA")


def test_option_write_valid():
    ow = OptionWrite(
        ticker="NVDA",
        strike=260.0,
        expiry="2026-06-20",
        contracts=3,
        est_premium_per_share=2.40,
        delta=0.36,
        assignment_probability=0.36,
        notes="HOLD-8, far-OTM bias",
    )
    assert ow.ticker == "NVDA"
    assert ow.contracts == 3
    assert ow.notes == "HOLD-8, far-OTM bias"


def test_option_write_frozen():
    ow = OptionWrite(
        ticker="NVDA", strike=260.0, expiry="2026-06-20", contracts=1,
        est_premium_per_share=2.40, delta=0.36, assignment_probability=0.36,
    )
    with pytest.raises(ValidationError):
        ow.strike = 270.0  # type: ignore[misc]


def test_rebalance_action_accepts_write_call():
    a = RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                        sizing="3 contracts $260C 2026-06-20")
    assert a.action == "WRITE_CALL"


def test_rebalance_action_rejects_unknown():
    with pytest.raises(ValidationError):
        RebalanceAction(action="ROLL", ticker="NVDA", sizing="x")  # type: ignore[arg-type]


def test_rebalance_plan_option_writes_default_empty():
    plan = RebalancePlan(status="NO_ACTION", aggressiveness_applied="balanced",
                         full_text="…")
    assert plan.option_writes == []


def test_rebalance_plan_option_writes_roundtrip():
    ow = OptionWrite(
        ticker="NVDA", strike=260.0, expiry="2026-06-20", contracts=2,
        est_premium_per_share=2.40, delta=0.36, assignment_probability=0.36,
    )
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                                 sizing="2 contracts")],
        option_writes=[ow],
        full_text="…",
    )
    blob = plan.model_dump(mode="json")
    restored = RebalancePlan.model_validate(blob)
    assert restored.option_writes[0].ticker == "NVDA"
    assert restored.actions[0].action == "WRITE_CALL"


def test_legacy_plan_without_option_writes_still_parses():
    legacy = {
        "status": "ACTION", "aggressiveness_applied": "balanced",
        "actions": [{"action": "SELL", "ticker": "FOO", "sizing": "full"}],
        "summary": "", "full_text": "…",
    }
    plan = RebalancePlan.model_validate(legacy)
    assert plan.option_writes == []
