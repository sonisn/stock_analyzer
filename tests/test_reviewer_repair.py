"""Verdict auto-repair — the silent-failure layer.

The reviewer agent occasionally emits a structured verdict that
contradicts the prose it wrote in the same response (SELL with HOLD
reasoning, TRIM with confidence 5, etc.). `_repair_verdict_inconsistencies`
detects and rewrites these so the rebalancer doesn't act on the LLM's
wrong-enum mistake. If this layer breaks silently, every rebalance is
subtly wrong — these tests pin the three repair rules.
"""
from __future__ import annotations

from stock_analyzer.discover.reviewer import _repair_verdict_inconsistencies
from stock_analyzer.models.llm import HoldingReview


def _make_review(**overrides) -> HoldingReview:
    """Build a HoldingReview with sensible defaults; override per test."""
    base = dict(
        ticker="AAPL",
        verdict="HOLD",
        confidence=7,
        trim_pct=None,
        position_context="100 shares @ $150",
        forward_outlook="Steady earnings ahead.",
        reasoning="Forward EPS revisions positive.",
        tax_lot_plan=[],
        what_would_change_mind="iPhone sales miss.",
        wash_sale_notice=None,
        full_text="...",
    )
    base.update(overrides)
    return HoldingReview(**base)


# --- Rule 1: confidence calibration --------------------------------------


def test_rule1_sell_with_low_confidence_repairs_to_hold():
    """SELL/TRIM require conf >= 7. SELL at conf=5 is the LLM picking
    the wrong enum; rewrite to HOLD."""
    review = _make_review(verdict="SELL", confidence=5)
    repaired = _repair_verdict_inconsistencies(review, "AAPL")
    assert repaired.verdict == "HOLD"


def test_rule1_hold_with_low_confidence_is_unchanged():
    """HOLD with low confidence is fine — uncertainty about whether to
    HOLD doesn't trigger the rewrite."""
    review = _make_review(verdict="HOLD", confidence=3)
    repaired = _repair_verdict_inconsistencies(review, "AAPL")
    assert repaired.verdict == "HOLD"
    assert repaired.confidence == 3


# --- Rule 2: prose contradiction -----------------------------------------


def test_rule2_prose_says_hold_overrides_sell_verdict():
    """Reasoning paragraph defends a HOLD; structured field accidentally
    flipped to SELL. Prose is authoritative — flip back to HOLD."""
    review = _make_review(
        verdict="SELL",
        confidence=8,  # high enough to bypass rule 1
        reasoning="The HOLD verdict preserves our long-term thesis here.",
    )
    repaired = _repair_verdict_inconsistencies(review, "ANET")
    assert repaired.verdict == "HOLD"


def test_rule2_prose_says_sell_upgrades_verdict_and_bumps_conf():
    """Reasoning explicitly says 'the SELL verdict ...' but structured
    field is HOLD with conf=5. Upgrade to SELL and bump conf to 7 to
    stay calibrated."""
    review = _make_review(
        verdict="HOLD",
        confidence=5,
        reasoning="The SELL verdict is justified by the worsening guidance.",
    )
    repaired = _repair_verdict_inconsistencies(review, "TSLA")
    assert repaired.verdict == "SELL"
    assert repaired.confidence == 7  # bumped from 5 to floor


# --- Rule 3: 'toward X' contradicts current verdict ----------------------


def test_rule3_toward_trim_while_verdict_is_trim_with_low_conf_repairs_to_hold():
    """'toward TRIM' implies the current verdict is something OTHER than
    TRIM. Combined with sub-calibrated confidence, repair to HOLD."""
    review = _make_review(
        verdict="TRIM",
        confidence=5,  # below 7 threshold
        what_would_change_mind="More cautious guidance would push toward TRIM.",
    )
    repaired = _repair_verdict_inconsistencies(review, "MRVL")
    assert repaired.verdict == "HOLD"


# --- baseline: no repair when everything is consistent -------------------


def test_no_repair_when_inputs_are_internally_consistent():
    """Verdict + confidence + prose all agree — must pass through
    unchanged (no model_copy call)."""
    review = _make_review(
        verdict="SELL",
        confidence=8,
        reasoning="Forward fundamentals deteriorating; price target below current.",
        what_would_change_mind="A surprise guidance raise would change my mind.",
    )
    repaired = _repair_verdict_inconsistencies(review, "INTC")
    assert repaired.verdict == "SELL"
    assert repaired.confidence == 8
    assert repaired is review  # no copy needed when no updates
