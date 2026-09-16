"""Mechanical stop-loss override: deterministic backstop for the Reviewer's
own soft DOWNTREND OVERRIDE prompt rule. A HOLD verdict on a position down
20%+ from cost basis is force-escalated to TRIM 25%, regardless of what the
LLM's reasoning argued."""

from __future__ import annotations

from stock_analyzer.discover.rebalance_holdings import apply_stop_loss_overrides
from stock_analyzer.models.llm import HoldingReview


def _review(ticker: str = "AAPL", **overrides) -> HoldingReview:
    base = dict(
        ticker=ticker,
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


def _positions(avg_buy_price: float) -> dict:
    return {"AAPL": {"avg_buy_price": avg_buy_price, "units": 100, "cost_basis": avg_buy_price * 100}}


def _technicals(price: float) -> dict:
    return {"AAPL": {"price": price}}


def test_hold_at_hard_stop_forced_to_trim():
    reviews = {"AAPL": _review(verdict="HOLD")}
    # avg 200, current 150 -> -25%
    updated, warnings = apply_stop_loss_overrides(reviews, _positions(200.0), _technicals(150.0))
    assert updated["AAPL"].verdict == "TRIM"
    assert updated["AAPL"].trim_pct == 25.0
    assert "MECHANICAL STOP-LOSS" in updated["AAPL"].reasoning
    assert len(warnings) == 1
    assert "AAPL" in warnings[0]


def test_hold_above_threshold_untouched():
    reviews = {"AAPL": _review(verdict="HOLD")}
    # avg 200, current 180 -> -10%, above -20% hard stop
    updated, warnings = apply_stop_loss_overrides(reviews, _positions(200.0), _technicals(180.0))
    assert updated["AAPL"] is reviews["AAPL"]
    assert warnings == []


def test_existing_trim_verdict_untouched():
    reviews = {"AAPL": _review(verdict="TRIM", trim_pct=50.0, confidence=8)}
    updated, warnings = apply_stop_loss_overrides(reviews, _positions(200.0), _technicals(120.0))
    assert updated["AAPL"] is reviews["AAPL"]
    assert warnings == []


def test_existing_sell_verdict_untouched():
    reviews = {"AAPL": _review(verdict="SELL", confidence=9)}
    updated, warnings = apply_stop_loss_overrides(reviews, _positions(200.0), _technicals(100.0))
    assert updated["AAPL"] is reviews["AAPL"]
    assert warnings == []


def test_missing_price_or_cost_basis_does_not_crash():
    reviews = {"AAPL": _review(verdict="HOLD")}
    updated, warnings = apply_stop_loss_overrides(reviews, {"AAPL": {"avg_buy_price": None}}, {})
    assert updated["AAPL"] is reviews["AAPL"]
    assert warnings == []

    updated2, warnings2 = apply_stop_loss_overrides(reviews, {}, {})
    assert updated2["AAPL"] is reviews["AAPL"]
    assert warnings2 == []


def test_exactly_at_threshold_triggers():
    reviews = {"AAPL": _review(verdict="HOLD")}
    # avg 100, current 80 -> exactly -20%
    updated, warnings = apply_stop_loss_overrides(reviews, _positions(100.0), _technicals(80.0))
    assert updated["AAPL"].verdict == "TRIM"
    assert len(warnings) == 1


def test_custom_hard_stop_pct():
    reviews = {"AAPL": _review(verdict="HOLD")}
    # avg 100, current 90 -> -10%
    updated, warnings = apply_stop_loss_overrides(
        reviews, _positions(100.0), _technicals(90.0), hard_stop_pct=-10.0
    )
    assert updated["AAPL"].verdict == "TRIM"
    assert len(warnings) == 1
