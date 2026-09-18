"""Drawdown review: holdings are long-term, so a position down 20%+ from
cost is flagged for a thesis re-check — its verdict is never changed."""

from __future__ import annotations

from stock_analyzer.discover.rebalance_holdings import flag_drawdown_reviews
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
    return {
        "AAPL": {"avg_buy_price": avg_buy_price, "units": 100, "cost_basis": avg_buy_price * 100}
    }


def _technicals(price: float) -> dict:
    return {"AAPL": {"price": price}}


def test_deep_loser_is_flagged_but_verdict_kept():
    reviews = {"AAPL": _review(verdict="HOLD")}
    # avg 200, current 150 -> -25%
    notes = flag_drawdown_reviews(reviews, _positions(200.0), _technicals(150.0))
    assert reviews["AAPL"].verdict == "HOLD"
    assert notes == [
        "AAPL: down -25% from cost — thesis re-checked, reviewer says HOLD (confidence 7/10)"
    ]


def test_sell_verdict_is_reported_too():
    reviews = {"AAPL": _review(verdict="SELL", confidence=8)}
    notes = flag_drawdown_reviews(reviews, _positions(200.0), _technicals(150.0))
    assert "reviewer says SELL (confidence 8/10)" in notes[0]


def test_above_threshold_and_missing_data_not_flagged():
    reviews = {"AAPL": _review()}
    assert flag_drawdown_reviews(reviews, _positions(200.0), _technicals(180.0)) == []
    assert flag_drawdown_reviews(reviews, {}, _technicals(150.0)) == []
    assert flag_drawdown_reviews(reviews, _positions(200.0), {}) == []


def test_exactly_at_threshold_and_custom_threshold():
    reviews = {"AAPL": _review()}
    assert len(flag_drawdown_reviews(reviews, _positions(200.0), _technicals(160.0))) == 1
    assert (
        flag_drawdown_reviews(reviews, _positions(200.0), _technicals(150.0), review_pct=-30.0)
        == []
    )
