"""Report parsers — verdict / confidence / action / status / picks.

These regex-based parsers operate on LLM-produced text. A prompt tweak
can silently break them — e.g., the LLM starts emitting 'Verdict (1-10):'
instead of 'Verdict:'. Both paths (structured object and free text)
must keep working: the structured form is canonical; the regex form is
a legacy / partial-run fallback that must not crash on edge cases.
"""
from __future__ import annotations

from stock_analyzer.discover.report import (
    parse_actions,
    parse_confidence,
    parse_picks,
    parse_rebalance_status,
    parse_verdict,
)
from stock_analyzer.models.llm import HoldingReview
from stock_analyzer.models.rebalance import RebalanceAction, RebalancePlan


def _hr(verdict: str, confidence: int) -> HoldingReview:
    return HoldingReview(
        ticker="AAPL", verdict=verdict, confidence=confidence,
        trim_pct=None, position_context="x", forward_outlook="y",
        reasoning="z", tax_lot_plan=[], what_would_change_mind="w",
        wash_sale_notice=None, full_text="...",
    )


# --- verdict + confidence: structured path is field read, regex is fallback


def test_parse_verdict_reads_field_on_structured_input():
    """Structured HoldingReview: no regex, just field access."""
    assert parse_verdict(_hr("SELL", 8)) == "SELL"
    assert parse_verdict(_hr("HOLD", 5)) == "HOLD"


def test_parse_verdict_regex_fallback_on_free_text():
    """Legacy DB rows (review_text only) — regex-scan for 'Verdict: X'."""
    text = "TICKER: NVDA\nVerdict: TRIM\nConfidence (1-10): 8"
    assert parse_verdict(text) == "TRIM"


def test_parse_verdict_defaults_to_hold_on_missing_or_garbage():
    """The safer default: when the parser can't find a verdict, return
    HOLD. The rebalancer's loss-override layer can still upgrade it."""
    assert parse_verdict("") == "HOLD"
    assert parse_verdict(None) == "HOLD"
    assert parse_verdict("No verdict here") == "HOLD"


def test_parse_confidence_regex_picks_up_integer():
    """Free-text path: 'Confidence (1-10): N' → int N."""
    text = "Verdict: SELL\nConfidence (1-10): 9\nForward outlook: ..."
    assert parse_confidence(text) == 9
    assert parse_confidence("No confidence line") is None
    assert parse_confidence(None) is None


# --- rebalance status: structured path + text fallback ------------------


def test_parse_rebalance_status_structured_returns_field():
    """RebalancePlan.status is canonical — no regex needed."""
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="balanced",
        actions=[RebalanceAction(action="SELL", ticker="TSLA", sizing="all")],
        summary="x", full_text="...",
    )
    assert parse_rebalance_status(plan) == "ACTION"


def test_parse_rebalance_status_free_text_recognizes_no_action():
    """Free-text fallback — must distinguish ACTION from NO_ACTION."""
    assert parse_rebalance_status(
        "Status: NO ACTION RECOMMENDED\nReasoning: ..."
    ) == "NO_ACTION"
    assert parse_rebalance_status(
        "Status: ACTION RECOMMENDED\nAction 1: SELL TSLA"
    ) == "ACTION"
    # No status line and not a known type → UNKNOWN, not a crash.
    assert parse_rebalance_status("garbage") == "UNKNOWN"
    assert parse_rebalance_status(None) == "UNKNOWN"


def test_parse_actions_reads_structured_plan_in_order():
    """RebalancePlan.actions: preserve order, return (action, ticker) tuples."""
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="balanced",
        actions=[
            RebalanceAction(action="SELL", ticker="TSLA", sizing="all"),
            RebalanceAction(action="ADD", ticker="GOOGL", sizing="$3400"),
        ],
        summary="x", full_text="...",
    )
    assert parse_actions(plan) == [("SELL", "TSLA"), ("ADD", "GOOGL")]


def test_parse_picks_reads_structured_ranker_sorted_by_rank():
    """RankerOutput must be unwrapped by parse_picks with output sorted
    by rank ascending (rank 1 is the top pick)."""
    from stock_analyzer.models.llm import RankerOutput
    # Build a minimal RankerOutput. Use model_construct() to skip the
    # heavy field requirements of RankerPick — parse_picks only reads
    # rank/ticker/one_liner.
    output = RankerOutput.model_construct(
        picks=[
            type("P", (), {"rank": 2, "ticker": "INTC", "one_liner": "B"})(),
            type("P", (), {"rank": 1, "ticker": "NVDA", "one_liner": "A"})(),
        ],
        pairs_not_to_hold_together=[],
        full_text="",
    )
    out = parse_picks(output)
    assert [t for _, t, _ in out] == ["NVDA", "INTC"]  # sorted by rank


def test_parse_picks_regex_fallback_on_free_text():
    """Legacy path: parse 'PICK N: TICKER — one-liner' lines."""
    text = (
        "PICK 1: NVDA — AI capex tailwind\n"
        "Some prose...\n"
        "PICK 2: INTC — foundry turnaround\n"
    )
    picks = parse_picks(text)
    assert [t for _, t, _ in picks] == ["NVDA", "INTC"]
    assert picks[0][2] == "AI capex tailwind"


def test_premium_income_renders_table_html():
    from stock_analyzer.discover.report_html import render_html_email
    from stock_analyzer.models.reports import Section

    sections = [
        Section(kind="heading", text="Test", level=1),
        Section(kind="premium_income", data={  # type: ignore[arg-type]
            "rows": [{
                "ticker": "NVDA", "strike": 260.0, "expiry": "2026-06-20",
                "contracts": 3, "premium_usd": 720.0,
                "delta": 0.36, "assignment_pct": 36,
            }],
            "gross_premium_usd": 720.0,
            "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
        }),
    ]
    html = render_html_email(sections, chart_cids={})
    assert "NVDA" in html
    assert "$260" in html
    assert "Gross premium" in html
    assert "$720" in html


def test_round_lot_coverage_html_emits_table():
    from stock_analyzer.discover.report_html import render_html_email
    from stock_analyzer.models.reports import Section

    sections = [
        Section(kind="round_lot_coverage", data={  # type: ignore[arg-type]
            "rows": [{
                "ticker": "TSLA", "shares": 335,
                "round_lots": 3, "round_lot_shares": 300,
                "stub_shares": 35, "stub_dollar_value": 10500.0,
                "to_next_lot_shares": 65, "to_next_lot_cost": 19500.0,
            }],
            "stub_pool_total_usd": 10500.0,
        }),
    ]
    html = render_html_email(sections, chart_cids={})
    assert "TSLA" in html
    assert "Round-Lot Coverage" in html
    assert "$10,500" in html


def test_premium_deployment_html_emits_box():
    from stock_analyzer.discover.report_html import render_html_email
    from stock_analyzer.models.reports import Section

    sections = [
        Section(kind="premium_deployment", data={  # type: ignore[arg-type]
            "gross_premium_usd": 720.0, "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
            "existing_cash_usd": 850.0,
            "stub_consolidation_usd": 10500.0,
            "total_dry_powder_usd": 11998.0,
            "deployments": [
                {"ticker": "AMZN", "action": "ADD", "sizing": "$1,400"},
            ],
        }),
    ]
    html = render_html_email(sections, chart_cids={})
    assert "Premium" in html and "Deployment" in html
    assert "AMZN" in html
    assert "ADD" in html
    assert "Total dry powder" in html
    assert "$11,998" in html


def test_pdf_renders_with_cc_sections_smoke():
    from stock_analyzer.discover.report_pdf import render_pdf
    from stock_analyzer.models.reports import Section

    sections = [
        Section(kind="heading", text="Test", level=1),
        Section(kind="premium_income", data={  # type: ignore[arg-type]
            "rows": [{"ticker": "NVDA", "strike": 260.0, "expiry": "2026-06-20",
                      "contracts": 3, "premium_usd": 720.0,
                      "delta": 0.36, "assignment_pct": 36}],
            "gross_premium_usd": 720.0,
            "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
        }),
        Section(kind="round_lot_coverage", data={  # type: ignore[arg-type]
            "rows": [{"ticker": "TSLA", "shares": 335, "round_lots": 3,
                      "round_lot_shares": 300, "stub_shares": 35,
                      "stub_dollar_value": 10500.0,
                      "to_next_lot_shares": 65, "to_next_lot_cost": 19500.0}],
            "stub_pool_total_usd": 10500.0,
        }),
        Section(kind="premium_deployment", data={  # type: ignore[arg-type]
            "gross_premium_usd": 720.0, "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
            "existing_cash_usd": 850.0,
            "stub_consolidation_usd": 10500.0,
            "total_dry_powder_usd": 11998.0,
            "deployments": [
                {"ticker": "AMZN", "action": "ADD", "sizing": "$1,400"},
            ],
        }),
    ]
    pdf = render_pdf(sections, chart_bytes={})
    assert isinstance(pdf, bytes)
    assert len(pdf) > 1000  # not just a header
