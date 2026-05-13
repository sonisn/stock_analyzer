"""Integration test: exercise the no-LLM plumbing of the discover pipeline.

Hits screen + persistence + report rendering end-to-end against fixture data.
Skips the actual LLM calls (those cost money). Use for verifying wiring
hasn't broken without paying for Opus.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from stock_analyzer.discover.persistence import (
    connect,
    insert_candidate,
    insert_pick,
    insert_run,
    insert_run_outputs,
    insert_scorecard,
)
from stock_analyzer.discover.rebalancer import REBALANCER_INSTRUCTIONS
from stock_analyzer.discover.report import (
    build_sections,
    parse_picks,
    render_html_email,
    render_pdf,
)
from stock_analyzer.discover.screen import (
    passes_hard_filter,
    score_candidate,
)

# --- fixture data -----------------------------------------------------------

GOOD_FUNDAMENTALS = {
    "market_cap": 50e9,
    "revenue_growth_yoy": 0.15,
    "operating_cash_flow": 5e9,
    "debt_to_equity": 0.5,
    "fcf_yield": 0.04,
    "operating_margin": 0.25,
    "sector": "Technology",
}

GOOD_TECHNICALS = {
    "price": 100.0,
    "sma_50": 95.0,
    "sma_200": 85.0,
    "above_200dma": True,
    "ma_alignment_50_200": True,
    "rs_3mo": 0.05,
    "rs_6mo": 0.08,
    "dist_from_52w_high": -0.10,
    "volume_trend_20_60": 0.10,
    "weekly_rsi": 55.0,
}

UNIVERSE_ENTRY = {"sources": ["insider", "watchlist"], "conviction": 6}

# What the Opus models would emit. Real output is plain text with the
# 'PICK n: TICKER —' and 'TICKER: SYM' markers; the report module parses these.
RANKER_OUTPUT = """\
---
PICK 1: NVDA — Premier AI infrastructure provider with sustained data center demand

Why this over alternatives:
NVDA leads AMD on CUDA ecosystem lock-in and gross margin.

Conviction (1-10): 9
Time horizon: 6-12 months
Sector concentration check: User already holds NVDA; adding would concentrate.

Bull thesis:
Hyperscaler capex through 2026 continues to favor NVDA.

What you're betting on:
Sustained AI capex through next year.
---
---
PICK 2: AMD — Cheaper data center entry, MI300 ramp validated

Why this over alternatives:
AMD offers similar exposure at lower valuation than NVDA.

Conviction (1-10): 7
Time horizon: 6-12 months
Sector concentration check: User has no AMD position.

Bull thesis:
MI300X traction with Meta and Microsoft.

What you're betting on:
Continued share gain in AI accelerators.
---

Pairs not to hold together: NVDA + AMD (both AI accelerators).
"""

REDTEAM_OUTPUT = """\
---
TICKER: NVDA

Bear case (what must go wrong for a 30%+ decline in 12 months):
Hyperscaler capex slows in 2026 as AI ROI questions surface.

Most fragile assumption in the bull thesis:
That capex continues at current pace.

Watch this number: Q3 data center revenue growth — below 30% YoY = thesis broken.

Fragility rank: 2
---
---
TICKER: AMD

Bear case (what must go wrong for a 30%+ decline in 12 months):
MI300 ramp stalls and CUDA ecosystem keeps customers locked to NVDA.

Most fragile assumption in the bull thesis:
That MI300 wins more hyperscaler share.

Watch this number: MI300 revenue mix.

Fragility rank: 1
---

Single most fragile pick: AMD — depends on out-executing NVDA, which is hard.
"""

SIZER_OUTPUT = """\
---
TICKER: NVDA
Allocation: $6,000
Rationale: High conviction (9), low fragility (2), but you already hold it.
---
---
TICKER: AMD
Allocation: $4,000
Rationale: Lower conviction, higher fragility — smaller position.
---

Concentration warnings: AI accelerators would be ~40% combined — at limit.
"""


# --- tests ------------------------------------------------------------------


def test_screen_filter_and_score_on_fixture():
    passes, reasons = passes_hard_filter(GOOD_FUNDAMENTALS, GOOD_TECHNICALS)
    assert passes is True, reasons
    scored = score_candidate(GOOD_FUNDAMENTALS, GOOD_TECHNICALS, UNIVERSE_ENTRY)
    assert 0 < scored["score"] <= 100
    assert set(scored["components"].keys()) == {"fundamentals", "trend", "conviction"}


def test_parse_picks_finds_both_picks():
    picks = parse_picks(RANKER_OUTPUT)
    assert len(picks) == 2
    assert picks[0] == (1, "NVDA", "Premier AI infrastructure provider with sustained data center demand")
    assert picks[1][1] == "AMD"


def test_persistence_round_trip(tmp_path: Path):
    db_path = str(tmp_path / "test.db")
    with connect(db_path) as conn:
        run_id = insert_run(
            conn,
            universe_size=10,
            survivors=2,
            picks=2,
            opus_model="claude-opus-4-7",
            sonnet_model="claude-sonnet-4-6",
            cash_budget=10000.0,
        )
        assert run_id > 0

        insert_candidate(
            conn, run_id, "NVDA",
            passed_filter=True,
            fail_reasons=[],
            score=85.0,
            score_components={"fundamentals": 35, "trend": 30, "conviction": 20},
            score_breakdown={"fundamentals": {"revenue_growth": 15}},
            sources=["insider"],
            conviction=4,
            sector="Technology",
            price=500.0,
        )
        insert_candidate(
            conn, run_id, "XYZ",
            passed_filter=False,
            fail_reasons=["market_cap=1e9 < $5B"],
            score=None,
            score_components=None,
            score_breakdown=None,
            sources=[],
            conviction=0,
            sector=None,
            price=None,
        )
        insert_scorecard(conn, run_id, "NVDA", "TICKER: NVDA\nScore: 9\n...")
        insert_pick(
            conn, run_id,
            rank=1, ticker="NVDA",
            ranker_text=RANKER_OUTPUT,
            bear_case_text=REDTEAM_OUTPUT,
            allocation_text=SIZER_OUTPUT,
        )
        insert_run_outputs(
            conn, run_id,
            ranker_full=RANKER_OUTPUT,
            redteam_full=REDTEAM_OUTPUT,
            sizer_full=SIZER_OUTPUT,
            holdings_summary="  - NVDA: 10 shares @ avg $400.00",
        )

    # Verify rows landed.
    conn2 = sqlite3.connect(db_path)
    try:
        n_runs = conn2.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_candidates = conn2.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        n_picks = conn2.execute("SELECT COUNT(*) FROM picks").fetchone()[0]
        n_outputs = conn2.execute("SELECT COUNT(*) FROM run_outputs").fetchone()[0]
        assert (n_runs, n_candidates, n_picks, n_outputs) == (1, 2, 1, 1)
    finally:
        conn2.close()


def _fixture_candidates():
    return [
        {
            "ticker": "NVDA",
            "passed_filter": True,
            "fail_reasons": [],
            "score": 85.0,
            "score_components": {"fundamentals": 35.0, "trend": 30.0, "conviction": 20.0},
            "sector": "Technology",
        },
        {
            "ticker": "AMD",
            "passed_filter": True,
            "fail_reasons": [],
            "score": 78.0,
            "score_components": {"fundamentals": 30.0, "trend": 28.0, "conviction": 20.0},
            "sector": "Technology",
        },
        {
            "ticker": "XYZ",
            "passed_filter": False,
            "fail_reasons": ["market_cap=1e9 < $5B"],
            "score": None,
            "score_components": None,
            "sector": None,
        },
    ]


def test_html_email_renders_with_chart_refs():
    sections = build_sections(
        ranker_text=RANKER_OUTPUT,
        redteam_text=REDTEAM_OUTPUT,
        sizer_text=SIZER_OUTPUT,
        candidates=_fixture_candidates(),
        universe_size=3,
        holdings_summary="  - NVDA: 10 shares @ avg $400.00",
        macro_summary="Curve flat at 0.20%; VIX moderate at 18.5",
        sector_rotation={"leaders": ["Technology"], "laggards": ["Utilities"]},
    )
    chart_cids = {"NVDA": "chart-NVDA", "AMD": "chart-AMD"}
    html_body = render_html_email(sections, chart_cids)
    # Picks have inline bull + bear + allocation
    assert "<h2>NVDA</h2>" in html_body
    assert "<h2>AMD</h2>" in html_body
    assert "Bull case" in html_body
    assert "Bear case" in html_body
    assert "Position sizing" in html_body
    # Chart CIDs are referenced for picks
    assert "cid:chart-NVDA" in html_body
    assert "cid:chart-AMD" in html_body
    # Macro + sector rotation rendered
    assert "Macro regime" in html_body
    assert "Sector rotation" in html_body
    # Rejected appendix present
    assert "Rejected candidates" in html_body
    assert "XYZ" in html_body


def _tiny_png_bytes() -> bytes:
    """Generate a valid tiny PNG via Pillow (already installed for reportlab)."""
    from io import BytesIO

    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (4, 4), color=(200, 200, 220)).save(buf, format="PNG")
    return buf.getvalue()


def test_pdf_renders_bytes_with_charts():
    sections = build_sections(
        ranker_text=RANKER_OUTPUT,
        redteam_text=REDTEAM_OUTPUT,
        sizer_text=SIZER_OUTPUT,
        candidates=_fixture_candidates(),
        universe_size=3,
        holdings_summary="  - NVDA: 10 shares @ avg $400.00",
    )
    chart = _tiny_png_bytes()
    pdf_bytes = render_pdf(sections, chart_bytes={"NVDA": chart, "AMD": chart})
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000  # Sanity: non-trivial document


def test_html_renders_visual_sections():
    """Status banner, metric strip, holdings dashboard, sector pie all render."""
    from stock_analyzer.discover.report import Section, render_html_email

    sections = [
        Section(kind="status_banner", text="STATUS: NO ACTION RECOMMENDED",
                status="NO_ACTION"),
        Section(kind="metric_strip", metrics=[
            ("Holdings", "13"), ("Cash", "$53"), ("P/L", "+18.4%"),
        ]),
        Section(kind="holdings_dashboard", holdings=[
            {"ticker": "NVDA", "verdict": "HOLD", "confidence": 8,
             "pnl_pct": 85.4, "sector": "Technology", "note": ""},
            {"ticker": "MRVL", "verdict": "TRIM", "confidence": 5,
             "pnl_pct": 120.2, "sector": "Technology",
             "note": "RSI overbought"},
        ]),
        Section(kind="sector_pie", pie_data=[
            ("Technology", 50_000), ("Energy", 12_000), ("Industrials", 8_000),
        ]),
    ]
    out = render_html_email(sections, chart_cids={})
    assert "STATUS: NO ACTION RECOMMENDED" in out
    assert "Holdings" in out and "Cash" in out
    # Verdict badges rendered with their colors
    assert "HOLD" in out and "TRIM" in out
    # Pie SVG renders
    assert "<svg" in out and "Technology" in out and "Energy" in out
    # P/L coloring class applied
    assert "pl-pos" in out or "pl-neg" in out


def test_pdf_renders_without_charts():
    """Chart-fetch failures must not crash PDF generation."""
    sections = build_sections(
        ranker_text=RANKER_OUTPUT,
        redteam_text=REDTEAM_OUTPUT,
        sizer_text=SIZER_OUTPUT,
        candidates=_fixture_candidates(),
        universe_size=3,
        holdings_summary="",
    )
    pdf_bytes = render_pdf(sections, chart_bytes={})
    assert pdf_bytes.startswith(b"%PDF-")


def test_orchestrator_imports():
    """If the orchestrator can be imported, all module wiring is consistent."""
    from stock_analyzer.cli.discover import run
    assert callable(run)


def test_rebalance_pipeline_imports_and_assembles():
    """RebalancePipeline must construct and produce a 10-step workflow."""
    from stock_analyzer.cli.rebalance import RebalancePipeline
    from stock_analyzer.config import Settings

    pipeline = RebalancePipeline(Settings.from_env())
    wf = pipeline.build_workflow()
    assert wf.name == "Portfolio Rebalance"
    step_names = [
        getattr(step, "name", type(step).__name__) for step in wf.steps
    ]
    # Rebalance adds 3 steps vs discover's 10; persist step is renamed.
    assert "review_holdings" in step_names
    assert "rebalance" in step_names
    assert "persist_and_email_rebalance" in step_names


def test_rebalance_section_layout():
    """Rebalance sections must put plan + reviews before the discover appendix."""
    from stock_analyzer.cli.rebalance import _build_rebalance_sections

    sections = _build_rebalance_sections(
        rebalance_text="REBALANCE PLAN\n\nStatus: NO ACTION RECOMMENDED\n\nSummary: ...",
        holdings_reviews={
            "ABC": "TICKER: ABC\nVerdict: HOLD\nConfidence (1-10): 8\nReasoning: solid",
            "XYZ": "TICKER: XYZ\nVerdict: SELL\nConfidence (1-10): 3\nReasoning: broken",
        },
        ranker_text=RANKER_OUTPUT,
        redteam_text=REDTEAM_OUTPUT,
        sizer_text=SIZER_OUTPUT,
        candidates=_fixture_candidates(),
        cash_balance=5000.0,
        macro_summary="Curve flat",
        sector_rotation=None,
        holdings_positions={
            "ABC": {"units": 100, "avg_buy_price": 50, "cost_basis": 5000},
            "XYZ": {"units": 50, "avg_buy_price": 80, "cost_basis": 4000},
        },
        holdings_technicals={
            "ABC": {"price": 65},
            "XYZ": {"price": 60},
        },
        holdings_fundamentals={
            "ABC": {"sector": "Technology"},
            "XYZ": {"sector": "Energy"},
        },
    )
    # First heading must be the rebalance title
    assert sections[0].text.startswith("Portfolio Rebalance")
    # Plan must appear before reviews must appear before discover appendix
    plan_idx = next(i for i, s in enumerate(sections) if "Rebalance plan" in s.text)
    reviews_idx = next(i for i, s in enumerate(sections) if "Per-holding reviews" in s.text)
    discover_idx = next(
        i for i, s in enumerate(sections) if "Discover picks" in s.text
    )
    assert plan_idx < reviews_idx < discover_idx
    # All holding tickers appear as headings
    headings = [s.text for s in sections if s.kind == "heading"]
    assert "ABC" in headings
    assert "XYZ" in headings


def test_rebalancer_prompt_includes_cc_rules():
    s = REBALANCER_INSTRUCTIONS
    assert "COVERED-CALL WRITING" in s
    assert "WRITE_CALL" in s
    assert "0.35" in s and "0.45" in s
    assert "STUB CONSOLIDATION" in s
    assert "PREMIUM REINVESTMENT" in s
    assert "option_writes" in s


def test_decide_includes_cc_context_in_prompt():
    from unittest.mock import MagicMock

    from stock_analyzer.discover.rebalance_schema import RebalancePlan
    from stock_analyzer.discover.rebalancer import Rebalancer

    captured: dict[str, str] = {}

    class _StubAgent:
        def run(self, prompt):
            captured["prompt"] = prompt
            return MagicMock(
                content=RebalancePlan(
                    status="NO_ACTION",
                    aggressiveness_applied="balanced",
                    full_text="…",
                )
            )

    r = Rebalancer.__new__(Rebalancer)
    r.agent = _StubAgent()
    cc_block = "===\nCOVERED-CALL CONTEXT\n===\nTICKER: NVDA\n  Shares held: 400"
    r.decide(
        holdings_reviews={},
        picks_text="",
        cash_available=1000.0,
        cc_context_block=cc_block,
    )
    assert "COVERED-CALL CONTEXT" in captured["prompt"]
    assert "TICKER: NVDA" in captured["prompt"]


def test_decide_omits_cc_block_when_empty():
    from unittest.mock import MagicMock

    from stock_analyzer.discover.rebalance_schema import RebalancePlan
    from stock_analyzer.discover.rebalancer import Rebalancer

    captured: dict[str, str] = {}

    class _StubAgent:
        def run(self, prompt):
            captured["prompt"] = prompt
            return MagicMock(
                content=RebalancePlan(
                    status="NO_ACTION",
                    aggressiveness_applied="balanced",
                    full_text="…",
                )
            )

    r = Rebalancer.__new__(Rebalancer)
    r.agent = _StubAgent()
    r.decide(
        holdings_reviews={},
        picks_text="",
        cash_available=1000.0,
        cc_context_block="",
    )
    assert "COVERED-CALL CONTEXT" not in captured["prompt"]
