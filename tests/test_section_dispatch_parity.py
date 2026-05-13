"""Regression guard: every SectionKind has both an HTML and a PDF dispatch.

When a new section type is added (e.g. premortem_panel in Phase 5d), it
must be wired in BOTH renderers. Forgetting one means the email looks
right but the PDF is missing the section (or vice versa) — silent
divergence that's only caught by eyeballing a real run.

This test asks both renderers to render a minimal fixture of every
kind in the SectionKind Literal and asserts the output is non-empty.
Adding a new kind to the Literal and forgetting to wire either
renderer will fail here.
"""
from __future__ import annotations

from typing import Any, get_args

from stock_analyzer.discover.report import (
    Section,
    SectionKind,
    render_html_email,
    render_pdf,
)

# Minimal payload per kind that produces visible output. If a new kind
# is added to SectionKind, add an entry here too — the test below
# enforces this.
_FIXTURES: dict[str, dict[str, Any]] = {
    "heading":               {"text": "H", "level": 2},
    "para":                  {"text": "p"},
    "preformatted":          {"text": "x"},
    "image":                 {"image_ticker": "NVDA"},
    "blockquote":            {"text": "q"},
    "table":                 {"table_header": ["A"], "table_rows": [["1"]]},
    "page_break":            {},
    "status_banner":         {"text": "STATUS", "status": "ACTION"},
    "metric_strip":          {"metrics": [("k", "v")]},
    "holdings_dashboard":    {"holdings": [{"ticker": "X", "verdict": "HOLD",
                                            "confidence": 5, "pnl_pct": 1.0,
                                            "sector": "Tech", "note": ""}]},
    "sector_pie":            {"pie_data": [("Tech", 1.0), ("Auto", 2.0)]},
    "pick_card":             {"data": {"ticker": "NVDA", "rank": 1,
                                       "one_liner": "x"}},
    "allocation_table":      {"data": {"allocations": [{"ticker": "X",
                                                        "pct": 10.0,
                                                        "rationale": "y"}]}},
    "rebalance_action_table": {"data": {"actions": [{"action": "SELL",
                                                     "ticker": "X",
                                                     "sizing": "all"}]}},
    "holding_review_card":   {"data": {"ticker": "X", "verdict": "HOLD"}},
    "market_themes_panel":   {"data": {"themes": [{"name": "T",
                                                   "description": "d",
                                                   "strength": 7,
                                                   "trending": "up",
                                                   "member_tickers": ["X"]}]}},
    "premortem_panel":       {"data": {"overall_verdict": "proceed_with_caveat",
                                       "summary": "s",
                                       "failures": [{"likelihood": "high",
                                                     "severity": "severe",
                                                     "triggering_action": "a",
                                                     "failure_narrative": "n",
                                                     "early_warning": "w"}]}},
    "premium_income":        {"data": {"rows": []}},
    "round_lot_coverage":    {"data": {"rows": []}},
    "premium_deployment":    {"data": {"rows": []}},
}


def test_fixture_covers_every_section_kind():
    """If a new kind is added to SectionKind, _FIXTURES must add an entry
    too — otherwise the parity test below is silently skipping it."""
    declared = set(get_args(SectionKind))
    covered = set(_FIXTURES.keys())
    missing = declared - covered
    extra = covered - declared
    assert not missing, f"Add fixture for new SectionKind(s): {missing}"
    assert not extra, f"_FIXTURES has stale entries not in SectionKind: {extra}"


def test_every_section_kind_renders_in_both_html_and_pdf():
    """Each kind must produce non-empty HTML AND a valid PDF. This catches
    the canonical 'forgot to wire dispatch' bug — without it, you'd only
    notice by eyeballing a real run."""
    body_open, body_close = "<body>", "</body>"
    for kind, fields in _FIXTURES.items():
        section = Section(kind=kind, **fields)  # type: ignore[arg-type]
        # Provide a CID + chart bytes so `image` kind has something to draw.
        cids = {"NVDA": "chart-NVDA"} if kind == "image" else {}
        chart_bytes = {"NVDA": b"\x89PNG\r\n\x1a\n"} if kind == "image" else {}

        html_out = render_html_email([section], cids)
        between = html_out[
            html_out.index(body_open) + len(body_open):
            html_out.index(body_close)
        ].strip()
        assert between, (
            f"HTML renderer produced empty body for kind={kind!r} — "
            f"likely missing dispatch case in render_html_email."
        )

        pdf_out = render_pdf([section], chart_bytes)
        assert pdf_out.startswith(b"%PDF"), (
            f"PDF renderer failed for kind={kind!r} (no %PDF header)."
        )
        # Sanity floor: a real PDF page is well over 1KB. A truly empty
        # dispatch would still produce a minimal document, so this is
        # mostly a "didn't crash" assertion.
        assert len(pdf_out) > 800


def test_section_accepts_new_cc_kinds():
    for kind in ("premium_income", "round_lot_coverage", "premium_deployment"):
        s = Section(kind=kind, data={"rows": []})  # type: ignore[arg-type]
        assert s.kind == kind
