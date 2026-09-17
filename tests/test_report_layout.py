"""Report layout fixes: rejected-candidates summarization, the
double-page-break bug, the Ranker-correlation-notes fallback bug, and the
holdings table. These regressed the discover report into a 35-page email
(17 of them a wall of one-paragraph-per-rejected-ticker prose, plus a fully
blank page from a duplicate page_break) — this file pins the fix."""

from __future__ import annotations

from stock_analyzer.discover.report_sections import _primary_reject_reason, build_sections
from stock_analyzer.models.llm import CorrelatedPair, RankerOutput, RankerPick
from stock_analyzer.models.reports import Section


def _pick(ticker: str = "NVDA") -> RankerPick:
    return RankerPick.model_construct(
        rank=1,
        ticker=ticker,
        one_liner="x",
        why_over_alternatives="x",
        conviction=8,
        time_horizon="6-12 months",
        sector_concentration_check="x",
        bull_thesis="x",
        what_youre_betting_on="x",
        scenarios=[],
    )


def _ranker_output(pairs: list[CorrelatedPair] | None = None) -> RankerOutput:
    return RankerOutput.model_construct(
        picks=[_pick()],
        pairs_not_to_hold_together=pairs or [],
        full_text="PICK 1: NVDA — thesis\ntrailing garbage that isn't correlation notes",
    )


def _candidate(ticker: str, passed: bool, fail_reasons: list[str] | None = None) -> dict:
    return {
        "ticker": ticker,
        "passed_filter": passed,
        "fail_reasons": fail_reasons or [],
        "sources": [],
        "conviction": 5,
        "sector": "Technology",
        "price": 100.0,
        "score": 50.0 if passed else None,
        "score_components": {"fundamentals": 20.0, "trend": 20.0, "conviction": 10.0}
        if passed
        else None,
        "score_breakdown": None,
        "themes": [],
    }


_BASE_KWARGS = dict(
    ranker_text="",
    redteam_text="",
    sizer_text="",
    universe_size=10,
    holdings_summary="",
)


# --- _primary_reject_reason --------------------------------------------------


def test_known_prefixes_map_to_stable_labels():
    assert _primary_reject_reason(["no fundamentals data"]) == "No fundamentals data"
    assert _primary_reject_reason(["no technicals data"]) == "No technicals data"
    assert _primary_reject_reason(["market_cap=1000000.0 < $2B"]) == "Market cap too small"
    assert _primary_reject_reason(["revenue_growth=0.05 < 8%"]) == "Revenue growth too slow"
    assert (
        _primary_reject_reason(["operating_cash_flow=-100.0 not positive"])
        == "Negative operating cash flow"
    )
    assert _primary_reject_reason(["debt_to_equity=3.00 > 2.0"]) == "Too much debt"
    assert _primary_reject_reason(["price not above 200DMA"]) == "Below 200-day average"
    assert _primary_reject_reason(["50DMA not above 200DMA"]) == "No moving-average uptrend"
    assert _primary_reject_reason(["rs_6mo=-0.07 not positive"]) == "Weak 6-month relative strength"
    assert _primary_reject_reason(["52w drawdown -0.5 > 30%"]) == "Too far below 52-week high"


def test_uses_first_reason_only():
    # A ticker failing multiple rules is grouped by its first (most
    # upstream) reason, not a compound label.
    assert (
        _primary_reject_reason(["price not above 200DMA", "50DMA not above 200DMA"])
        == "Below 200-day average"
    )


def test_empty_reasons_is_unknown():
    assert _primary_reject_reason([]) == "Unknown"


def test_unrecognized_reason_truncated_fallback():
    label = _primary_reject_reason(["some brand new reason nobody categorized yet"])
    assert label == "some brand new reason nobody categorized yet"[:40]


# --- rejected-candidates section: grouped, not one-paragraph-per-ticker -----


def test_rejected_candidates_grouped_into_pie_and_short_lists():
    candidates = [
        _candidate("NVDA", True),
        *[_candidate(t, False, ["price not above 200DMA"]) for t in ["A", "B", "C"]],
        *[_candidate(t, False, ["revenue_growth=0.05 < 8%"]) for t in ["D", "E"]],
    ]
    sections = build_sections(candidates=candidates, **_BASE_KWARGS)

    heading_idx = next(
        i for i, s in enumerate(sections) if s.kind == "heading" and s.text == "Rejected candidates"
    )
    tail = sections[heading_idx:]

    pie_sections = [s for s in tail if s.kind == "sector_pie"]
    assert len(pie_sections) == 1
    pie_data = dict(pie_sections[0].pie_data)
    assert pie_data == {"Below 200-day average": 3.0, "Revenue growth too slow": 2.0}

    # Grouped into 2 short paragraphs, NOT 5 one-per-ticker paragraphs.
    para_texts = [s.text for s in tail if s.kind == "para"]
    grouped = [t for t in para_texts if t.startswith("Below 200-day average") or t.startswith("Revenue growth")]
    assert len(grouped) == 2
    assert any(t.startswith("Below 200-day average (3): A, B, C") for t in grouped)
    assert any(t.startswith("Revenue growth too slow (2): D, E") for t in grouped)
    assert not any(s.text.startswith("A: ") for s in tail if s.kind == "para")


def test_no_rejected_candidates_section_when_none_rejected():
    candidates = [_candidate("NVDA", True)]
    sections = build_sections(candidates=candidates, **_BASE_KWARGS)
    assert not any(s.kind == "heading" and s.text == "Rejected candidates" for s in sections)


# --- no duplicate page_break between last pick and Allocation summary ------


def test_no_consecutive_page_breaks_around_allocation_summary():
    candidates = [_candidate("NVDA", True)]
    sections = build_sections(
        candidates=candidates,
        **{**_BASE_KWARGS, "ranker_text": "PICK 1: NVDA — x", "ranker_output": _ranker_output()},
    )
    kinds = [s.kind for s in sections]
    for i in range(len(kinds) - 1):
        assert not (kinds[i] == "page_break" and kinds[i + 1] == "page_break"), (
            f"consecutive page_break sections at index {i} — a blank page in the PDF"
        )


# --- Ranker correlation notes: structured output is authoritative ---------


def test_correlation_notes_empty_pairs_shows_clean_none():
    candidates = [_candidate("NVDA", True)]
    sections = build_sections(
        candidates=candidates,
        **{**_BASE_KWARGS, "ranker_text": "PICK 1: NVDA — x", "ranker_output": _ranker_output()},
    )
    idx = next(
        i
        for i, s in enumerate(sections)
        if s.kind == "heading" and s.text == "Ranker correlation notes"
    )
    next_section = sections[idx + 1]
    assert next_section.kind == "para"
    assert next_section.text == "(none)"


def test_correlation_notes_with_pairs_renders_each_pair():
    pair = CorrelatedPair(ticker_a="NVDA", ticker_b="AMD", shared_driver="AI capex")
    candidates = [_candidate("NVDA", True)]
    sections = build_sections(
        candidates=candidates,
        **{
            **_BASE_KWARGS,
            "ranker_text": "PICK 1: NVDA — x",
            "ranker_output": _ranker_output(pairs=[pair]),
        },
    )
    idx = next(
        i
        for i, s in enumerate(sections)
        if s.kind == "heading" and s.text == "Ranker correlation notes"
    )
    next_section = sections[idx + 1]
    assert next_section.kind == "para"
    assert "NVDA + AMD: AI capex" in next_section.text


def test_correlation_notes_no_structured_output_falls_back_to_text_parse():
    candidates = [_candidate("NVDA", True)]
    sections = build_sections(
        candidates=candidates,
        **{
            **_BASE_KWARGS,
            "ranker_text": "PICK 1: NVDA — x\ntrailing prose",
            "ranker_output": None,
        },
    )
    idx = next(
        i
        for i, s in enumerate(sections)
        if s.kind == "heading" and s.text == "Ranker correlation notes"
    )
    next_section = sections[idx + 1]
    assert next_section.kind == "preformatted"


# --- holdings: structured rows render as a table, not a monospace dump ----


def test_holdings_rows_render_as_table():
    candidates = [_candidate("NVDA", True)]
    rows = [["NVDA", "10", "$100.00", "$1,000"]]
    sections = build_sections(
        candidates=candidates,
        holdings_rows=rows,
        **_BASE_KWARGS,
    )
    idx = next(
        i
        for i, s in enumerate(sections)
        if s.kind == "heading" and s.text.startswith("Current holdings")
    )
    next_section = sections[idx + 1]
    assert next_section.kind == "table"
    assert next_section.table_rows == rows


def test_holdings_without_rows_falls_back_to_preformatted():
    candidates = [_candidate("NVDA", True)]
    sections = build_sections(
        candidates=candidates,
        **{**_BASE_KWARGS, "holdings_summary": "- NVDA: 10 shares @ avg $100.00"},
    )
    idx = next(
        i
        for i, s in enumerate(sections)
        if s.kind == "heading" and s.text.startswith("Current holdings")
    )
    next_section = sections[idx + 1]
    assert next_section.kind == "preformatted"


def test_section_model_accepts_table_rows_as_list_of_lists():
    s = Section(kind="table", table_header=["A"], table_rows=[["1"]])
    assert s.table_rows == [["1"]]
