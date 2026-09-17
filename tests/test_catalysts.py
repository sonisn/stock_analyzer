"""Forward catalysts: Tavily news normalization, deterministic validation of
LLM-extracted catalysts, and their use in the Ranker prompt and reports.
No network or LLM calls — Tavily is a fake client, the Ranker's model call
is stubbed."""

from __future__ import annotations

import json
from datetime import date

import pytest

from stock_analyzer.data import ticker_news
from stock_analyzer.data.ticker_news import TavilyQuotaExceeded, fetch_ticker_news
from stock_analyzer.discover.catalysts import (
    catalysts_to_dicts,
    format_catalyst_block,
    repair_catalysts,
    validate_catalysts,
)
from stock_analyzer.discover.ranker import Ranker
from stock_analyzer.discover.rebalance_holdings import build_holding_review_payloads
from stock_analyzer.discover.report import Section, render_html_email, render_pdf
from stock_analyzer.models.llm import AnalystReport, Catalyst, HoldingReview, RankerOutput

TODAY = date(2026, 9, 17)


def _cat(source: str = "news:N1", expected_date: str | None = "2026-10-20", **kw) -> Catalyst:
    base = dict(
        event="Q3 earnings",
        expected_date=expected_date,
        direction="uncertain",
        impact="high",
        source=source,
    )
    base.update(kw)
    return Catalyst(**base)


def _report(ticker: str = "NVDA", catalysts: list[Catalyst] | None = None) -> AnalystReport:
    return AnalystReport(
        ticker=ticker,
        score=7,
        one_liner="x",
        competitive_position="x",
        growth_runway="x",
        top_risks=["x"],
        valuation_context="x",
        catalyst_calendar="x",
        upcoming_catalysts=catalysts or [],
        full_text="TICKER: NVDA\nScore: 7",
    )


# --- Tavily normalization ----------------------------------------------------


class _FakeTavily:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"results": r}


def _result(url, title="t", published="2026-09-10", content="c" * 1000):
    return {"url": url, "title": title, "published_date": published, "content": content}


def test_news_sorted_newest_first_with_stable_ids_and_trimmed_fields():
    client = _FakeTavily(
        [
            [
                _result("https://www.reuters.com/a", published="2026-09-01"),
                _result("https://www.reuters.com/a", published="2026-09-01"),  # dup
                _result("https://cnbc.com/b", published="2026-09-15"),
                _result("https://wsj.com/c", published=None),
            ]
        ]
    )
    items = fetch_ticker_news("NVDA", "NVIDIA", client=client)
    assert [i["source"] for i in items] == ["cnbc.com", "reuters.com", "wsj.com"]
    assert [i["id"] for i in items] == ["N1", "N2", "N3"]
    assert all(len(i["snippet"]) == 400 for i in items)
    assert "url" not in items[0]
    assert "NVIDIA (NVDA)" in client.calls[0]["query"]
    assert client.calls[0]["include_domains"]


def test_falls_back_to_unfiltered_search_when_premium_domains_empty():
    client = _FakeTavily([[], [_result("https://smallcapnews.com/x")]])
    items = fetch_ticker_news("TINY", client=client)
    assert len(client.calls) == 2
    assert "include_domains" not in client.calls[1]
    assert items[0]["source"] == "smallcapnews.com"


def test_search_error_returns_empty_without_spending_a_retry():
    client = _FakeTavily([RuntimeError("connection reset")])
    assert fetch_ticker_news("NVDA", client=client) == []
    assert len(client.calls) == 1


_QUOTA_MSG = "This request exceeds your plan's set usage limit."


def test_quota_error_raises_instead_of_returning_empty():
    client = _FakeTavily([RuntimeError(_QUOTA_MSG)])
    with pytest.raises(TavilyQuotaExceeded):
        fetch_ticker_news("NVDA", client=client)


def test_batch_stops_calling_tavily_after_quota_error(monkeypatch):
    client = _FakeTavily([RuntimeError(_QUOTA_MSG)] * 10)
    monkeypatch.setenv("TAVILY_API_KEY", "test")
    monkeypatch.setattr(ticker_news, "TavilyClient", lambda api_key: client)
    monkeypatch.setattr(ticker_news, "_TAVILY_MAX_WORKERS", 1)
    out = ticker_news.batch_ticker_news(["A", "B", "C", "D"])
    assert out == {"A": [], "B": [], "C": [], "D": []}
    assert len(client.calls) == 1


# --- validation ---------------------------------------------------------------


def test_unknown_news_id_is_dropped():
    kept, warnings = validate_catalysts([_cat("news:N9")], news_ids={"N1"}, today=TODAY)
    assert kept == []
    assert warnings


def test_invented_source_is_dropped():
    kept, _ = validate_catalysts([_cat("my memory")], news_ids={"N1"}, today=TODAY)
    assert kept == []


def test_fixed_sources_are_accepted_without_news():
    cats = [_cat(s) for s in ("quarterly_mda", "earnings_transcript", "earnings_calendar")]
    kept, warnings = validate_catalysts(cats, news_ids=set(), today=TODAY)
    assert len(kept) == 3
    assert warnings == []


def test_past_dated_catalyst_is_dropped():
    kept, warnings = validate_catalysts(
        [_cat(expected_date="2026-08-01")], news_ids={"N1"}, today=TODAY
    )
    assert kept == []
    assert "past" in warnings[0]


def test_month_only_date_normalized_and_garbage_date_nulled():
    kept, _ = validate_catalysts(
        [_cat(expected_date="2026-11"), _cat(expected_date="sometime in Q4")],
        news_ids={"N1"},
        today=TODAY,
    )
    assert kept[0].expected_date == "2026-11-01"
    assert kept[1].expected_date is None


def test_repair_leaves_clean_reports_untouched_and_fixes_bad_ones():
    clean = _report("AAA", [_cat("news:N1")])
    dirty = _report("BBB", [_cat("news:N1"), _cat("news:N7")])
    empty = _report("CCC")
    news = {"AAA": [{"id": "N1"}], "BBB": [{"id": "N1"}]}
    out, warnings = repair_catalysts({"AAA": clean, "BBB": dirty, "CCC": empty}, news, today=TODAY)
    assert out["AAA"] is clean
    assert out["CCC"] is empty
    assert len(out["BBB"].upcoming_catalysts) == 1
    assert len(warnings) == 1


def test_repair_works_on_holding_reviews_too():
    review = HoldingReview(
        ticker="AAPL",
        verdict="HOLD",
        confidence=7,
        position_context="x",
        forward_outlook="x",
        reasoning="x",
        what_would_change_mind="x",
        full_text="x",
        upcoming_catalysts=[_cat("news:N3")],
    )
    out, _ = repair_catalysts({"AAPL": review}, {"AAPL": []}, today=TODAY)
    assert out["AAPL"].upcoming_catalysts == []


# --- schema back-compat + provider compatibility -----------------------------


def test_reports_without_catalysts_still_validate():
    raw = _report().model_dump()
    raw.pop("upcoming_catalysts")
    assert AnalystReport.model_validate(raw).upcoming_catalysts == []


def test_catalyst_schema_has_no_integer_enums():
    # Gemini's structured-output converter rejects integer enum values
    # (the fragility_rank bug) — keep every enum here string-valued.
    schema = json.dumps(Catalyst.model_json_schema())
    for prop in Catalyst.model_json_schema()["properties"].values():
        for value in prop.get("enum", []):
            assert isinstance(value, str), schema


# --- formatting + downstream use --------------------------------------------


def test_block_lists_dated_catalysts_first():
    block = format_catalyst_block(
        [
            _cat(expected_date=None, event="EVENT_WITHOUT_DATE"),
            _cat(expected_date="2026-10-01", event="EVENT_WITH_DATE"),
        ]
    )
    assert block.index("EVENT_WITH_DATE") < block.index("EVENT_WITHOUT_DATE")


def test_block_says_none_when_empty():
    assert "none identified" in format_catalyst_block([])


def test_ranker_prompt_carries_each_candidates_catalysts():
    ranker = Ranker([("claude", "m1")])
    captured: dict[str, str] = {}

    class _Result:
        content = RankerOutput.model_construct(
            picks=[], pairs_not_to_hold_together=[], full_text=""
        )

    def _fake_round(agent, prompt):
        captured["prompt"] = prompt
        return _Result()

    ranker._run_round = _fake_round  # type: ignore[method-assign]
    ranker._rank_once(
        ranker._agents[0],
        {"NVDA": _report("NVDA", [_cat(event="Blackwell ramp update")]), "AMD": _report("AMD")},
        "",
        5,
        "",
    )
    prompt = captured["prompt"]
    assert "Blackwell ramp update" in prompt
    assert "none identified" in prompt  # AMD had no catalysts


def test_holding_payload_includes_recent_news():
    payloads = build_holding_review_payloads(
        positions={"AAPL": {"avg_buy_price": 100.0, "units": 1, "cost_basis": 100.0}},
        fund={},
        tech={"AAPL": {"price": 110.0}},
        rfs={},
        insider_selling={},
        finnhub_signals={},
        eps_revisions={},
        position_splits={},
        account_meta={},
        tax_lots_raw={},
        share_trades={},
        holdings_quarterly_mda={},
        holdings_peers={},
        holdings_transcripts={},
        news={},
        risk_factors_chars=10,
        quarterly_mda_chars=10,
        transcript_chars=10,
        recent_news={"AAPL": [{"id": "N1", "title": "x"}]},
    )
    assert payloads["AAPL"]["recent_news"] == [{"id": "N1", "title": "x"}]


def test_cards_render_catalysts_in_html_and_pdf():
    catalysts = catalysts_to_dicts([_cat(event="FDA panel vote", direction="uncertain")])
    sections = [
        Section(kind="pick_card", data={"ticker": "NVDA", "rank": 1, "catalysts": catalysts}),
        Section(
            kind="holding_review_card",
            data={"ticker": "AAPL", "verdict": "HOLD", "catalysts": catalysts},
        ),
    ]
    html_out = render_html_email(sections, {})
    assert html_out.count("Upcoming catalysts") == 2
    assert "FDA panel vote" in html_out
    assert render_pdf(sections, {}).startswith(b"%PDF")
