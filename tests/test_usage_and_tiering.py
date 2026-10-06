"""Per-run LLM cost accounting and the two-tier Analyst. No LLM calls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic_ai.usage import RequestUsage

from stock_analyzer.discover.analyst import analyze_tiered
from stock_analyzer.discover.report_sections import append_usage_section
from stock_analyzer.llm import ModelAgent
from stock_analyzer.usage import TRACKER, UsageTracker


def _metrics(inp=0, out=0, cache_read=0, cache_write=0):
    return SimpleNamespace(
        input_tokens=inp,
        output_tokens=out,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )


# --- tracker ---------------------------------------------------------------


def test_cost_uses_list_prices_and_cache_multipliers():
    t = UsageTracker()
    t.record("Analyst", "claude-sonnet-5", _metrics(inp=1_000_000, out=1_000_000))
    t.record("Analyst", "claude-sonnet-5", _metrics(cache_read=1_000_000, cache_write=1_000_000))
    (row,) = t.rows()
    assert row.calls == 2
    # $2 in + $10 out + $0.20 cache read (0.1x) + $2.50 cache write (1.25x)
    assert row.cost_usd == pytest.approx(14.70)
    assert t.total_cost() == (pytest.approx(14.70), True)


def test_unpriced_models_report_tokens_but_flag_total_incomplete():
    t = UsageTracker()
    t.record("Ranker", "gemini-flash-lite-latest", _metrics(inp=500, out=100))
    t.record("Ranker", "claude-opus-5", _metrics(inp=1_000_000))
    data = t.report_data()
    gemini = next(r for r in data["rows"] if r["model"] == "gemini-flash-lite-latest")
    assert gemini["cost_usd"] is None
    assert gemini["input_tokens"] == 500
    assert data["total_cost_usd"] == pytest.approx(5.0)
    assert data["cost_complete"] is False


def test_the_ranker_consensus_models_are_priced():
    # gpt-6-astra ran uncounted by the cost cap until 2026-09-27: at $10/$50
    # it is the dearest model in the pipeline.
    t = UsageTracker()
    t.record("Ranker", "gpt-6-astra", _metrics(inp=24_272, out=11_697))
    t.record("Ranker", "gemini-pro-latest", _metrics(inp=25_801, out=12_223))
    assert t.total_cost() == (pytest.approx(0.8277 + 0.1983, abs=1e-3), True)


def test_rows_sorted_most_expensive_first():
    t = UsageTracker()
    t.record("Cheap", "claude-haiku-4-5", _metrics(inp=1000))
    t.record("Pricey", "claude-opus-5", _metrics(out=100_000))
    assert [r.stage for r in t.rows()] == ["Pricey", "Cheap"]


def test_missing_metrics_is_ignored():
    t = UsageTracker()
    t.record("Analyst", "claude-sonnet-5", None)
    assert t.rows() == []


def test_agent_run_records_usage_under_agent_name(monkeypatch):
    from .llm_fakes import script

    TRACKER.reset()
    script(monkeypatch, [("x", "stop")], usage=RequestUsage(output_tokens=2000))
    ModelAgent("Sizer", "claude", "claude-opus-5").run("prompt")
    (row,) = TRACKER.rows()
    assert (row.stage, row.model, row.output_tokens) == ("Sizer", "claude-opus-5", 2000)
    TRACKER.reset()


# --- report section --------------------------------------------------------


def test_usage_section_renders_table_and_total():
    t = UsageTracker()
    t.record("Analyst", "claude-sonnet-5", _metrics(inp=1_000_000))
    t.record("RedTeam", "gemini-flash-lite-latest", _metrics(inp=10))
    sections: list = []
    append_usage_section(sections, t.report_data())
    kinds = [s.kind for s in sections]
    assert kinds == ["heading", "table", "para"]
    assert sections[1].table_rows[0][-1] == "$2.00"
    assert "unpriced" in sections[2].text


def test_usage_section_skipped_without_calls():
    sections: list = []
    append_usage_section(sections, UsageTracker().report_data())
    append_usage_section(sections, None)
    assert sections == []


# --- two-tier analyst --------------------------------------------------------


class _FakeAnalyst:
    def __init__(self, name: str, fail: set[str] | None = None):
        self.name = name
        self.fail = fail or set()
        self.seen: list[str] = []

    def analyze(self, ticker, payload):
        self.seen.append(ticker)
        return None if ticker in self.fail else f"{self.name}:{ticker}"


_PAYLOADS = {t: {} for t in ["A", "B", "C", "D"]}  # screen-score order


def test_deep_model_gets_top_tickers_and_light_the_rest():
    deep, light = _FakeAnalyst("deep"), _FakeAnalyst("light")
    out = analyze_tiered(deep, light, _PAYLOADS, {"A", "B"})
    assert sorted(deep.seen) == ["A", "B"]
    assert sorted(light.seen) == ["C", "D"]
    assert list(out) == ["A", "B", "C", "D"]  # order preserved for the Ranker
    assert out["C"] == "light:C"


def test_light_failures_are_retried_on_deep_model():
    deep, light = _FakeAnalyst("deep"), _FakeAnalyst("light", fail={"D"})
    out = analyze_tiered(deep, light, _PAYLOADS, {"A"})
    assert out["D"] == "deep:D"
    assert sorted(deep.seen) == ["A", "D"]


def test_no_light_model_sends_everything_to_deep():
    deep = _FakeAnalyst("deep")
    out = analyze_tiered(deep, None, _PAYLOADS, set())
    assert sorted(deep.seen) == ["A", "B", "C", "D"]
    assert len(out) == 4


def test_deep_failures_are_simply_dropped():
    deep = _FakeAnalyst("deep", fail={"A"})
    out = analyze_tiered(deep, _FakeAnalyst("light"), _PAYLOADS, {"A"})
    assert "A" not in out
    assert list(out) == ["B", "C", "D"]


def test_a_billed_cost_replaces_the_list_price_estimate():
    """OpenRouter bills the cheap host it routed to, well under list price."""

    t = UsageTracker()
    t.record(
        "Reader", "z-ai/glm-5.3", SimpleNamespace(input_tokens=10**6, output_tokens=0, cost_usd=0.5)
    )
    t.record(
        "Reader",
        "z-ai/glm-5.3",
        SimpleNamespace(input_tokens=10**6, output_tokens=0, cost_usd=0.25),
    )
    assert t.rows()[0].cost_usd == 0.75
    # One call without a billed figure: back to the (higher) list-price estimate.
    t.record("Reader", "z-ai/glm-5.3", _metrics(inp=10))
    assert t.rows()[0].cost_usd != 0.75
