"""Per-run cost cap: call-level refusal, stage planning, report line."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from stock_analyzer import usage
from stock_analyzer.discover.analyst import (
    ANALYST_INSTRUCTIONS,
    EXPECTED_OUTPUT_TOKENS,
    plan_under_budget,
)
from stock_analyzer.discover.ranker import Ranker
from stock_analyzer.discover.report_sections import append_usage_section
from stock_analyzer.llm import AgnoAgent, run_with_fallback
from stock_analyzer.usage import BUDGET, TRACKER, BudgetExceededError, estimate_cost

from .test_ranker_consensus import _output


@pytest.fixture(autouse=True)
def _clean_budget():
    TRACKER.reset()
    BUDGET.configure(None)
    yield
    TRACKER.reset()
    BUDGET.configure(None)


def _spend(model: str, input_tokens: int, output_tokens: int = 0) -> None:
    TRACKER.record(
        "Test", model, SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    )


def test_estimates_and_extra_prices(monkeypatch):
    monkeypatch.setitem(usage._PRICES_PER_MTOK, "claude-haiku-4-5", (1.0, 5.0))
    assert estimate_cost("claude-haiku-4-5", 3_500_000, 200_000) == pytest.approx(1.0 + 1.0)
    assert estimate_cost("gemini-x", 1000, 1000) is None
    usage.set_extra_prices("gemini-x=2:8, junk, gpt-y=bad:1")
    assert usage.price_for("gemini-x") == (2.0, 8.0)
    assert usage.price_for("gpt-y") is None
    monkeypatch.delitem(usage._PRICES_PER_MTOK, "gemini-x")


def test_hold_refuses_calls_that_would_pass_the_cap_including_in_flight():
    BUDGET.configure(1.0)
    _spend("claude-haiku-4-5", 500_000)  # $0.50 spent
    with (
        BUDGET.hold("A", "claude-haiku-4-5", 0, 60_000),  # $0.30 reserved
        pytest.raises(BudgetExceededError),
        BUDGET.hold("B", "claude-haiku-4-5", 0, 60_000),  # would reach $1.10
    ):
        pass
    with BUDGET.hold("C", "claude-haiku-4-5", 0, 60_000):  # reservation released
        pass
    assert "refused a B call" in BUDGET.notes[0]
    with BUDGET.hold("unpriced", "gemini-unknown", 10**9, 10**9):
        pass  # unknown prices never block


def test_agent_run_is_refused_and_never_falls_back():
    BUDGET.configure(0.01)
    agent = AgnoAgent("Analyst", "claude", "claude-sonnet-4-6", instructions="x" * 100_000)
    agent.agent.run = lambda *a, **k: pytest.fail("the model must not be called")
    with pytest.raises(BudgetExceededError):
        run_with_fallback(agent, lambda: pytest.fail("no fallback on a budget refusal"), "hi")


def test_analyst_plan_downgrades_before_dropping_names():
    order = [f"T{i}" for i in range(10)]
    chars = dict.fromkeys(order, 35_000)  # ~10k tokens in per call
    deep = set(order[:4])
    sonnet, haiku = "claude-sonnet-4-6", "claude-haiku-4-5"
    per_haiku = estimate_cost(haiku, 35_000 + len(ANALYST_INSTRUCTIONS), EXPECTED_OUTPUT_TOKENS)

    keep, new_deep, notes = plan_under_budget(order, chars, deep, sonnet, haiku, None)
    assert (keep, new_deep, notes) == (order, deep, [])

    keep, new_deep, notes = plan_under_budget(order, chars, deep, sonnet, haiku, 1000.0)
    assert new_deep == deep and notes == []

    keep, new_deep, notes = plan_under_budget(order, chars, deep, sonnet, haiku, per_haiku * 10.5)
    assert keep == order and new_deep == set() and "instead of claude-sonnet-4-6" in notes[0]

    keep, new_deep, notes = plan_under_budget(order, chars, deep, sonnet, haiku, per_haiku * 6.5)
    assert keep == order[:6] and "top 6 of 10" in notes[-1]

    keep, _, _ = plan_under_budget(order, chars, deep, sonnet, haiku, 0.0)
    assert len(keep) == 5  # never below the floor; the call-level cap backstops


def test_ranker_skips_extra_rounds_it_cannot_afford():
    ranker = Ranker([("claude", "claude-opus-4-7"), ("claude", "claude-opus-4-7")])
    outputs = iter([_output(["AAPL", "MSFT"]), _output(["NVDA"])])
    ranker._rank_once = lambda agent, *a, **k: next(outputs)  # type: ignore[method-assign]
    BUDGET.configure(0.40)  # 25% reserve leaves $0.30, below one Opus round
    result = ranker.rank({"AAPL": "analysis " * 1000}, "")
    assert [p.ticker for p in result.picks] == ["AAPL", "MSFT"]
    assert "ran 1 of 2 Ranker rounds" in BUDGET.notes[0]


def test_usage_section_reports_the_cap_and_cuts():
    BUDGET.configure(5.0)
    _spend("claude-haiku-4-5", 1_000_000)
    BUDGET.note("analysed only the top 6 of 10 survivors by screen score")
    sections = []
    append_usage_section(sections, TRACKER.report_data())
    assert sections[-1].text == (
        "Cost cap $5.00: estimated priced spend $1.00. To stay under it this run: "
        "analysed only the top 6 of 10 survivors by screen score."
    )
