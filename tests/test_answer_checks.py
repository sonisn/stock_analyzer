"""Stage answer rules (discover/answer_checks.py) and the retry they drive."""

from __future__ import annotations

from types import SimpleNamespace

from stock_analyzer.discover.answer_checks import (
    ranker_check,
    rebalancer_check,
    redteam_check,
    reviewer_check,
    sizer_check,
)

from .test_ranker_consensus import _output


def _scenarios(*labels):
    return [SimpleNamespace(label=label) for label in labels]


def test_a_ranker_answer_from_the_candidates_passes():
    out = _output(["NVDA", "MSFT"])
    assert ranker_check(["nvda", "MSFT", "AAPL"], top_n=2)(out) == []


def test_a_ranker_pick_that_was_never_a_candidate_is_named():
    (problem,) = ranker_check(["NVDA", "MSFT"], top_n=2)(_output(["NVDA", "PLTR"]))
    assert "PLTR" in problem and "not among the candidate" in problem


def test_ranker_count_duplicates_and_ranks_are_checked():
    problems = ranker_check(["A", "B", "C"], top_n=3)(_output(["A", "A"]))
    assert any("more than once" in p for p in problems)
    assert any("exactly 3" in p for p in problems)


def test_fewer_candidates_than_asked_for_needs_only_that_many():
    assert ranker_check(["A", "B"], top_n=5)(_output(["A", "B"])) == []


def test_ranker_scenarios_must_be_bull_base_bear():
    pick = SimpleNamespace(ticker="A", rank=1, scenarios=_scenarios("bull", "bull", "bear"))
    (problem,) = ranker_check(["A"], top_n=1)(SimpleNamespace(picks=[pick]))
    assert "bull, base and bear" in problem


def test_redteam_needs_one_bear_case_per_pick_and_no_strays():
    out = SimpleNamespace(bear_cases=[SimpleNamespace(ticker="A"), SimpleNamespace(ticker="Z")])
    problems = redteam_check(["A", "B"])(out)
    assert any("Z" in p and "not among the picks" in p for p in problems)
    assert any("no bear case for B" in p for p in problems)


def test_sizer_allocates_only_to_picks():
    out = SimpleNamespace(allocations=[SimpleNamespace(ticker="A"), SimpleNamespace(ticker="X")])
    (problem,) = sizer_check(["A", "B"])(out)
    assert "X" in problem


def test_a_low_confidence_sale_verdict_is_flagged_and_a_hold_is_not():
    assert reviewer_check(SimpleNamespace(verdict="SELL", confidence=5))
    assert reviewer_check(SimpleNamespace(verdict="TRIM", confidence=7)) == []
    assert reviewer_check(SimpleNamespace(verdict="HOLD", confidence=2)) == []


def test_the_rebalancer_sells_only_what_is_held():
    actions = [
        SimpleNamespace(action="SELL", ticker="AAPL"),
        SimpleNamespace(action="WRITE_CALL", ticker="ZZZ"),
        SimpleNamespace(action="BUY", ticker="NEW"),  # buys may name anything
    ]
    (problem,) = rebalancer_check(["AAPL"])(SimpleNamespace(actions=actions))
    assert "ZZZ" in problem and "NEW" not in problem


def test_the_ranker_sends_a_stray_pick_back_to_the_model(monkeypatch):
    """End to end through the real agent: the first answer names a ticker
    it was never given, the model is told so, and the corrected answer is
    the one used."""
    import json

    from stock_analyzer.discover.ranker import Ranker

    from .llm_fakes import script

    def answer(tickers):
        picks = [
            {
                "rank": i + 1,
                "ticker": t,
                "one_liner": "x",
                "why_over_alternatives": "x",
                "conviction": 7,
                "sector_concentration_check": "x",
                "bull_thesis": "x",
                "what_youre_betting_on": "x",
                "scenarios": [
                    {"label": label, "probability": p, "target_return_pct": r, "rationale": "x"}
                    for label, p, r in (("bull", 0.3, 40), ("base", 0.5, 10), ("bear", 0.2, -20))
                ],
            }
            for i, t in enumerate(tickers)
        ]
        return json.dumps({"picks": picks, "full_text": "text"})

    seen = script(
        monkeypatch, [(answer(["NVDA", "PLTR"]), "stop"), (answer(["NVDA", "MSFT"]), "stop")]
    )
    ranker = Ranker([("claude", "claude-opus-5-5")])
    out = ranker.rank({"NVDA": "a", "MSFT": "b"}, "", top_n=2)
    assert [p.ticker for p in out.picks] == ["NVDA", "MSFT"]
    assert seen.calls == 2
    retry = [
        part.content
        for message in seen.requests[1]["messages"]
        for part in message.parts
        if type(part).__name__ == "RetryPromptPart"
    ]
    assert "PLTR" in str(retry)
