"""Ranker consensus math: agreement_ratio / voting_providers.

`Ranker.rank()` majority-votes ticker membership across N independent
per-provider rounds; this asserts the agreement fraction and provider
attribution it attaches to the winning run's picks, without making any
real LLM calls (each round's `_rank_once` is monkeypatched to return a
canned `RankerOutput`).
"""

from __future__ import annotations

import pytest

from stock_analyzer.discover.ranker import Ranker
from stock_analyzer.models.llm import RankerOutput, RankerPick


def _output(tickers: list[str]) -> RankerOutput:
    picks = [
        RankerPick.model_construct(
            rank=i + 1,
            ticker=t,
            one_liner="x",
            why_over_alternatives="x",
            conviction=7,
            time_horizon="6-12 months",
            sector_concentration_check="x",
            bull_thesis="x",
            what_youre_betting_on="x",
            scenarios=[],
        )
        for i, t in enumerate(tickers)
    ]
    return RankerOutput.model_construct(
        picks=picks, pairs_not_to_hold_together=[], full_text="text"
    )


def _ranker(rounds):
    # Real Ranker construction (builds real AgnoAgent/model dataclasses,
    # no network call) so `rank()`'s wiring is exercised as-is; only the
    # per-round LLM call itself is stubbed.
    return Ranker(rounds)


def test_unanimous_pick_gets_agreement_ratio_one():
    ranker = _ranker([("claude", "m1"), ("gemini", "m2"), ("openai", "m3")])
    outputs = iter(
        [_output(["AAPL", "MSFT"]), _output(["AAPL", "NVDA"]), _output(["AAPL", "MSFT"])]
    )
    ranker._rank_once = lambda agent, *a, **k: next(outputs)  # type: ignore[method-assign]

    result = ranker.rank({}, "", macro_context="")
    by_ticker = {p.ticker: p for p in result.picks}

    assert by_ticker["AAPL"].agreement_ratio == 1.0
    assert set(by_ticker["AAPL"].voting_providers) == {"claude", "gemini", "openai"}


def test_bare_majority_pick_gets_fractional_agreement_ratio():
    ranker = _ranker([("claude", "m1"), ("gemini", "m2"), ("openai", "m3")])
    outputs = iter(
        [_output(["AAPL", "MSFT"]), _output(["AAPL", "NVDA"]), _output(["AAPL", "MSFT"])]
    )
    ranker._rank_once = lambda agent, *a, **k: next(outputs)  # type: ignore[method-assign]

    result = ranker.rank({}, "", macro_context="")
    by_ticker = {p.ticker: p for p in result.picks}

    assert by_ticker["MSFT"].agreement_ratio == pytest.approx(2 / 3)
    assert set(by_ticker["MSFT"].voting_providers) == {"claude", "openai"}
    # NVDA only appeared in one round (1/3, below the ceil(3/2)=2 majority
    # threshold) so it never reaches the winning run's pick set at all.
    assert "NVDA" not in by_ticker


def test_single_round_leaves_agreement_ratio_none():
    ranker = _ranker([("claude", "m1")])
    ranker._rank_once = lambda agent, *a, **k: _output(["AAPL"])  # type: ignore[method-assign]

    result = ranker.rank({}, "", macro_context="")

    assert result.picks[0].agreement_ratio is None
    assert result.picks[0].voting_providers is None


def test_no_consensus_falls_back_to_first_run_with_no_agreement_data():
    ranker = _ranker([("claude", "m1"), ("gemini", "m2"), ("openai", "m3")])
    # Every round picks a completely different ticker — no majority possible.
    outputs = iter([_output(["AAPL"]), _output(["MSFT"]), _output(["NVDA"])])
    ranker._rank_once = lambda agent, *a, **k: next(outputs)  # type: ignore[method-assign]

    result = ranker.rank({}, "", macro_context="")

    assert result.picks[0].ticker == "AAPL"
    assert result.picks[0].agreement_ratio is None
