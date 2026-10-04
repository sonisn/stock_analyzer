"""Rules each deciding stage's answer must keep, checked before it is used.

Each function builds an `llm.Check`: given the stage's parsed answer, the
list of rules it breaks (empty when none). `llm.ModelAgent` sends a broken
answer back to the model once with that list; an answer still broken is
kept and the deterministic safeguards downstream decide, as before.

These are cheap facts the run already holds — which tickers were
candidates, which are held — so a retry is paid for only when the model
actually named something it was never given.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from typing import Any

from ..llm import Check


def _norm(ticker: str) -> str:
    return (ticker or "").strip().upper()


def _unknown(tickers: Iterable[str], known: set[str]) -> list[str]:
    return sorted({_norm(t) for t in tickers} - known)


def ranker_check(candidates: Iterable[str], top_n: int) -> Check:
    """Picks are drawn from the candidates analysed, once each, as many as
    asked for, ranked 1..n, with bull/base/bear scenarios."""
    known = {_norm(t) for t in candidates}
    want = min(top_n, len(known))

    def check(output: Any) -> list[str]:
        picks = list(getattr(output, "picks", None) or [])
        problems: list[str] = []
        tickers = [_norm(p.ticker) for p in picks]
        if unknown := _unknown(tickers, known):
            problems.append(
                f"picks {', '.join(unknown)} are not among the candidate tickers analysed; "
                "pick only from the candidates provided"
            )
        if dupes := sorted(t for t, n in Counter(tickers).items() if n > 1):
            problems.append(f"{', '.join(dupes)} picked more than once")
        if len(picks) != want:
            problems.append(f"{len(picks)} picks given; exactly {want} are required")
        if sorted(p.rank for p in picks) != list(range(1, len(picks) + 1)):
            problems.append("ranks must run 1..n with no gaps or repeats")
        for p in picks:
            labels = sorted(s.label for s in p.scenarios or [])
            if labels and labels != ["base", "bear", "bull"]:
                problems.append(f"{_norm(p.ticker)} needs exactly one bull, base and bear scenario")
        return problems

    return check


def redteam_check(pick_tickers: Iterable[str]) -> Check:
    """One bear case per pick, and none for a ticker that was not picked."""
    known = {_norm(t) for t in pick_tickers}

    def check(output: Any) -> list[str]:
        named = [_norm(b.ticker) for b in getattr(output, "bear_cases", None) or []]
        problems: list[str] = []
        if unknown := _unknown(named, known):
            problems.append(f"bear cases for {', '.join(unknown)}, which are not among the picks")
        if missing := sorted(known - set(named)):
            problems.append(f"no bear case for {', '.join(missing)}")
        return problems

    return check


def sizer_check(pick_tickers: Iterable[str]) -> Check:
    """Allocations go only to the picks."""
    known = {_norm(t) for t in pick_tickers}

    def check(output: Any) -> list[str]:
        named = [a.ticker for a in getattr(output, "allocations", None) or []]
        if unknown := _unknown(named, known):
            return [f"allocations to {', '.join(unknown)}, which are not among the picks"]
        return []

    return check


def reviewer_check(output: Any) -> list[str]:
    """TRIM and SELL need confidence 7 or more (the calibration rule). The
    old repair silently turned such a verdict into HOLD; the model now gets
    to say which it meant."""
    verdict = getattr(output, "verdict", "HOLD")
    confidence = getattr(output, "confidence", 10)
    if verdict != "HOLD" and confidence < 7:
        return [
            f"verdict {verdict} with confidence {confidence}: TRIM and SELL require "
            "confidence >= 7. Either raise the confidence with the evidence for it, "
            "or make the verdict HOLD"
        ]
    return []


_SALE_ACTIONS = frozenset({"SELL", "TRIM", "WRITE_CALL"})


def rebalancer_check(held_tickers: Iterable[str]) -> Check:
    """A sale, trim or covered call names a position actually held."""
    held = {_norm(t) for t in held_tickers}

    def check(output: Any) -> list[str]:
        named = [
            a.ticker for a in getattr(output, "actions", None) or [] if a.action in _SALE_ACTIONS
        ]
        if unknown := _unknown(named, held):
            return [
                f"SELL/TRIM/WRITE_CALL actions on {', '.join(unknown)}, which are not held; "
                "those actions apply only to current holdings"
            ]
        return []

    return check
