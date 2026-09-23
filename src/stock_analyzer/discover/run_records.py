"""The rows every run writes, discover and rebalance alike: screened
candidates, their point-in-time snapshots, analyst scorecards, and picks
with the forecasts calibration grades later."""

from __future__ import annotations

from typing import Any

from sqlmodel import Session

from ..db.repository import (
    insert_candidate,
    insert_candidate_snapshot,
    insert_pick,
    insert_pick_catalysts,
    insert_scorecard,
)
from .catalysts import catalysts_to_dicts


def insert_candidates(session: Session, run_id: int, candidates: list[dict[str, Any]]) -> None:
    for c in candidates:
        insert_candidate(
            session,
            run_id,
            c["ticker"],
            passed_filter=c["passed_filter"],
            fail_reasons=c["fail_reasons"],
            score=c["score"],
            score_components=c["score_components"],
            score_breakdown=c["score_breakdown"],
            sources=c["sources"],
            conviction=c["conviction"],
            sector=c["sector"],
            price=c["price"],
        )


def insert_snapshots(
    session: Session,
    run_id: int,
    candidates: list[dict[str, Any]],
    fundamentals: dict[str, Any],
    revisions: dict[str, Any],
) -> None:
    """Fundamentals and estimate revisions as they were at screen time."""
    for c in candidates:
        if c["ticker"] in fundamentals:
            insert_candidate_snapshot(
                session,
                run_id,
                c["ticker"],
                fundamentals.get(c["ticker"]),
                revisions.get(c["ticker"]),
            )


def insert_scorecards(session: Session, run_id: int, analyses: dict[str, Any]) -> None:
    for ticker, report in analyses.items():
        analyst_text = getattr(report, "full_text", None) or (
            report if isinstance(report, str) else ""
        )
        insert_scorecard(session, run_id, ticker, analyst_text)


def insert_picks(
    session: Session,
    run_id: int,
    picks: list[tuple[int, str, str]],
    *,
    ranker_output: object,
    candidates: list[dict[str, Any]],
    analyses: dict[str, Any],
) -> None:
    """Forecast fields travel with the pick so calibration can grade them
    later; `entry_price` is the screen-time price, never a refetch, so a
    historical pick is never repriced with new data."""
    from ..cli.discover_steps.helpers import _pick_forecasts

    forecasts = _pick_forecasts(ranker_output)
    prices = {c["ticker"]: c.get("price") for c in candidates}
    for rank, ticker, _ in picks:
        forecast = forecasts.get(ticker, {})
        insert_pick(
            session,
            run_id,
            rank=rank,
            ticker=ticker,
            conviction=forecast.get("conviction"),
            ev_pct=forecast.get("ev_pct"),
            entry_price=prices.get(ticker),
            time_horizon=forecast.get("time_horizon"),
            scenarios=forecast.get("scenarios"),
            agreement_ratio=forecast.get("agreement_ratio"),
            voting_providers=forecast.get("voting_providers"),
        )
        analysis = analyses.get(ticker)
        if analysis is not None and getattr(analysis, "upcoming_catalysts", None):
            insert_pick_catalysts(
                session, run_id, ticker, catalysts_to_dicts(analysis.upcoming_catalysts)
            )
