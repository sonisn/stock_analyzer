"""CRUD repository for the SQLite analytics DB.

Signatures mirror the legacy discover/persistence.py API exactly — same
keyword args, same return types — except the first argument is now a
Session instead of a sqlite3.Connection. JSON marshalling lives here:
the table classes hold raw TEXT, the repository converts to/from
Python types at the boundary.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlmodel import Session, select

from .tables import (
    Candidate,
    HoldingReviewRow,
    Pick,
    Run,
    RunOutput,
    Scorecard,
)

# --- runs -----------------------------------------------------------------

def insert_run(
    session: Session,
    *,
    universe_size: int,
    survivors: int,
    picks: int,
    opus_model: str,
    sonnet_model: str,
    cash_budget: float | None,
    kind: str = "discover",
) -> int:
    """Insert a run row, return the assigned run_id."""
    row = Run(
        run_at=datetime.now().isoformat(timespec="seconds"),
        kind=kind,
        universe_size=universe_size,
        survivors=survivors,
        picks=picks,
        opus_model=opus_model,
        sonnet_model=sonnet_model,
        cash_budget=cash_budget,
    )
    session.add(row)
    session.flush()  # populate row.id without committing
    if row.id is None:
        raise RuntimeError("Run.id was None after session.flush() — SQLite autoincrement failed")
    return int(row.id)


# --- candidates -----------------------------------------------------------

def insert_candidate(
    session: Session,
    run_id: int,
    ticker: str,
    *,
    passed_filter: bool,
    fail_reasons: list[str],
    score: float | None,
    score_components: dict[str, Any] | None,
    score_breakdown: dict[str, Any] | None,
    sources: list[str],
    conviction: int,
    sector: str | None,
    price: float | None,
) -> None:
    session.add(Candidate(
        run_id=run_id,
        ticker=ticker,
        passed_filter=int(passed_filter),
        fail_reasons=json.dumps(fail_reasons),
        score=score,
        score_components=json.dumps(score_components) if score_components else None,
        score_breakdown=json.dumps(score_breakdown) if score_breakdown else None,
        sources=json.dumps(sources),
        conviction=conviction,
        sector=sector,
        price=price,
    ))


# --- scorecards -----------------------------------------------------------

def insert_scorecard(
    session: Session, run_id: int, ticker: str, text: str
) -> None:
    session.add(Scorecard(run_id=run_id, ticker=ticker, analyst_text=text))


# --- picks ----------------------------------------------------------------

def insert_pick(
    session: Session,
    run_id: int,
    *,
    rank: int,
    ticker: str,
    ranker_text: str,
    bear_case_text: str | None,
    allocation_text: str | None,
) -> None:
    session.add(Pick(
        run_id=run_id,
        rank=rank,
        ticker=ticker,
        ranker_text=ranker_text,
        bear_case_text=bear_case_text,
        allocation_text=allocation_text,
    ))


# --- holdings reviews -----------------------------------------------------

def insert_holdings_review(
    session: Session,
    run_id: int,
    ticker: str,
    *,
    verdict: str | None,
    confidence: int | None,
    review_text: str,
) -> None:
    session.add(HoldingReviewRow(
        run_id=run_id,
        ticker=ticker,
        verdict=verdict,
        confidence=confidence,
        review_text=review_text,
    ))


def fetch_recent_holdings_history(
    session: Session, *, n_runs: int = 3, kind: str = "rebalance"
) -> dict[str, list[dict[str, Any]]]:
    """Return {ticker: [{run_at, verdict, confidence}, ...]} oldest-first for
    the last `n_runs` runs of `kind`. Same shape as the legacy function."""
    # DESC + LIMIT to grab the most recent N rows; then reverse to chronological
    # (ascending) order so the LLM reads them oldest-first.
    recent_runs = list(session.exec(
        select(Run.id, Run.run_at)
        .where(Run.kind == kind)
        .order_by(Run.id.desc())
        .limit(n_runs)
    ))
    if not recent_runs:
        return {}
    recent_runs.reverse()
    out: dict[str, list[dict[str, Any]]] = {}
    for run_row in recent_runs:
        rows = session.exec(
            select(
                HoldingReviewRow.ticker,
                HoldingReviewRow.verdict,
                HoldingReviewRow.confidence,
            ).where(HoldingReviewRow.run_id == run_row.id)
        )
        for review_row in rows:
            out.setdefault(review_row.ticker, []).append({
                "run_at": run_row.run_at,
                "verdict": review_row.verdict,
                "confidence": review_row.confidence,
            })
    return out


# --- run outputs ----------------------------------------------------------

def insert_run_outputs(
    session: Session,
    run_id: int,
    *,
    ranker_full: str,
    redteam_full: str,
    sizer_full: str,
    holdings_summary: str,
    rebalance_text: str | None = None,
    dashboard_data: dict[str, Any] | None = None,
) -> None:
    session.add(RunOutput(
        run_id=run_id,
        ranker_full=ranker_full,
        redteam_full=redteam_full,
        sizer_full=sizer_full,
        holdings_summary=holdings_summary,
        rebalance_text=rebalance_text,
        dashboard_data=(
            json.dumps(dashboard_data) if dashboard_data is not None else None
        ),
    ))


__all__ = [
    "insert_run",
    "insert_candidate",
    "insert_scorecard",
    "insert_pick",
    "insert_holdings_review",
    "fetch_recent_holdings_history",
    "insert_run_outputs",
]
