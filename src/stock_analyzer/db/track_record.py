"""Read-only analytics queries used by discover/track_record.py.

These produce the (run_at, ticker) tuples consumed by _dedup_oldest in
the orchestration module. Query logic stays here so SQL stays out of
the business layer; the orchestrator handles ordering + deduplication.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy import func
from sqlmodel import Session, select

from .tables import HoldingReviewRow, Pick, Run


def fetch_recent_pick_runs_with_model(
    session: Session, *, lookback_days: int
) -> list[tuple[str, str, str | None]]:
    """Every (run_at, ticker, opus_model) for BUY picks in the last
    `lookback_days`, oldest-first. opus_model may be None for legacy
    runs that did not record it. Dedup happens in the caller."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    rows = session.exec(
        select(Run.run_at, Pick.ticker, Run.opus_model)
        .join(Pick, Pick.run_id == Run.id)
        .where(Run.run_at >= cutoff)
        .order_by(Run.run_at.asc())
    )
    return [(row.run_at, row.ticker, row.opus_model) for row in rows]


def fetch_recent_verdict_runs(
    session: Session,
    verdict: Literal["SELL", "TRIM", "HOLD"],
    *,
    lookback_days: int,
) -> list[tuple[str, str]]:
    """(run_at, ticker) for holdings_reviews rows with the given verdict
    in the last `lookback_days`, oldest-first. Filters apply
    UPPER(COALESCE(verdict, '')) match so legacy lowercased rows still
    register. Dedup happens in the caller."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    verdict_upper = func.upper(func.coalesce(HoldingReviewRow.verdict, ""))
    rows = session.exec(
        select(Run.run_at, HoldingReviewRow.ticker)
        .join(HoldingReviewRow, HoldingReviewRow.run_id == Run.id)
        .where(Run.run_at >= cutoff)
        .where(verdict_upper == verdict)
        .order_by(Run.run_at.asc())
    )
    return [(row.run_at, row.ticker) for row in rows]


def fetch_recent_sell_runs(
    session: Session, *, lookback_days: int
) -> list[tuple[str, str]]:
    """Back-compat wrapper: SELL + TRIM combined. New code should call
    fetch_recent_verdict_runs per verdict instead so trim/sell can be
    reported separately."""
    sells = fetch_recent_verdict_runs(session, "SELL", lookback_days=lookback_days)
    trims = fetch_recent_verdict_runs(session, "TRIM", lookback_days=lookback_days)
    return sorted(sells + trims, key=lambda row: row[0])


__all__ = [
    "fetch_recent_pick_runs_with_model",
    "fetch_recent_verdict_runs",
    "fetch_recent_sell_runs",
]
