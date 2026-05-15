"""Read-only analytics queries used by discover/track_record.py.

These produce the (run_at, ticker) tuples consumed by _dedup_oldest in
the orchestration module. Query logic stays here so SQL stays out of
the business layer; the orchestrator handles ordering + deduplication.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func
from sqlmodel import Session, select

from .tables import HoldingReviewRow, Pick, Run


def fetch_recent_pick_runs(
    session: Session, *, lookback_days: int
) -> list[tuple[str, str]]:
    """Every (run_at, ticker) for BUY picks in the last `lookback_days`,
    oldest-first. Dedup happens in the caller."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    rows = session.exec(
        select(Run.run_at, Pick.ticker)
        .join(Pick, Pick.run_id == Run.id)
        .where(Run.run_at >= cutoff)
        .order_by(Run.run_at.asc())
    )
    return [(row.run_at, row.ticker) for row in rows]


def fetch_recent_sell_runs(
    session: Session, *, lookback_days: int
) -> list[tuple[str, str]]:
    """Every (run_at, ticker) for SELL/TRIM holdings reviews in the last
    `lookback_days`, oldest-first. Dedup happens in the caller.

    SELL and TRIM both count: TRIM is a softer SELL but still a directional
    'reduce exposure' call we should be held accountable for. HOLD and NULL
    are filtered out."""
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    verdict_upper = func.upper(func.coalesce(HoldingReviewRow.verdict, ""))
    rows = session.exec(
        select(Run.run_at, HoldingReviewRow.ticker)
        .join(HoldingReviewRow, HoldingReviewRow.run_id == Run.id)
        .where(Run.run_at >= cutoff)
        .where(verdict_upper.in_(("SELL", "TRIM")))
        .order_by(Run.run_at.asc())
    )
    return [(row.run_at, row.ticker) for row in rows]


__all__ = ["fetch_recent_pick_runs", "fetch_recent_sell_runs"]
