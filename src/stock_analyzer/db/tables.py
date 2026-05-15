"""SQLModel table classes mapped 1:1 to the existing SQLite schema.

JSON-blob columns (fail_reasons, score_components, score_breakdown,
sources, dashboard_data) stay as Optional[str] here; repository
functions own the json.dumps/json.loads boundary. This keeps the
on-disk format byte-identical to the legacy raw-sqlite schema.

Composite primary keys use multiple Field(primary_key=True) entries.
Foreign keys preserve ON DELETE CASCADE via the ondelete arg.
"""
from __future__ import annotations

from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: int | None = Field(default=None, primary_key=True)
    run_at: str
    kind: str = Field(default="discover")
    universe_size: int
    survivors: int
    picks: int
    opus_model: str | None = None
    sonnet_model: str | None = None
    cash_budget: float | None = None


class Candidate(SQLModel, table=True):
    __tablename__ = "candidates"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    passed_filter: int
    fail_reasons: str | None = None          # JSON list
    score: float | None = None
    score_components: str | None = None      # JSON
    score_breakdown: str | None = None       # JSON
    sources: str | None = None               # JSON list
    conviction: int | None = None
    sector: str | None = None
    price: float | None = None


class Scorecard(SQLModel, table=True):
    __tablename__ = "scorecards"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    analyst_text: str | None = None


class Pick(SQLModel, table=True):
    __tablename__ = "picks"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    rank: int = Field(primary_key=True)
    ticker: str
    ranker_text: str
    bear_case_text: str | None = None
    allocation_text: str | None = None


class HoldingReviewRow(SQLModel, table=True):
    """ORM table for holdings reviews. The `Row` suffix avoids collision
    with `stock_analyzer.models.llm.HoldingReview` (the Pydantic DTO that
    represents the LLM's structured output)."""

    __tablename__ = "holdings_reviews"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    verdict: str | None = None
    confidence: int | None = None
    review_text: str | None = None


class RunOutput(SQLModel, table=True):
    __tablename__ = "run_outputs"

    run_id: int = Field(
        primary_key=True,
        foreign_key="runs.id",
        ondelete="CASCADE",
    )
    ranker_full: str | None = None
    redteam_full: str | None = None
    sizer_full: str | None = None
    holdings_summary: str | None = None
    rebalance_text: str | None = None
    dashboard_data: str | None = None        # JSON


__all__ = [
    "Run", "Candidate", "Scorecard", "Pick", "HoldingReviewRow", "RunOutput",
]
