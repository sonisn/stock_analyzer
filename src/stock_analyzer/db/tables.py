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
    fail_reasons: str | None = None  # JSON list
    score: float | None = None
    score_components: str | None = None  # JSON
    score_breakdown: str | None = None  # JSON
    sources: str | None = None  # JSON list
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
    """One of a discover run's top picks.

    The forecast columns below (`conviction`, `ev_pct`, `entry_price`) and
    the `PickScenario` rows exist so the ranker's own calibration can be
    measured. The ranker prompt tells Opus it will be "measured on the EV
    vs realized return"; until these were persisted that promise could not
    be kept, because only the prose rendering reached disk and the
    probabilities were recomputed for the report and then discarded.

    `entry_price` is the screen-time price, stored rather than refetched so
    calibration never reprices a historical pick with today's data.
    """

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
    # --- forecast, for calibration scoring ---
    conviction: int | None = None
    ev_pct: float | None = None  # Σ(probability × target_return_pct)
    entry_price: float | None = None
    time_horizon: str | None = None
    # --- ranker consensus provenance, for track-record-by-provider ---
    # Fraction of consensus rounds that picked this ticker (e.g. 2/3 ->
    # 0.667). NULL for single-round runs, which have no agreement signal.
    agreement_ratio: float | None = None
    # Comma-joined provider names (e.g. "claude,openai") whose consensus
    # round picked this ticker. NULL for single-round runs.
    voting_providers: str | None = None


class PickScenario(SQLModel, table=True):
    """One bull/base/bear scenario behind a pick's expected return.

    Stored per scenario rather than as a JSON blob on `picks` so a
    reliability check ("of the picks where you said bear was 15% likely,
    how often did the bear case actually land?") is a plain query.
    """

    __tablename__ = "pick_scenarios"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    rank: int = Field(primary_key=True)
    label: str = Field(primary_key=True)  # bull | base | bear
    ticker: str
    probability: float
    target_return_pct: float


class PickCatalyst(SQLModel, table=True):
    """An upcoming catalyst the Analyst named for a pick, kept so it can be
    graded once its date passes (discover/catalyst_grading.py)."""

    __tablename__ = "pick_catalysts"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    seq: int = Field(primary_key=True)
    event: str
    expected_date: str | None = None  # YYYY-MM-DD
    direction: str  # positive | negative | uncertain
    impact: str  # high | medium | low
    source: str


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
    dashboard_data: str | None = None  # JSON


class CandidateSnapshot(SQLModel, table=True):
    """Point-in-time fundamentals a run saw for one candidate.

    Prices can be refetched for any past date, but yfinance fundamentals are
    a live snapshot with no history — once a run is over, what the screen
    saw is gone. Storing a compact numeric copy lets a future model train
    on fundamentals without leaking today's values into past rows. Only
    names whose fundamentals were actually fetched get a row.
    """

    __tablename__ = "candidate_snapshots"

    run_id: int = Field(foreign_key="runs.id", primary_key=True, ondelete="CASCADE")
    ticker: str = Field(primary_key=True)
    data: str  # JSON {field: number}


class CandidateOutcome(SQLModel, table=True):
    """Realized forward excess return of a screened candidate: entry at the
    first close after the run, exit `horizon_days` trading days later."""

    __tablename__ = "candidate_outcomes"

    run_id: int = Field(foreign_key="runs.id", primary_key=True, ondelete="CASCADE")
    ticker: str = Field(primary_key=True)
    horizon_days: int = Field(primary_key=True)
    entry_date: str
    exit_date: str
    return_pct: float
    spy_return_pct: float
    excess_pct: float


class ModelVersion(SQLModel, table=True):
    """One trained forward-return model and its walk-forward validation.
    Only rows with accepted=1 are ever used by the screen."""

    __tablename__ = "model_versions"

    id: int | None = Field(default=None, primary_key=True)
    created_at: str
    horizon_days: int
    population: str
    features: str  # JSON list
    coefficients: str  # JSON {feature: weight}
    metrics: str  # JSON
    train_start: str
    train_end: str
    accepted: int = 0


__all__ = [
    "Run",
    "Candidate",
    "Scorecard",
    "Pick",
    "PickScenario",
    "PickCatalyst",
    "HoldingReviewRow",
    "RunOutput",
    "CandidateSnapshot",
    "CandidateOutcome",
    "ModelVersion",
]
