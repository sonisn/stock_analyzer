# Models & ORM Refactor — Design

**Date:** 2026-05-15
**Status:** Approved (pending user spec review)
**Author:** snehal.soni

## Problem

Data model definitions and persistence code are scattered across business-logic
modules:

- 13 Pydantic models live in `discover/schemas.py` (505 lines).
- 3 more Pydantic models in `discover/rebalance_schema.py`.
- More Pydantic models defined inside `discover/track_record.py`,
  `discover/premortem.py`, `discover/report_sections.py`,
  `data/transactions.py`, `reporting/html.py`.
- Frozen `@dataclass` classes live alongside business logic in
  `discover/cc_eligibility.py`, `data/options_chain.py`,
  `data/options_symbols.py`, `data/historical_volatility.py`,
  plus a private `_Quote` in `discover/track_record.py`.
- `discover/persistence.py` holds raw `sqlite3` with hand-written SQL strings,
  ALTER TABLE migrations, and CRUD helpers for 6 tables.
- `discover/track_record.py` *also* opens its own `sqlite3` connection to run
  analytics queries against the same DB.

This makes the codebase harder to read (you have to scroll past 200 lines of
model class definitions to find the agent logic in the same file), harder to
debug (no single place to look up "what shape is a `RankerPick`?"), and mixes
two distinct concerns — *what data looks like* and *how it persists*.

## Goals

1. Every data model class lives under `src/stock_analyzer/models/`, grouped by
   domain.
2. Single modeling style throughout: **Pydantic v2**. All frozen dataclasses get
   converted.
3. All SQL persistence moves to `src/stock_analyzer/db/`, using **SQLModel** as
   the ORM. Raw `sqlite3` calls in `discover/persistence.py` and
   `discover/track_record.py` are removed.
4. Existing SQLite database file and rows are preserved 1:1 — historical
   rebalance runs remain queryable through the new code.
5. Refactor lands in two independently revertible phases.

## Non-goals

- No new business logic, no new tables, no new features.
- No restructuring of which functions do what — function signatures and
  semantics stay the same; only the location and the underlying SQL mechanism
  change.
- No backward-compatibility re-export shims. Old module paths are deleted, all
  import sites updated atomically inside each phase.

## Final directory layout

```
src/stock_analyzer/
  models/
    __init__.py        # curated re-exports for convenience imports
    llm.py             # HoldingReview, Scenario, RankerPick, CorrelatedPair,
                       # RankerOutput, BearCase, RedTeamOutput, MarketTheme,
                       # MarketThemes, Allocation, SizerOutput, AnalystReport
    rebalance.py       # OptionWrite, RebalanceAction, RebalancePlan
    market.py          # OptionQuote, OptionChain, ParsedOCC, OCCParseError,
                       # RealizedVolatility
    portfolio.py       # Lot, TickerTaxSummary, EligibleHolding,
                       # RoundLotCoverage, IvHvRegime
    track_record.py    # PickReturn, DirectionStats, TrackRecord, Quote
    reports.py         # Section, TickerSection, PreMortemFailure, PreMortem
  db/
    __init__.py        # re-exports get_session and table classes
    session.py         # engine factory, get_session() contextmanager,
                       # PRAGMA foreign_keys hook, ALTER TABLE migrations
    tables.py          # 6 SQLModel table=True classes:
                       # Run, Candidate, Scorecard, Pick,
                       # HoldingReviewRow, RunOutput
    repository.py      # all CRUD repository functions, grouped by section
                       # comments: --- runs ---, --- candidates ---, etc.
    track_record.py    # read-only analytics queries currently in
                       # discover/track_record.py (joins Pick + Run for return
                       # calc)
```

Files **deleted** at the end:
- `src/stock_analyzer/discover/schemas.py`
- `src/stock_analyzer/discover/rebalance_schema.py`
- `src/stock_analyzer/discover/persistence.py`

Files where **only the class definitions are removed** (business logic stays):
- `src/stock_analyzer/discover/track_record.py` — Pydantic models +
  `_Quote` dataclass + raw sqlite queries all move out; the orchestration
  function stays and now imports from `models.track_record` and `db.track_record`.
- `src/stock_analyzer/discover/premortem.py`
- `src/stock_analyzer/discover/cc_eligibility.py`
- `src/stock_analyzer/discover/report_sections.py`
- `src/stock_analyzer/data/transactions.py`
- `src/stock_analyzer/data/options_chain.py`
- `src/stock_analyzer/data/options_symbols.py`
- `src/stock_analyzer/data/historical_volatility.py`
- `src/stock_analyzer/reporting/html.py`

## Pydantic conventions

Every model defines:

```python
from pydantic import BaseModel, ConfigDict

class Foo(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
    )
    ...
```

Rationale:
- `frozen=True` matches the immutability semantics of the existing
  `@dataclass(frozen=True)` classes; equivalent ergonomics with validation.
- `extra="forbid"` catches typos in LLM JSON output the way today's
  Pydantic models do.
- `str_strip_whitespace=True` matches the input-cleanup the existing models do
  via validators.

Validators are preserved verbatim — current `@field_validator` for OCC
normalization, score clamping, and conviction bounds copy across unchanged.

Naming change: `_Quote` in `discover/track_record.py` becomes public `Quote` in
`models/track_record.py` (it was only private by location, not by intent).

`OCCParseError` is **not** a model but a `class OCCParseError(ValueError)`
exception. It moves to `models/market.py` next to `ParsedOCC` because that's
where consumers will look for it.

## SQLModel design

### Tables (`db/tables.py`)

Six `SQLModel, table=True` classes mapping 1:1 to the current columns. Example:

```python
from typing import Optional
from sqlmodel import SQLModel, Field

class Run(SQLModel, table=True):
    __tablename__ = "runs"
    id: Optional[int] = Field(default=None, primary_key=True)
    run_at: str  # ISO timestamp; matches current TEXT column
    kind: str = Field(default="discover")
    universe_size: int
    survivors: int
    picks: int
    opus_model: Optional[str] = None
    sonnet_model: Optional[str] = None
    cash_budget: Optional[float] = None
```

Composite primary keys (e.g., `candidates(run_id, ticker)`,
`picks(run_id, rank)`, `holdings_reviews(run_id, ticker)`,
`run_outputs(run_id)`) are declared via `primary_key=True` on multiple
fields where supported, or via `__table_args__` with a `PrimaryKeyConstraint`
otherwise.

Foreign keys preserve `ON DELETE CASCADE`:

```python
run_id: int = Field(foreign_key="runs.id", ondelete="CASCADE", primary_key=True)
```

**JSON columns stay as TEXT.** The current schema stores `fail_reasons`,
`score_components`, `score_breakdown`, `sources`, and `dashboard_data` as
`json.dumps`-encoded TEXT. The SQLModel table model declares these as
`Optional[str]`; repository functions handle `json.dumps`/`json.loads` at the
boundary. Rationale: preserves on-disk format exactly, no migration needed.

### Session (`db/session.py`)

```python
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
import os
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, Session, create_engine

@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_conn, _):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.close()

@contextmanager
def get_session(db_path: str) -> Iterator[Session]:
    p = Path(os.path.expanduser(db_path))
    p.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{p}")
    SQLModel.metadata.create_all(engine)  # no-op on existing DB
    _apply_legacy_migrations(engine)       # ALTER TABLEs from old _MIGRATIONS
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
```

`_apply_legacy_migrations` preserves the existing `_MIGRATIONS` tuple from
`discover/persistence.py` verbatim (idempotent ALTER TABLEs that handle older
local DBs created before the `kind`, `rebalance_text`, `dashboard_data`
columns existed).

### Repository (`db/repository.py`)

One file holding all CRUD functions, grouped by section comment blocks:

```python
# --- runs ---
def insert_run(session: Session, *, universe_size: int, ...) -> int: ...

# --- candidates ---
def insert_candidate(session: Session, run_id: int, ticker: str, *, ...) -> None: ...

# --- scorecards ---
def insert_scorecard(session: Session, run_id: int, ticker: str, text: str) -> None: ...

# --- picks ---
def insert_pick(session: Session, run_id: int, *, rank: int, ticker: str, ...) -> None: ...

# --- holdings reviews ---
def insert_holdings_review(session: Session, run_id: int, ticker: str, *, ...) -> None: ...
def fetch_recent_holdings_history(
    session: Session, *, n_runs: int = 3, kind: str = "rebalance"
) -> dict[str, list[dict[str, Any]]]: ...

# --- run outputs ---
def insert_run_outputs(session: Session, run_id: int, *, ...) -> None: ...
```

Function signatures match the current `persistence.py` API except that the
first argument changes from `sqlite3.Connection` to `Session`. Return types and
keyword args stay identical so callsite changes are minimal (1 line per call).

JSON marshalling happens here:

```python
def insert_candidate(session: Session, run_id: int, ticker: str, *,
                     fail_reasons: list[str], ...) -> None:
    row = Candidate(
        run_id=run_id,
        ticker=ticker,
        fail_reasons=json.dumps(fail_reasons),
        ...
    )
    session.add(row)
```

### Track record queries (`db/track_record.py`)

The two functions currently in `discover/track_record.py` that take a
`sqlite3.Connection` (`_fetch_recent_picks`, `_fetch_recent_sells`) move here
and get rewritten as SQLModel `select()` queries joining `Pick`/`HoldingReviewRow`
to `Run`:

```python
def fetch_recent_picks(session: Session, *, lookback_days: int) -> list[tuple[str, str]]:
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    return list(session.exec(
        select(Run.run_at, Pick.ticker)
        .join(Pick, Pick.run_id == Run.id)
        .where(Run.run_at >= cutoff)
        .order_by(Run.run_at.asc())
    ))

def fetch_recent_sells(session: Session, *, lookback_days: int) -> list[tuple[str, str]]:
    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    return list(session.exec(
        select(Run.run_at, HoldingReviewRow.ticker)
        .join(HoldingReviewRow, HoldingReviewRow.run_id == Run.id)
        .where(
            Run.run_at >= cutoff,
            func.upper(func.coalesce(HoldingReviewRow.verdict, "")).in_(("SELL", "TRIM")),
        )
        .order_by(Run.run_at.asc())
    ))
```

The shared post-processing helper `_dedup_oldest` (which computes `age_days`
and dedupes to oldest-per-ticker) stays in `discover/track_record.py` since it
operates on already-fetched rows, not on the DB. The functions above return
2-tuples of `(run_at, ticker)`; the consumer applies `_dedup_oldest` exactly as
today.

The orchestration function in `discover/track_record.py` keeps doing what it
does today; it just calls into `db.track_record` and `models.track_record`
instead of holding the SQL and model definitions inline.

## Callsite updates

Files that import models or persistence today and need import-line updates:

| File | Current import | New import |
|---|---|---|
| `cli/discover.py` | `from ..discover.persistence import connect, insert_run, ...` | `from ..db.session import get_session` + `from ..db.repository import insert_run, ...` |
| `cli/discover.py` | `from ..discover.schemas import RankerOutput, MarketTheme, ...` | `from ..models.llm import RankerOutput, MarketTheme, ...` |
| `cli/rebalance.py` | same as above for persistence and schemas | same swap |
| `discover/track_record.py` | inline `_Quote`, inline Pydantic, inline sqlite | `from ..models.track_record import Quote, PickReturn, ...` + `from ..db.track_record import ...` + `from ..db.session import get_session` |
| `discover/analyst.py`, `ranker.py`, `redteam.py`, `sizer.py`, `reviewer.py`, `market_themes.py`, `cc_eligibility.py`, `cc_validation.py`, `cc_backfill.py`, `cc_render.py`, `rebalancer.py`, `report_sections.py` | `from .schemas import X` / `from .rebalance_schema import X` | `from ..models.llm import X` / `from ..models.rebalance import X` |
| `discover/premortem.py`, `data/transactions.py`, `data/options_chain.py`, `data/options_symbols.py`, `data/historical_volatility.py`, `reporting/html.py` | inline class definitions | `from ..models.<domain> import X` (relative path depends on package) |
| `tests/test_pipeline_wiring.py` | `from stock_analyzer.discover.persistence import ...` | `from stock_analyzer.db.session import get_session` + `from stock_analyzer.db.repository import ...` |
| `tests/test_track_record.py` | `from stock_analyzer.discover.persistence import connect` | `from stock_analyzer.db.session import get_session` |
| `tests/test_cc_schema.py`, `test_cc_validation.py`, `test_cc_backfill.py`, `test_cc_eligibility.py`, `test_schemas_ev.py`, `test_round_lot_coverage.py`, etc. | imports from `discover.schemas`, `discover.rebalance_schema`, `discover.cc_eligibility`, etc. | imports from corresponding `models.*` module |

## Phasing

### Phase 1 — Models consolidation
Single commit. Order of operations:

1. Add `pydantic` is already a dep — no `pyproject.toml` change needed.
2. Create `src/stock_analyzer/models/` package with the 6 module files.
3. Move every model class definition into the appropriate module. Convert all
   `@dataclass(frozen=True)` classes to `BaseModel` with `frozen=True` config.
   Convert `_Quote` to public `Quote`.
4. Add `__init__.py` re-exports for convenience.
5. Update every import site listed in the table above.
6. Delete `discover/schemas.py` and `discover/rebalance_schema.py`. Remove the
   moved class blocks from the other files (logic stays).
7. Run full test suite. All tests green before commit.

### Phase 2 — DB / SQLModel
Single commit. Order of operations:

1. Add `sqlmodel>=0.0.22` to `pyproject.toml` dependencies. Run `uv lock`.
2. Create `src/stock_analyzer/db/` package with the 4 module files.
3. Implement `tables.py` mapping 1:1 to current schema (verified against
   `_SCHEMA` string in old `persistence.py`).
4. Implement `session.py` with engine factory, FK pragma hook, and the
   verbatim legacy `_MIGRATIONS` tuple.
5. Port every function from `discover/persistence.py` into
   `db/repository.py`, signature-preserving except for the `Session` first arg.
6. Port the raw-sqlite queries from `discover/track_record.py` into
   `db/track_record.py` as SQLModel `select()` calls.
7. Update every callsite to use `get_session()` + repository functions.
8. Delete `discover/persistence.py`. Strip raw sqlite calls and inline SQL
   strings from `discover/track_record.py`.
9. Add `tests/test_db_roundtrip.py` (new) and `tests/test_db_back_compat.py`
   (new) — see Testing section.
10. Run full test suite against a temp DB **and** against
    `tests/fixtures/legacy_runs.db` (a copy of a real DB or a hand-crafted
    fixture with pre-migration columns). All tests green before commit.

## Testing

- Existing test suite must pass green at the end of each phase. No tests
  modified except for import-path updates.
- Phase 2 adds two new tests:
  - `tests/test_db_roundtrip.py`: opens a fresh temp SQLite, inserts one row
    into every table through the repository layer, reads each back, asserts
    field-by-field equality. Catches SQLModel-mapping bugs (column-name typos,
    wrong FK direction, JSON marshalling regressions) that existing tests
    don't exercise.
  - `tests/test_db_back_compat.py`: opens a pre-existing fixture DB shipped
    under `tests/fixtures/legacy_runs.db` (a small hand-crafted DB that
    represents both a "new" DB and a "pre-migration" DB), verifies
    `_apply_legacy_migrations` runs cleanly, and that
    `fetch_recent_holdings_history` returns the expected rows.
- No new dependency on Testcontainers or other DB infra — SQLite-only tests.

## Risk register

| Risk | Mitigation |
|---|---|
| `PRAGMA foreign_keys = ON` disabled by SQLAlchemy default | Event listener on `Engine, "connect"` re-enables it. Standard SQLAlchemy pattern, verified. |
| JSON-blob columns | Stay as TEXT in the table model. Repository functions own marshalling. On-disk format unchanged. |
| Frozen-Pydantic models break code that mutated instances | All existing dataclass classes are already `frozen=True`. The existing Pydantic models in `schemas.py` are also used as immutable LLM outputs. Grep for any `model_instance.field = ...` assignments during Phase 1 implementation; fix any that exist. |
| LLM JSON parsing (`agno`, `anthropic`) breaks if model paths change | Those libraries call `Model.model_validate_json()` on the class, which is path-agnostic. Only the import line at the agent callsite changes. |
| Composite primary keys in SQLModel | Declare via `primary_key=True` on multiple fields; SQLModel supports this directly. Fallback: `__table_args__ = (PrimaryKeyConstraint(...),)`. |
| `_MIGRATIONS` ALTER TABLEs need to run before SQLModel introspects | `metadata.create_all()` is a no-op on existing tables; legacy ALTERs run immediately after, before any session work. Order matters and is documented in `session.py`. |
| Phase-1 commit breaks Phase-2 setup | Phase 1 produces a green test suite that doesn't import sqlmodel at all. Phase 2 is purely additive on top — independently revertible by reverting one commit. |

## Dependencies

- Phase 1: no new dependencies (pydantic already at `>=2.10`).
- Phase 2: add `sqlmodel>=0.0.22` (which transitively requires SQLAlchemy ≥ 2).

## Open questions

None at design time. All scope decisions captured above.
