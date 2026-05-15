# Track-Record Analytics Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing `TrackRecord` to report per-LLM-model performance, hold/trim/sell verdict accuracy split, and Sharpe ratio per direction — surfaced in both the PDF/email report and the Opus prompt context block.

**Architecture:** Pure additive extension. `Direction` literal widens from 2 to 4 values; `DirectionStats` gains `sharpe`; `TrackRecord` gains `hold_stats`, `trim_stats`, `model_breakdown`, `overall_sharpe`; `sell_stats` narrows semantically to SELL-only. Two new SQL queries in `db/track_record.py`; one new helper (`_sharpe`) in `discover/track_record.py`; `measure_track_record` extended to score four verdict types instead of two; formatters extended to render all directions plus the model row.

**Tech Stack:** Python 3.14, Pydantic v2, SQLModel, `statistics.stdev/mean` from stdlib (no new deps), pytest with `unittest.mock` for yfinance.

**Design spec:** `docs/superpowers/specs/2026-05-15-track-record-analytics-design.md`

---

## Conventions used in this plan

- All Pydantic model changes preserve `model_config = ConfigDict(frozen=True)`.
- All new tests use `unittest.mock.patch.object(tr, "_fetch_quote", ...)` to mock yfinance — matching the existing pattern in `tests/test_track_record.py`.
- Sharpe is computed unannualized — per-decision Sharpe matching the per-decision alpha unit.
- "Run the test" steps use `uv run pytest <path> -v`.
- "Full suite" steps use `uv run pytest -q`. Expected baseline: 175 passing before any change in this plan.
- All commits are done at the END after every test passes, in one squashed commit per phase. The plan has two phases.

---

## Phase 1 — Models + DB queries + orchestration (single commit)

### Task 1: Widen `Direction` literal and extend `DirectionStats` with Sharpe

**Files:**
- Modify: `src/stock_analyzer/models/track_record.py`

- [ ] **Step 1: Read the current file**

```bash
cat src/stock_analyzer/models/track_record.py
```

Confirm the current shape: `Direction = Literal["buy", "sell"]`, `DirectionStats` has no `sharpe` field, `TrackRecord` has only `buy_stats` and `sell_stats`.

- [ ] **Step 2: Apply the model changes**

Replace the body of `src/stock_analyzer/models/track_record.py` with this content. Keep the existing module docstring at the top:

```python
"""Pydantic models for the track-record measurement pipeline.

Captures one scored decision (``PickReturn``) — buy / hold / trim / sell —
plus per-direction stats (``DirectionStats``), per-Opus-model breakdown
(``ModelStats``), and the top-level aggregate (``TrackRecord``) used by
the report header and the ranker prompt.

Sign convention for ``alpha_pct``: positive always means "the call was
right". BUY and HOLD use ``alpha = stock_ret - spy_ret`` (the holding
direction — vindicated when the stock outperforms SPY). TRIM and SELL
flip the sign — vindicated when the stock underperforms SPY after the
verdict.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Direction = Literal["buy", "hold", "trim", "sell"]


class Quote(BaseModel):
    """One yfinance price snapshot: pick-date close and measurement-date
    close (renamed from ``_Quote`` now that it's part of the public
    model surface)."""

    model_config = ConfigDict(frozen=True)

    pick_price: float | None
    measured_price: float | None


class PickReturn(BaseModel):
    """One scored decision — its realized return and how it compared to SPY.

    ``direction`` is one of buy / hold / trim / sell. ``alpha_pct`` is
    sign-adjusted so positive always means "the call was right".
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    pick_date: str          # ISO yyyy-mm-dd
    age_days: int
    direction: Direction = "buy"
    pick_price: float | None
    measured_price: float | None
    pick_return_pct: float | None
    spy_return_pct: float | None
    alpha_pct: float | None  # direction-aware: positive = right call
    is_mature: bool          # >= _MIN_AGE_DAYS old


class DirectionStats(BaseModel):
    """Aggregate stats for one direction (buy / hold / trim / sell)."""

    model_config = ConfigDict(frozen=True)

    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int
    losers: int
    flats: int
    sharpe: float | None  # None when n_mature < 5 or stdev <= 0.001


class ModelStats(BaseModel):
    """Per-Opus-model performance for BUY decisions only."""

    model_config = ConfigDict(frozen=True)

    opus_model: str
    n_mature: int
    mean_alpha_pct: float | None
    sharpe: float | None


class TrackRecord(BaseModel):
    """Aggregate summary of mature decisions over the lookback window.

    Top-level ``mean_*`` / ``winners`` / ``losers`` / ``flats`` cover ALL
    mature decisions across every direction. ``buy_stats`` / ``hold_stats`` /
    ``trim_stats`` / ``sell_stats`` break it down. ``sell_stats`` is
    SELL-only (TRIM moved to its own field); ``model_breakdown`` carries
    BUY-only per-Opus-model rows for models with n_mature >= 3.
    """

    model_config = ConfigDict(frozen=True)

    n_picks_total: int
    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int
    losers: int
    flats: int
    overall_sharpe: float | None

    buy_stats: DirectionStats
    hold_stats: DirectionStats
    trim_stats: DirectionStats
    sell_stats: DirectionStats          # SELL-only (was SELL+TRIM bundled).

    model_breakdown: list[ModelStats]

    picks: list[PickReturn]
    pending: list[PickReturn]


__all__ = [
    "Direction",
    "Quote",
    "PickReturn",
    "DirectionStats",
    "ModelStats",
    "TrackRecord",
]
```

- [ ] **Step 3: Verify import**

```bash
uv run python -c "from stock_analyzer.models.track_record import Direction, Quote, PickReturn, DirectionStats, ModelStats, TrackRecord; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 4: Run full suite — most tests will fail (expected)**

```bash
uv run pytest -q 2>&1 | tail -10
```
Expected: failures in tests that construct `TrackRecord` or `DirectionStats` without the new fields (`hold_stats`, `trim_stats`, `model_breakdown`, `overall_sharpe`, `sharpe`). Note the failure count so you can verify progress. This is OK — the failures will resolve when Task 3 updates `_empty_record` and `measure_track_record`.

Do not commit yet.

---

### Task 2: Add new DB queries in `db/track_record.py`

**Files:**
- Modify: `src/stock_analyzer/db/track_record.py`

- [ ] **Step 1: Replace the file body with the extended version**

```python
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
    # Concatenate and re-sort oldest-first by run_at so the caller sees a
    # single oldest-first stream.
    return sorted(sells + trims, key=lambda row: row[0])


__all__ = [
    "fetch_recent_pick_runs",
    "fetch_recent_pick_runs_with_model",
    "fetch_recent_verdict_runs",
    "fetch_recent_sell_runs",
]
```

- [ ] **Step 2: Verify imports**

```bash
uv run python -c "
from stock_analyzer.db.track_record import (
    fetch_recent_pick_runs,
    fetch_recent_pick_runs_with_model,
    fetch_recent_verdict_runs,
    fetch_recent_sell_runs,
)
print('ok')
"
```
Expected: prints `ok`.

- [ ] **Step 3: Smoke test new queries against a temp DB**

```bash
uv run python -c "
import tempfile, pathlib
from stock_analyzer.db.session import get_session
from stock_analyzer.db.repository import insert_run, insert_pick, insert_holdings_review
from stock_analyzer.db.track_record import (
    fetch_recent_pick_runs_with_model,
    fetch_recent_verdict_runs,
)

with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / 'q.db'
    with get_session(str(p)) as s:
        rid = insert_run(
            s, universe_size=1, survivors=1, picks=1,
            opus_model='claude-opus-4-7', sonnet_model='x',
            cash_budget=None, kind='discover',
        )
        insert_pick(s, rid, rank=1, ticker='NVDA',
                    ranker_text='p', bear_case_text=None, allocation_text=None)
        insert_holdings_review(s, rid, 'AAPL', verdict='HOLD', confidence=8, review_text='r')
        insert_holdings_review(s, rid, 'TSLA', verdict='TRIM', confidence=6, review_text='r')
        insert_holdings_review(s, rid, 'GOOG', verdict='SELL', confidence=5, review_text='r')

    with get_session(str(p)) as s:
        picks = fetch_recent_pick_runs_with_model(s, lookback_days=30)
        holds = fetch_recent_verdict_runs(s, 'HOLD', lookback_days=30)
        trims = fetch_recent_verdict_runs(s, 'TRIM', lookback_days=30)
        sells = fetch_recent_verdict_runs(s, 'SELL', lookback_days=30)

assert len(picks) == 1 and picks[0][1] == 'NVDA' and picks[0][2] == 'claude-opus-4-7'
assert len(holds) == 1 and holds[0][1] == 'AAPL'
assert len(trims) == 1 and trims[0][1] == 'TSLA'
assert len(sells) == 1 and sells[0][1] == 'GOOG'
print('queries ok')
"
```
Expected: prints `queries ok`.

Do not commit yet.

---

### Task 3: Rewrite `measure_track_record` orchestration

**Files:**
- Modify: `src/stock_analyzer/discover/track_record.py`

- [ ] **Step 1: Read the current orchestration function and helpers**

```bash
sed -n '160,310p' src/stock_analyzer/discover/track_record.py
```

Note the current shape of `measure_track_record`, `_aggregate`, and `_empty_record`. You'll replace all three.

- [ ] **Step 2: Add a `_sharpe` helper**

Add this helper near the other private helpers (right above `_aggregate` works). Imports needed at top of file: `import statistics`.

```python
def _sharpe(alphas: list[float]) -> float | None:
    """Per-decision Sharpe = mean(alpha) / stdev(alpha). Returns None when
    the sample is too small (< 5) or effectively flat (stdev < 0.001).
    Unannualized — matches the unit of the underlying alpha (per-decision
    measurement over the 90-day window)."""
    if len(alphas) < 5:
        return None
    stdev = statistics.stdev(alphas)
    if stdev < 0.001:
        return None
    return statistics.mean(alphas) / stdev
```

- [ ] **Step 3: Update imports at the top of the file**

Find the existing import block. Ensure these are present (some already exist):

```python
import statistics
from collections import defaultdict
```

Update the `from ..db.track_record import ...` line to:

```python
from ..db.track_record import (
    fetch_recent_pick_runs_with_model,
    fetch_recent_verdict_runs,
)
```

(The old `fetch_recent_pick_runs` and `fetch_recent_sell_runs` are no longer used by this module — but they're still exported by `db/track_record.py` for back-compat.)

Update the `from ..models.track_record import ...` block:

```python
from ..models.track_record import (
    Direction,
    DirectionStats,
    ModelStats,
    PickReturn,
    Quote,
    TrackRecord,
)
```

- [ ] **Step 4: Update `_aggregate` to compute Sharpe**

Replace the existing `_aggregate` body with:

```python
def _aggregate(
    mature: list[PickReturn], *, pending_count: int = 0
) -> DirectionStats:
    """Compute mean returns + win/loss/flat counts + Sharpe for a list of
    mature decisions. Alpha is already direction-aware (positive = right
    call), so we threshold ±0.5% on alpha regardless of direction."""
    if not mature:
        return DirectionStats(
            n_mature=0, n_pending=pending_count,
            mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
            winners=0, losers=0, flats=0, sharpe=None,
        )
    mean_ret = sum(p.pick_return_pct or 0 for p in mature) / len(mature)
    mean_spy = sum(p.spy_return_pct or 0 for p in mature) / len(mature)
    mean_alpha = sum(p.alpha_pct or 0 for p in mature) / len(mature)
    winners = sum(1 for p in mature if (p.alpha_pct or 0) > 0.5)
    losers = sum(1 for p in mature if (p.alpha_pct or 0) < -0.5)
    flats = len(mature) - winners - losers
    alphas = [p.alpha_pct for p in mature if p.alpha_pct is not None]
    return DirectionStats(
        n_mature=len(mature),
        n_pending=pending_count,
        mean_return_pct=mean_ret,
        mean_spy_return_pct=mean_spy,
        mean_alpha_pct=mean_alpha,
        winners=winners,
        losers=losers,
        flats=flats,
        sharpe=_sharpe(alphas),
    )
```

- [ ] **Step 5: Update `_empty_record` to include the new fields**

Replace `_empty_record`:

```python
def _empty_record() -> TrackRecord:
    empty_dir = DirectionStats(
        n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, sharpe=None,
    )
    return TrackRecord(
        n_picks_total=0, n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, overall_sharpe=None,
        buy_stats=empty_dir,
        hold_stats=empty_dir,
        trim_stats=empty_dir,
        sell_stats=empty_dir,
        model_breakdown=[],
        picks=[], pending=[],
    )
```

- [ ] **Step 6: Replace `measure_track_record` with the four-verdict variant**

Replace the entire body of `measure_track_record` with:

```python
def measure_track_record(
    db_path: str, *, lookback_days: int = 180
) -> TrackRecord:
    """Top-level entry — query buy picks AND hold/trim/sell verdicts, fetch
    forward prices, summarize per direction and per Opus model.

    `lookback_days` bounds how far back we look; defaults to 180 so the
    system has enough mature decisions to compute meaningful stats once
    it's been running a while. Empty TrackRecord is returned if the DB
    has no decisions yet.
    """
    try:
        with get_session(db_path) as session:
            pick_rows = fetch_recent_pick_runs_with_model(session, lookback_days=lookback_days)
            hold_rows = fetch_recent_verdict_runs(session, "HOLD", lookback_days=lookback_days)
            trim_rows = fetch_recent_verdict_runs(session, "TRIM", lookback_days=lookback_days)
            sell_rows = fetch_recent_verdict_runs(session, "SELL", lookback_days=lookback_days)
        # Buy rows come as 3-tuples (run_at, ticker, opus_model); strip the
        # model for the date-aware dedup step then re-attach by ticker.
        buy_pairs = [(r, t) for r, t, _m in pick_rows]
        raw_buys = _dedup_oldest(buy_pairs)
        raw_holds = _dedup_oldest(hold_rows)
        raw_trims = _dedup_oldest(trim_rows)
        raw_sells = _dedup_oldest(sell_rows)
        # Build {ticker: opus_model} from the OLDEST (first) occurrence so it
        # matches the dedup'd row. _dedup_oldest already keeps oldest-per-ticker.
        ticker_model: dict[str, str | None] = {}
        for _r, ticker, model in pick_rows:
            ticker_model.setdefault(ticker, model)
    except Exception as e:
        logger.warning("track-record fetch failed (%s) — returning empty", e)
        return _empty_record()
    if not (raw_buys or raw_holds or raw_trims or raw_sells):
        return _empty_record()

    distinct_dates = sorted({
        d for _, d, _ in raw_buys + raw_holds + raw_trims + raw_sells
    })
    spy_cache: dict[str, Quote] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for d, q in zip(distinct_dates, ex.map(_fetch_spy_quote, distinct_dates), strict=False):
            spy_cache[d] = q

    decisions: list[PickReturn] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = []
        for t, d, a in raw_buys:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "buy",
            ))
        for t, d, a in raw_holds:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "hold",
            ))
        for t, d, a in raw_trims:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "trim",
            ))
        for t, d, a in raw_sells:
            futures.append(ex.submit(
                _score_pick, t, d, a,
                spy_cache.get(d, Quote(pick_price=None, measured_price=None)), "sell",
            ))
        for fut in futures:
            try:
                decisions.append(fut.result())
            except Exception as e:
                logger.debug("decision scoring failed: %s", e)

    delisted = [p for p in decisions if p.pick_price is None]
    if delisted:
        logger.info(
            "Track record: skipped %d delisted/unmeasurable ticker(s): %s",
            len(delisted),
            ", ".join(sorted({p.ticker for p in delisted})[:10]),
        )
    decisions = [p for p in decisions if p.pick_price is not None]

    mature = [p for p in decisions if p.is_mature and p.alpha_pct is not None]
    pending = [p for p in decisions if not p.is_mature or p.alpha_pct is None]

    overall = _aggregate(mature)
    buy_stats = _aggregate(
        [p for p in mature if p.direction == "buy"],
        pending_count=sum(1 for p in pending if p.direction == "buy"),
    )
    hold_stats = _aggregate(
        [p for p in mature if p.direction == "hold"],
        pending_count=sum(1 for p in pending if p.direction == "hold"),
    )
    trim_stats = _aggregate(
        [p for p in mature if p.direction == "trim"],
        pending_count=sum(1 for p in pending if p.direction == "trim"),
    )
    sell_stats = _aggregate(
        [p for p in mature if p.direction == "sell"],
        pending_count=sum(1 for p in pending if p.direction == "sell"),
    )

    model_breakdown = _compute_model_breakdown(
        [p for p in mature if p.direction == "buy"], ticker_model,
    )

    record = TrackRecord(
        n_picks_total=len(decisions),
        n_mature=len(mature),
        n_pending=len(pending),
        mean_return_pct=overall.mean_return_pct,
        mean_spy_return_pct=overall.mean_spy_return_pct,
        mean_alpha_pct=overall.mean_alpha_pct,
        winners=overall.winners,
        losers=overall.losers,
        flats=overall.flats,
        overall_sharpe=overall.sharpe,
        buy_stats=buy_stats,
        hold_stats=hold_stats,
        trim_stats=trim_stats,
        sell_stats=sell_stats,
        model_breakdown=model_breakdown,
        picks=mature,
        pending=pending,
    )
    logger.info(
        "Track record: buys=%d hold=%d trim=%d sell=%d mature; %d pending; "
        "overall_sharpe=%s",
        buy_stats.n_mature, hold_stats.n_mature,
        trim_stats.n_mature, sell_stats.n_mature,
        record.n_pending,
        f"{record.overall_sharpe:.2f}" if record.overall_sharpe is not None else "n/a",
    )
    return record


def _compute_model_breakdown(
    buy_mature: list[PickReturn],
    ticker_model: dict[str, str | None],
) -> list[ModelStats]:
    """Group mature BUY decisions by their originating opus_model. Models
    with n_mature < 3 are dropped — their stats are too noisy to report
    individually (the decisions still appear in the overall buy aggregate).
    Picks whose opus_model is None are grouped under 'unknown'."""
    by_model: dict[str, list[PickReturn]] = defaultdict(list)
    for p in buy_mature:
        model = ticker_model.get(p.ticker) or "unknown"
        by_model[model].append(p)
    out: list[ModelStats] = []
    for model, picks in by_model.items():
        if len(picks) < 3:
            continue
        alphas = [p.alpha_pct for p in picks if p.alpha_pct is not None]
        if not alphas:
            continue
        out.append(ModelStats(
            opus_model=model,
            n_mature=len(picks),
            mean_alpha_pct=sum(alphas) / len(alphas),
            sharpe=_sharpe(alphas),
        ))
    # Sort by mean_alpha desc so the strongest model is listed first.
    return sorted(
        out, key=lambda m: m.mean_alpha_pct or 0, reverse=True,
    )
```

- [ ] **Step 7: Verify the module imports cleanly**

```bash
uv run python -c "from stock_analyzer.discover.track_record import measure_track_record, _sharpe, _compute_model_breakdown; print('ok')"
```
Expected: prints `ok`.

- [ ] **Step 8: Run the existing test suite — should be mostly green except for two known failures**

```bash
uv run pytest -q 2>&1 | tail -15
```

Most tests should pass now. Two known failure classes if any remain:
1. `tests/test_track_record_cc.py` — these tests construct `TrackRecord` in test code and will need the new fields. Address in Task 4.
2. Any test that constructs `DirectionStats` directly will need `sharpe=None`. Address in Task 4.

Count remaining failures so you can verify Task 4 closes them.

Do not commit yet.

---

### Task 4: Fix existing tests broken by the model widening

**Files:**
- Modify: `tests/test_track_record.py`
- Modify: `tests/test_track_record_cc.py`
- Other: any other test file that breaks under the new model shape

- [ ] **Step 1: Find every test that constructs `DirectionStats` or `TrackRecord`**

```bash
grep -rn "DirectionStats(\|TrackRecord(" tests src
```

Worklist: every callsite that passes positional or keyword args. The shape changed:
- `DirectionStats` gained `sharpe: float | None` — add `sharpe=None` to every constructor.
- `TrackRecord` gained `hold_stats: DirectionStats`, `trim_stats: DirectionStats`, `model_breakdown: list[ModelStats]`, `overall_sharpe: float | None` — add to every constructor.

- [ ] **Step 2: For each test that constructs `DirectionStats`, add `sharpe=None`**

Example pattern:
```python
# Before:
DirectionStats(
    n_mature=0, n_pending=0,
    mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
    winners=0, losers=0, flats=0,
)
# After:
DirectionStats(
    n_mature=0, n_pending=0,
    mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
    winners=0, losers=0, flats=0, sharpe=None,
)
```

Apply via Edit per occurrence.

- [ ] **Step 3: For each test that constructs `TrackRecord`, add the new fields**

Example pattern:
```python
# Before:
TrackRecord(
    n_picks_total=..., n_mature=..., n_pending=...,
    mean_return_pct=..., ..., winners=..., losers=..., flats=...,
    buy_stats=..., sell_stats=...,
    picks=..., pending=...,
)
# After:
TrackRecord(
    n_picks_total=..., n_mature=..., n_pending=...,
    mean_return_pct=..., ..., winners=..., losers=..., flats=...,
    overall_sharpe=None,
    buy_stats=..., hold_stats=<empty_dir>, trim_stats=<empty_dir>, sell_stats=...,
    model_breakdown=[],
    picks=..., pending=...,
)
```

Where `<empty_dir>` is `DirectionStats(n_mature=0, n_pending=0, mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None, winners=0, losers=0, flats=0, sharpe=None)`. If the test file uses a shared `_empty_dir` fixture or local helper, define one and reuse it.

If a test imports `ModelStats`, add that import — but only when the test actually needs it. Don't pre-emptively add unused imports.

- [ ] **Step 4: Run the full suite**

```bash
uv run pytest -q 2>&1 | tail -10
```
Expected: 175 passed (same baseline). All existing tests now use the new model shape.

- [ ] **Step 5: Ruff**

```bash
uv run ruff check src tests
```
Expected: `All checks passed!`. If not, run `uv run ruff check --fix src tests` and re-verify.

Do not commit yet.

---

### Task 5: New tests for Sharpe + per-model breakdown + direction sign conventions

**Files:**
- Modify: `tests/test_track_record.py`

- [ ] **Step 1: Add Sharpe threshold tests**

Append to `tests/test_track_record.py`:

```python
# --- Sharpe sample-size and zero-stdev guards ----------------------------


def test_sharpe_returns_none_below_n5():
    """Sharpe is None when the mature sample has fewer than 5 entries."""
    alphas = [1.0, 2.0, 3.0, 4.0]
    assert tr._sharpe(alphas) is None


def test_sharpe_computes_at_n5():
    """Sharpe is mean/stdev once the sample reaches 5 entries."""
    alphas = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = pytest.approx(
        sum(alphas) / 5 / 1.5811388300841898  # statistics.stdev(1..5)
    )
    assert tr._sharpe(alphas) == expected


def test_sharpe_returns_none_when_stdev_essentially_zero():
    """Sharpe is None when every alpha is identical (stdev < 0.001)."""
    alphas = [0.5, 0.5, 0.5, 0.5, 0.5]
    assert tr._sharpe(alphas) is None
```

- [ ] **Step 2: Add hold and trim sign-convention tests**

Append:

```python
# --- hold / trim alpha sign conventions ----------------------------------


def test_hold_alpha_uses_buy_sign_convention():
    """HOLD vindicated when the stock outperforms SPY — same sign as BUY."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=110.0),
    ):
        result = tr._score_pick(
            "AAPL", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 408.0),
            direction="hold",
        )
    # Stock +10%, SPY +2% → HOLD alpha = +8% (HOLD was vindicated).
    assert result.pick_return_pct == pytest.approx(10.0)
    assert result.spy_return_pct == pytest.approx(2.0)
    assert result.alpha_pct == pytest.approx(8.0)
    assert result.direction == "hold"


def test_trim_alpha_sign_flips_so_underperforming_stock_is_a_win():
    """TRIM right when the stock underperforms SPY — same sign-flip as SELL."""
    with patch.object(
        tr, "_fetch_quote",
        return_value=tr.Quote(pick_price=100.0, measured_price=88.0),
    ):
        result = tr._score_pick(
            "INTC", "2026-02-01", age_days=60,
            spy_quote=_spy_quote(400.0, 408.0),
            direction="trim",
        )
    # Stock -12%, SPY +2% → raw alpha -14% → TRIM flips → +14% (wise trim).
    assert result.pick_return_pct == pytest.approx(-12.0)
    assert result.spy_return_pct == pytest.approx(2.0)
    assert result.alpha_pct == pytest.approx(14.0)
    assert result.direction == "trim"
```

- [ ] **Step 3: Add per-model breakdown tests**

Append:

```python
# --- model breakdown -----------------------------------------------------


def test_compute_model_breakdown_drops_models_below_n3():
    """Models with fewer than 3 mature decisions are dropped (still
    counted in the overall buy aggregate, just not surfaced as a row)."""
    picks = [
        tr.PickReturn(
            ticker=f"T{i}", pick_date="2026-02-01", age_days=60,
            direction="buy",
            pick_price=100.0, measured_price=110.0,
            pick_return_pct=10.0, spy_return_pct=2.0, alpha_pct=8.0,
            is_mature=True,
        )
        for i in range(4)
    ]
    # 3 picks on opus-4-7, 1 on opus-4-6.
    ticker_model = {"T0": "opus-4-7", "T1": "opus-4-7", "T2": "opus-4-7", "T3": "opus-4-6"}
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 1
    assert out[0].opus_model == "opus-4-7"
    assert out[0].n_mature == 3
    assert out[0].mean_alpha_pct == pytest.approx(8.0)


def test_compute_model_breakdown_groups_none_as_unknown():
    """Picks whose opus_model is None bucket under 'unknown'."""
    picks = [
        tr.PickReturn(
            ticker=f"X{i}", pick_date="2026-02-01", age_days=60,
            direction="buy",
            pick_price=100.0, measured_price=104.0,
            pick_return_pct=4.0, spy_return_pct=2.0, alpha_pct=2.0,
            is_mature=True,
        )
        for i in range(3)
    ]
    ticker_model = {"X0": None, "X1": None, "X2": None}
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 1
    assert out[0].opus_model == "unknown"
    assert out[0].n_mature == 3


def test_compute_model_breakdown_sorted_by_mean_alpha_desc():
    """Strongest model listed first."""
    picks = [
        tr.PickReturn(
            ticker=f"A{i}", pick_date="2026-02-01", age_days=60,
            direction="buy", pick_price=100.0, measured_price=120.0,
            pick_return_pct=20.0, spy_return_pct=2.0, alpha_pct=18.0,
            is_mature=True,
        ) for i in range(3)
    ] + [
        tr.PickReturn(
            ticker=f"B{i}", pick_date="2026-02-01", age_days=60,
            direction="buy", pick_price=100.0, measured_price=104.0,
            pick_return_pct=4.0, spy_return_pct=2.0, alpha_pct=2.0,
            is_mature=True,
        ) for i in range(3)
    ]
    ticker_model = {f"A{i}": "weak" for i in range(3)}
    ticker_model.update({f"B{i}": "strong" for i in range(3)})
    # Intentionally mis-labeled to make sure sorting is by alpha not name.
    # A* picks have +18% alpha but are labeled "weak"; B* picks +2% labeled "strong".
    # After sorting, the "weak" model should appear FIRST (alpha 18 > alpha 2).
    out = tr._compute_model_breakdown(picks, ticker_model)
    assert len(out) == 2
    assert out[0].opus_model == "weak"
    assert out[0].mean_alpha_pct == pytest.approx(18.0)
    assert out[1].opus_model == "strong"
    assert out[1].mean_alpha_pct == pytest.approx(2.0)
```

- [ ] **Step 4: Run the new tests**

```bash
uv run pytest tests/test_track_record.py -v 2>&1 | tail -20
```
Expected: all new tests PASS (8 of them). Existing tests still pass.

- [ ] **Step 5: Full suite + ruff**

```bash
uv run pytest -q && uv run ruff check src tests
```
Expected: 183 passed (175 baseline + 8 new), ruff clean.

- [ ] **Step 6: Commit Phase 1**

```bash
git add -A
git status
```

Inspect: expect changes in `models/track_record.py`, `db/track_record.py`, `discover/track_record.py`, `tests/test_track_record.py`, and any other tests modified in Task 4.

```bash
git commit -m "$(cat <<'EOF'
feat(cc): per-model + hold/trim/sell + Sharpe analytics in TrackRecord

Phase 1 of the track-record analytics extension (design:
docs/superpowers/specs/2026-05-15-track-record-analytics-design.md).

- Direction literal widens: buy | sell -> buy | hold | trim | sell.
- DirectionStats gains sharpe (None when n_mature<5 or stdev<0.001).
- TrackRecord gains hold_stats, trim_stats, model_breakdown,
  overall_sharpe. sell_stats narrows to SELL-only (TRIM moved to
  trim_stats); the old SELL+TRIM bundling now lives in
  fetch_recent_sell_runs as a back-compat wrapper around the new
  per-verdict query.
- New ModelStats Pydantic model — opus_model, n_mature, mean_alpha,
  sharpe. BUY-only, dropped below n_mature<3 to filter noise.
- New db queries: fetch_recent_pick_runs_with_model,
  fetch_recent_verdict_runs(verdict).
- measure_track_record fetches all four verdict streams, scores each
  with the right alpha sign convention (buy/hold use stock-SPY;
  trim/sell use SPY-stock so positive=right-call), aggregates per
  direction + per model.
- 8 new tests covering Sharpe thresholds, hold/trim sign conventions,
  model breakdown grouping/dropping/sorting.

Rendering changes (Phase 2) land in a follow-up commit.
EOF
)"
```

Verify with `git log --oneline -3`. Working tree should be clean.

---

## Phase 2 — Rendering: prompt block + PDF/email section (single commit)

### Task 6: Extend `format_track_record_summary` and `format_track_record_block`

**Files:**
- Modify: `src/stock_analyzer/discover/track_record.py` (formatters at the bottom of the file)

- [ ] **Step 1: Read the existing formatters**

```bash
sed -n '310,410p' src/stock_analyzer/discover/track_record.py
```

You'll replace `format_track_record_summary`, `_format_decision_line`,
`format_track_record_lines`, and `format_track_record_block`.

- [ ] **Step 2: Replace the formatter section**

Replace the entire formatter section (from `# --- formatters` down to the end of `format_track_record_block`) with:

```python
# --- formatters -----------------------------------------------------------


def _sharpe_text(sharpe: float | None, n_mature: int) -> str:
    """Render Sharpe as either '0.42' or 'n/a (n<5)' / 'n/a (flat)'."""
    if sharpe is not None:
        return f"{sharpe:.2f}"
    if n_mature < 5:
        return "n/a (n<5)"
    return "n/a (flat)"


def format_track_record_summary(record: TrackRecord) -> str:
    """One-line summary suitable for the dashboard / short prompt context.
    Renders the OVERALL aggregate plus per-direction sub-totals when each
    direction has at least one mature decision."""
    if record.n_mature == 0:
        if record.n_pending:
            return (
                f"Track record: 0 mature decisions yet ({record.n_pending} "
                f"too young to score; min age {_MIN_AGE_DAYS}d)."
            )
        return "Track record: no prior decisions in the lookback window."

    parts: list[str] = []
    parts.append(f"Track record (last ~{_MEASUREMENT_WINDOW_DAYS}d):")
    if record.buy_stats.n_mature:
        bs = record.buy_stats
        parts.append(
            f" Buy {bs.n_mature} mature, "
            f"alpha {bs.mean_alpha_pct:+.1f}% ({bs.winners}W/{bs.losers}L/{bs.flats}F)."
        )
    if record.hold_stats.n_mature:
        hs = record.hold_stats
        parts.append(
            f" Hold {hs.n_mature} mature, "
            f"alpha {hs.mean_alpha_pct:+.1f}% ({hs.winners}W/{hs.losers}L/{hs.flats}F)."
        )
    if record.trim_stats.n_mature:
        ts = record.trim_stats
        parts.append(
            f" Trim {ts.n_mature} mature, "
            f"alpha {ts.mean_alpha_pct:+.1f}% ({ts.winners}W/{ts.losers}L/{ts.flats}F)."
        )
    if record.sell_stats.n_mature:
        ss = record.sell_stats
        parts.append(
            f" Sell {ss.n_mature} mature, "
            f"alpha {ss.mean_alpha_pct:+.1f}% ({ss.winners}W/{ss.losers}L/{ss.flats}F)."
        )
    if record.n_pending:
        parts.append(f" {record.n_pending} pending.")
    return "".join(parts)


def _format_decision_line(p: PickReturn) -> str:
    """Render one mature decision; non-buy decisions get a [VERDICT] tag
    so the user can tell directions apart in the listing."""
    tag = {
        "buy": "[BUY] ",
        "hold": "[HOLD]",
        "trim": "[TRIM]",
        "sell": "[SELL]",
    }.get(p.direction, "[?]   ")
    return (
        f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  "
        f"return {p.pick_return_pct:+.1f}%  "
        f"SPY {p.spy_return_pct:+.1f}%  "
        f"alpha {p.alpha_pct:+.1f}%"
    )


def format_track_record_lines(record: TrackRecord, *, limit: int = 15) -> list[str]:
    """Per-decision lines for the report body. One section per direction
    with mature data; each section shows the top decisions by alpha."""
    lines: list[str] = []
    by_dir: dict[str, list[PickReturn]] = {
        "buy": [], "hold": [], "trim": [], "sell": [],
    }
    for p in record.picks:
        by_dir.setdefault(p.direction, []).append(p)

    section_headers = [
        ("buy", "  -- BUY picks (mature) --"),
        ("hold", "  -- HOLD verdicts (mature) --"),
        ("trim", "  -- TRIM verdicts (mature) --"),
        ("sell", "  -- SELL verdicts (mature) --"),
    ]
    for direction, header in section_headers:
        picks = sorted(
            by_dir[direction], key=lambda p: p.alpha_pct or 0, reverse=True,
        )
        if not picks:
            continue
        lines.append(header)
        for p in picks[:limit]:
            lines.append(_format_decision_line(p))

    if record.pending:
        lines.append("  -- pending (too young to score) --")
        for p in sorted(record.pending, key=lambda p: p.pick_date, reverse=True)[:5]:
            tag = {
                "buy": "[BUY] ",
                "hold": "[HOLD]",
                "trim": "[TRIM]",
                "sell": "[SELL]",
            }.get(p.direction, "[?]   ")
            live_ret = (
                f"live {p.pick_return_pct:+.1f}%"
                if p.pick_return_pct is not None else "—"
            )
            lines.append(
                f"  {tag} {p.ticker:6s}  {p.pick_date}  age {p.age_days}d  {live_ret}"
            )
    return lines


def _format_direction_block_line(
    label: str, stats: DirectionStats, *, first_sharpe_label_done: list[bool],
) -> str:
    """One line per direction in the multi-line block — sample size, alpha,
    Sharpe. `first_sharpe_label_done` is a one-element list used as a
    mutable flag: the FIRST direction line spells out "Sharpe (per-decision)"
    to communicate the unit, subsequent lines just say "Sharpe"."""
    label_text = "Sharpe (per-decision)" if not first_sharpe_label_done[0] else "Sharpe"
    first_sharpe_label_done[0] = True
    alpha_text = (
        f"{stats.mean_alpha_pct:+.1f}%"
        if stats.mean_alpha_pct is not None else "n/a"
    )
    return (
        f"{label} track record: {stats.n_mature} mature, "
        f"alpha {alpha_text}, "
        f"{label_text} {_sharpe_text(stats.sharpe, stats.n_mature)}"
    )


def format_track_record_block(record: TrackRecord) -> str:
    """Multi-line block suitable for prepending to the ranker / rebalancer
    prompt as historical context. Direction lines emitted for every
    direction with at least one mature decision; model_breakdown rendered
    on the last line when non-empty."""
    if record.n_picks_total == 0:
        return ""
    lines: list[str] = []
    first_sharpe_label_done = [False]
    for label, stats in [
        ("Buy", record.buy_stats),
        ("Hold", record.hold_stats),
        ("Trim", record.trim_stats),
        ("Sell", record.sell_stats),
    ]:
        if stats.n_mature:
            lines.append(_format_direction_block_line(
                label, stats, first_sharpe_label_done=first_sharpe_label_done,
            ))
    if record.model_breakdown:
        model_parts = [
            f"{m.opus_model} ({m.n_mature} picks, {m.mean_alpha_pct:+.1f}%)"
            for m in record.model_breakdown
        ]
        lines.append("Model breakdown: " + " | ".join(model_parts))
    head = "\n".join(lines) if lines else format_track_record_summary(record)
    body = format_track_record_lines(record, limit=10)
    return head + "\n" + "\n".join(body) if body else head
```

- [ ] **Step 3: Verify imports + smoke-test the rendering**

```bash
uv run python -c "
from stock_analyzer.discover.track_record import (
    format_track_record_summary,
    format_track_record_block,
    _sharpe_text,
)
from stock_analyzer.models.track_record import TrackRecord, DirectionStats, ModelStats, PickReturn

empty_dir = DirectionStats(
    n_mature=0, n_pending=0,
    mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
    winners=0, losers=0, flats=0, sharpe=None,
)
buy = DirectionStats(
    n_mature=6, n_pending=0,
    mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
    winners=4, losers=1, flats=1, sharpe=0.42,
)
rec = TrackRecord(
    n_picks_total=6, n_mature=6, n_pending=0,
    mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
    winners=4, losers=1, flats=1, overall_sharpe=0.42,
    buy_stats=buy, hold_stats=empty_dir, trim_stats=empty_dir, sell_stats=empty_dir,
    model_breakdown=[
        ModelStats(opus_model='claude-opus-4-7', n_mature=4, mean_alpha_pct=10.0, sharpe=0.55),
    ],
    picks=[], pending=[],
)
print(format_track_record_block(rec))
"
```
Expected output (roughly):
```
Buy track record: 6 mature, alpha +8.0%, Sharpe (per-decision) 0.42
Model breakdown: claude-opus-4-7 (4 picks, +10.0%)
```

- [ ] **Step 4: Smoke-test the empty case**

```bash
uv run python -c "
from stock_analyzer.discover.track_record import format_track_record_block
from stock_analyzer.models.track_record import TrackRecord, DirectionStats

empty_dir = DirectionStats(
    n_mature=0, n_pending=0,
    mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
    winners=0, losers=0, flats=0, sharpe=None,
)
rec = TrackRecord(
    n_picks_total=0, n_mature=0, n_pending=0,
    mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
    winners=0, losers=0, flats=0, overall_sharpe=None,
    buy_stats=empty_dir, hold_stats=empty_dir, trim_stats=empty_dir, sell_stats=empty_dir,
    model_breakdown=[],
    picks=[], pending=[],
)
result = format_track_record_block(rec)
assert result == '', f'expected empty, got: {result!r}'
print('empty case ok')
"
```
Expected: `empty case ok`.

Do not commit yet — tests for the formatters land in Task 7.

---

### Task 7: Tests for the rendering layer

**Files:**
- Modify: `tests/test_track_record.py` (append new tests at the bottom)

- [ ] **Step 1: Add a shared `_empty_dir` helper near the top of the test file**

Find the existing `_spy_quote` helper (around line 24). Just below it, add:

```python
def _empty_dir() -> tr.DirectionStats:
    return tr.DirectionStats(
        n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, sharpe=None,
    )
```

If the file already has an equivalent helper (added in Task 4), skip this step.

- [ ] **Step 2: Add rendering tests**

Append:

```python
# --- format_track_record_block rendering ---------------------------------


def test_block_renders_all_directions_with_data():
    """Every direction with n_mature >= 1 renders one line; model_breakdown
    renders when non-empty; first Sharpe label is spelled out."""
    buy = tr.DirectionStats(
        n_mature=6, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=4, losers=1, flats=1, sharpe=0.42,
    )
    hold = tr.DirectionStats(
        n_mature=3, n_pending=0,
        mean_return_pct=4.0, mean_spy_return_pct=3.0, mean_alpha_pct=1.0,
        winners=2, losers=1, flats=0, sharpe=None,
    )
    rec = tr.TrackRecord(
        n_picks_total=9, n_mature=9, n_pending=0,
        mean_return_pct=7.0, mean_spy_return_pct=2.5, mean_alpha_pct=4.5,
        winners=6, losers=2, flats=1, overall_sharpe=0.30,
        buy_stats=buy, hold_stats=hold, trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[
            tr.ModelStats(opus_model="claude-opus-4-7", n_mature=4, mean_alpha_pct=10.0, sharpe=0.55),
        ],
        picks=[], pending=[],
    )
    out = tr.format_track_record_block(rec)
    assert "Buy track record:" in out
    assert "Hold track record:" in out
    assert "Trim track record:" not in out
    assert "Sell track record:" not in out
    assert "Model breakdown:" in out
    assert "claude-opus-4-7 (4 picks, +10.0%)" in out
    # First Sharpe label is the full one; subsequent lines just say Sharpe.
    assert "Sharpe (per-decision)" in out


def test_block_omits_directions_with_zero_mature():
    """Directions with n_mature == 0 are completely absent from the block."""
    buy = tr.DirectionStats(
        n_mature=3, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=2, losers=0, flats=1, sharpe=None,
    )
    rec = tr.TrackRecord(
        n_picks_total=3, n_mature=3, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=2, losers=0, flats=1, overall_sharpe=None,
        buy_stats=buy, hold_stats=_empty_dir(), trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[],
        picks=[], pending=[],
    )
    out = tr.format_track_record_block(rec)
    assert "Buy track record:" in out
    assert "Hold track record:" not in out
    assert "Trim track record:" not in out
    assert "Sell track record:" not in out
    assert "Model breakdown:" not in out


def test_block_renders_sharpe_na_when_none():
    """Sharpe None renders as either 'n/a (n<5)' or 'n/a (flat)'."""
    small_sample = tr.DirectionStats(
        n_mature=3, n_pending=0,
        mean_return_pct=10.0, mean_spy_return_pct=2.0, mean_alpha_pct=8.0,
        winners=2, losers=0, flats=1, sharpe=None,
    )
    flat_sample = tr.DirectionStats(
        n_mature=5, n_pending=0,
        mean_return_pct=5.0, mean_spy_return_pct=2.0, mean_alpha_pct=3.0,
        winners=5, losers=0, flats=0, sharpe=None,
    )
    rec = tr.TrackRecord(
        n_picks_total=8, n_mature=8, n_pending=0,
        mean_return_pct=7.0, mean_spy_return_pct=2.0, mean_alpha_pct=5.0,
        winners=7, losers=0, flats=1, overall_sharpe=None,
        buy_stats=small_sample, hold_stats=flat_sample,
        trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[], picks=[], pending=[],
    )
    out = tr.format_track_record_block(rec)
    assert "n/a (n<5)" in out
    assert "n/a (flat)" in out


def test_block_returns_empty_when_no_decisions():
    """Empty record renders as empty string (prompt-context gets nothing)."""
    rec = tr.TrackRecord(
        n_picks_total=0, n_mature=0, n_pending=0,
        mean_return_pct=None, mean_spy_return_pct=None, mean_alpha_pct=None,
        winners=0, losers=0, flats=0, overall_sharpe=None,
        buy_stats=_empty_dir(), hold_stats=_empty_dir(),
        trim_stats=_empty_dir(), sell_stats=_empty_dir(),
        model_breakdown=[], picks=[], pending=[],
    )
    assert tr.format_track_record_block(rec) == ""
```

- [ ] **Step 3: Run the new tests**

```bash
uv run pytest tests/test_track_record.py -v 2>&1 | tail -25
```
Expected: 4 new tests PASS. Full file passes.

- [ ] **Step 4: Full suite + ruff**

```bash
uv run pytest -q && uv run ruff check src tests
```
Expected: 187 passed (175 baseline + 8 from Phase 1 + 4 from this task), ruff clean.

- [ ] **Step 5: Commit Phase 2**

```bash
git add -A
git status
```

Inspect: changes should be limited to `discover/track_record.py` (formatters) and `tests/test_track_record.py` (4 new tests + the optional `_empty_dir` helper).

```bash
git commit -m "$(cat <<'EOF'
feat(cc): render per-direction track-record block with Sharpe + model row

Phase 2 of the track-record analytics extension (Phase 1 commit added
the data + aggregation in feat: ... TrackRecord).

- format_track_record_block now emits one line per direction with
  n_mature >= 1; each line carries alpha and per-decision Sharpe
  (n/a hints below n=5 or for flat distributions).
- The first non-n/a Sharpe value spells out "Sharpe (per-decision)" so
  the user knows the unit; subsequent lines abbreviate.
- Model breakdown rendered as a single trailing line when non-empty:
  'Model breakdown: opus-4-7 (12 picks, +7.2%) | opus-4-6 (8 picks, +4.8%)'.
- format_track_record_summary now reports all four directions in one
  line when present (was bundled into "Sell" before).
- format_track_record_lines splits the per-decision listing into four
  sections (BUY / HOLD / TRIM / SELL).
- _format_decision_line learns tags for HOLD and TRIM in addition to
  the existing BUY / SELL tags.

Surfaces these in both the PDF/email report (via build_sections
emitting track_record_block as a 'preformatted' section) and the Opus
ranker/rebalancer prompts (via format_track_record_block on the
prompt's historical-context block).

4 new tests covering: all-directions rendering with model breakdown,
zero-mature-direction omission, Sharpe n/a hint text, empty-record
empty-string return.
EOF
)"
```

Verify with `git log --oneline -3`. Working tree clean.

---

## Self-Review

**Spec coverage:** Walked through every requirement in
`docs/superpowers/specs/2026-05-15-track-record-analytics-design.md`:

| Spec requirement | Plan task |
|---|---|
| Widen `Direction` literal to 4 values | Task 1 |
| `DirectionStats.sharpe` field | Task 1 |
| `ModelStats` model | Task 1 |
| `TrackRecord.{hold_stats,trim_stats,model_breakdown,overall_sharpe}` | Task 1 |
| `sell_stats` narrows to SELL-only | Task 1 (model docstring) + Task 3 (orchestration) |
| `fetch_recent_pick_runs_with_model` | Task 2 |
| `fetch_recent_verdict_runs(verdict)` | Task 2 |
| `fetch_recent_sell_runs` becomes back-compat wrapper | Task 2 |
| `_sharpe` helper with n>=5 + stdev>0.001 guards | Task 3 |
| `measure_track_record` fetches all 4 verdict streams | Task 3 |
| HOLD/BUY use `stock-spy`, TRIM/SELL use `spy-stock` | Task 3 (passes the correct `direction` to `_score_pick`; the existing sign-flip logic in `_score_pick` keys off `direction != "buy"` — verified in Task 3.6 by the test added in Task 5.2 which asserts HOLD uses buy convention) |
| `_compute_model_breakdown` drops models below n=3 and groups None as "unknown" | Task 3 + Task 5 |
| `_empty_record` updated for new fields | Task 3 |
| Existing tests updated for new model shape | Task 4 |
| New tests: per-model grouping, Sharpe thresholds, hold/trim sign | Task 5 |
| `format_track_record_block` renders 4 directions + model row | Task 6 |
| First Sharpe label spelled "Sharpe (per-decision)" | Task 6 |
| n/a hints `(n<5)` and `(flat)` | Task 6 |
| New rendering tests | Task 7 |

**HOLD alpha sign-convention edge case** — the spec says HOLD uses `stock_ret - spy_ret` (same as BUY). The existing `_score_pick` code branches on `direction == "buy"` to decide whether to sign-flip. For HOLD to use the buy convention, `_score_pick` needs to treat HOLD the same as BUY. Currently the code reads `alpha = raw_alpha if direction == "buy" else -raw_alpha` — so HOLD would get sign-flipped, which is WRONG per the spec.

**Plan fix:** Task 3 must also update `_score_pick` to treat HOLD as a "hold-direction" (no flip) and TRIM as "sell-direction" (flip). Adding a corrective step now.

### Task 3 addendum — fix `_score_pick` sign-flip branch

**Append to Task 3 between Step 6 and Step 7:**

- [ ] **Step 6.5: Update the sign-flip in `_score_pick`**

Find this line in `discover/track_record.py` (around line 145):

```python
alpha = raw_alpha if direction == "buy" else -raw_alpha
```

Replace with:

```python
# BUY and HOLD: vindicated when the stock outperforms SPY (positive raw_alpha).
# TRIM and SELL: vindicated when the stock underperforms SPY — sign-flip so
# positive always means "the call was right" across all four directions.
alpha = raw_alpha if direction in ("buy", "hold") else -raw_alpha
```

Run the existing `test_buy_alpha_is_stock_minus_spy_when_stock_beats_spy` test to confirm BUY isn't broken:

```bash
uv run pytest tests/test_track_record.py::test_buy_alpha_is_stock_minus_spy_when_stock_beats_spy -v
```
Expected: PASS.

The new HOLD/TRIM tests added in Task 5 verify the four-direction branch.

**Placeholder scan:** No "TBD" / "TODO" / "fill in" / "similar to" appears in this plan. Every code block contains the actual code an engineer would paste.

**Type consistency:**
- `Direction` literal: `"buy" | "hold" | "trim" | "sell"` — used in Task 1 (model), Task 3 (`_score_pick` calls), Task 5 (test assertions), Task 6 (`_format_decision_line` tag map). Consistent.
- `DirectionStats.sharpe: float | None` — Task 1 declaration, Task 3 (`_aggregate` populates), Task 4 (existing-test constructor fix), Task 5 (`_sharpe` returns), Task 6 (`_sharpe_text` reads), Task 7 (test fixtures). Consistent.
- `ModelStats.{opus_model, n_mature, mean_alpha_pct, sharpe}` — Task 1 declaration, Task 3 (`_compute_model_breakdown` builds), Task 5 (test asserts), Task 6 (block formatter reads). Consistent.
- `fetch_recent_verdict_runs(session, verdict, *, lookback_days)` — Task 2 declaration, Task 3 (4 callsites in `measure_track_record`). Consistent.
- `fetch_recent_pick_runs_with_model` returns `list[tuple[str, str, str | None]]` — Task 2 declaration, Task 3 unpacks as `(r, t, m)` and builds `ticker_model: dict[str, str | None]` — consistent.

Plan is internally consistent and ready to execute.
