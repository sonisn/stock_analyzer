# Track-Record Analytics Extension — Design

**Date:** 2026-05-15
**Status:** Approved (pending spec review)
**Author:** snehal.soni

## Problem

The existing `TrackRecord` reports two aggregate stats — buy track record and a
combined SELL+TRIM "sell" track record — plus per-decision detail. That summary
is consumed by:

1. The PDF / email report header.
2. The Opus ranker and rebalancer prompts as historical context.

Three useful signals are missing from both surfaces:

1. **Per-LLM-model attribution.** Picks come from different Opus versions over
   time (`runs.opus_model` is stored but never analyzed). Are newer Opus
   versions producing higher-alpha picks, justifying their higher per-token
   cost?
2. **Hold and trim accuracy.** TRIM is bundled with SELL in `sell_stats`. HOLD
   verdicts (the default conservative call) are not tracked at all. Together
   that hides "am I being over-conservative?" from the user.
3. **Risk-adjusted return.** Mean alpha alone doesn't tell the user whether
   the picks are noisy or consistent. Sharpe (mean_alpha / stdev_alpha)
   collapses both signals into one number.

All three are computable from data already captured in the DB; no schema
change is required.

## Goals

1. Extend `TrackRecord` to carry per-direction stats for all four verdicts
   (`buy`, `hold`, `trim`, `sell`) plus a per-model breakdown plus Sharpe per
   direction and overall.
2. Surface the new stats in both the PDF/email report and the Opus
   ranker/rebalancer prompt.
3. Preserve the existing `measure_track_record` orchestration shape — extend
   it, don't rewrite it.
4. Apply sample-size guards so noisy values render as `n/a` rather than
   misleading numbers.

## Non-goals

- No new DB columns or tables.
- No per-run cost tracking (deferred to a future spec — requires agno hook
  instrumentation).
- No sector/theme attribution (deferred to a future spec — requires either
  JSON parsing of `dashboard_data` or a new table).
- No portfolio-level drawdown or win/loss-streak tracking (deferred — Sharpe
  is sufficient as the single risk-adjusted signal for now).
- No streamlit dashboard updates (no current code path reads `sell_stats`
  outside `discover/track_record.py`; in-scope only if the dashboard read
  surface changes during implementation).

## Architecture

### Data layer — `src/stock_analyzer/db/track_record.py`

Add two new queries alongside the existing `fetch_recent_pick_runs` and
`fetch_recent_sell_runs`:

```python
def fetch_recent_pick_runs_with_model(
    session: Session, *, lookback_days: int
) -> list[tuple[str, str, str | None]]:
    """(run_at, ticker, opus_model) for BUY picks in the lookback window,
    oldest-first. opus_model may be None for older runs that did not
    record it."""

def fetch_recent_verdict_runs(
    session: Session,
    verdict: Literal["SELL", "TRIM", "HOLD"],
    *,
    lookback_days: int,
) -> list[tuple[str, str]]:
    """(run_at, ticker) for the given verdict in the lookback window,
    oldest-first. Replaces the SELL+TRIM bundling in fetch_recent_sell_runs."""
```

Keep `fetch_recent_sell_runs` as a thin wrapper around
`fetch_recent_verdict_runs("SELL", ...)` plus
`fetch_recent_verdict_runs("TRIM", ...)` so external callers (if any are
added later) can still ask for the bundled set. The orchestration in
`discover/track_record.py` calls `fetch_recent_verdict_runs` per verdict
directly.

### Models — `src/stock_analyzer/models/track_record.py`

```python
Direction = Literal["buy", "hold", "trim", "sell"]


class ModelStats(BaseModel):
    """Per-Opus-model performance for BUY decisions."""

    model_config = ConfigDict(frozen=True)

    opus_model: str
    n_mature: int
    mean_alpha_pct: float | None
    sharpe: float | None


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
    sharpe: float | None  # NEW — None when n_mature < 5 or stdev ≈ 0


class TrackRecord(BaseModel):
    """Aggregate summary of mature decisions over the lookback window."""

    model_config = ConfigDict(frozen=True)

    # Top-level aggregates over ALL directions (preserved for back-compat).
    n_picks_total: int
    n_mature: int
    n_pending: int
    mean_return_pct: float | None
    mean_spy_return_pct: float | None
    mean_alpha_pct: float | None
    winners: int
    losers: int
    flats: int
    overall_sharpe: float | None  # NEW

    # Per-direction breakdown — sell_stats narrows to SELL-only.
    buy_stats: DirectionStats
    hold_stats: DirectionStats   # NEW
    trim_stats: DirectionStats   # NEW (was bundled into sell_stats)
    sell_stats: DirectionStats   # SEMANTIC CHANGE — now SELL-only.

    model_breakdown: list[ModelStats]  # NEW — BUY decisions grouped by opus_model

    picks: list[PickReturn]
    pending: list[PickReturn]
```

### Orchestration — `discover/track_record.py::measure_track_record`

The existing flow:
1. `_dedup_oldest` on buy + sell rows
2. SPY benchmark fetch per distinct decision date
3. `_score_pick` per decision (parallel ThreadPoolExecutor)
4. Aggregate via `_aggregate` per direction
5. Return `TrackRecord(...)`

Extend to:
1. Fetch buy rows (with opus_model), hold rows, trim rows, sell rows separately.
2. SPY benchmark fetch unchanged (now covers the union of all four
   decision-date sets).
3. `_score_pick` per decision, with the `direction` value reflecting the
   verdict source. Alpha sign convention:
   - `buy`: `stock_ret - spy_ret`
   - `hold`: `stock_ret - spy_ret` (same as buy — positive when the HOLD was
     vindicated by the stock outperforming)
   - `trim`: `spy_ret - stock_ret` (sign-flipped — positive when the TRIM
     was right because the stock underperformed)
   - `sell`: `spy_ret - stock_ret` (sign-flipped — same logic as trim)
4. Aggregate into four `DirectionStats` plus the overall row.
5. Compute Sharpe per direction and overall via a new helper:
   ```python
   def _sharpe(alphas: list[float]) -> float | None:
       if len(alphas) < 5:
           return None
       mean = statistics.mean(alphas)
       stdev = statistics.stdev(alphas)
       if stdev < 0.001:
           return None
       return mean / stdev
   ```
   (Unannualized — per-decision Sharpe matching the per-decision alpha unit.)
6. Compute `model_breakdown` by grouping buy decisions by `opus_model`,
   dropping models with `n_mature < 3` (those rows are too noisy to report
   individually; their decisions still appear in the overall buy aggregate).

### Rendering

#### `format_track_record_block` (Opus prompt + report header)

Current output:
```
Buy track record: 23 mature, mean +6.4% vs SPY +2.1% — Sell track record: 8 mature, alpha +3.1%, 5W/3L
```

New output (block form for readability — both report and prompt):
```
Buy track record: 23 mature, mean +6.4% vs SPY +2.1%, Sharpe (per-decision) 0.41
Hold track record: 14 mature, mean +2.1% vs SPY +1.8%, Sharpe 0.18
Trim track record: 5 mature, alpha +4.2%, Sharpe 0.55
Sell track record: 3 mature, alpha +1.8%, Sharpe n/a (n<5)
Model breakdown: opus-4-7 (12 picks, +7.2%) | opus-4-6 (8 picks, +4.8%)
```

`Sharpe (per-decision)` is spelled out once on the first non-`n/a` line in
each block to communicate the unit; subsequent lines just say `Sharpe`.
`n/a (n<5)` is the only "below threshold" hint string; flat-distribution
cases use `n/a (flat)` to distinguish them.

When a direction has zero mature decisions, the line is omitted entirely
(today's "Sell track record: 0 mature ..." line never renders — preserve that).

When `model_breakdown` is empty (all picks below the n=3 threshold), the line
is omitted.

`format_track_record_summary` (the single-line variant used in the dashboard /
short prompt context) is left unchanged for back-compat — it still emits the
overall + buy + bundled-sell summary. The detailed per-direction view is in
`format_track_record_block`.

Wait — `sell_stats` is now SELL-only. The single-line summary must reflect
that, OR keep computing a "bundled sell" for the summary only. Decision: the
summary line uses the OVERALL stats (`mean_alpha_pct`, `winners`, `losers`,
`flats`) which are unchanged. No SELL-specific number appears in the summary.

#### PDF/email — `discover/report_sections.py`

The `track_record` section kind already renders to a styled block. Update its
renderer to emit one row per non-empty direction with sample size + alpha +
Sharpe, plus a final row for `model_breakdown` when non-empty. Layout
follows the existing per-row format used by `holdings_dashboard` / other
section kinds for visual consistency.

## Sample-size guards

| Computation | Minimum | Below threshold |
|---|---|---|
| Direction-level Sharpe | n_mature ≥ 5 AND stdev > 0.001 | `sharpe = None`, render as `"Sharpe n/a (n<5)"` or `"Sharpe n/a (flat)"` |
| Model-level row | n_mature ≥ 3 | row omitted (decisions still counted in overall buy_stats) |
| Direction row in block | n_mature ≥ 1 | row omitted entirely |
| `overall_sharpe` | total mature ≥ 5 AND stdev > 0.001 | `None`, omitted from "Overall:" prefix |

`statistics.stdev` requires ≥ 2 samples; the n≥5 guard supersedes that.

## Semantic change to `sell_stats`

`sell_stats` previously bundled SELL + TRIM verdicts. After this change it
holds SELL-only data. TRIM moves to its own `trim_stats` field.

Who reads `sell_stats` today (confirmed by `grep -rn "sell_stats" src tests`):
- `discover/track_record.py:232, 248, 258-260, 336-337` — the orchestration
  and rendering functions in this same module.
- `models/track_record.py:74, 89` — the type definition itself.

No tests, no CLIs, no dashboards, no other code path. The atomic update inside
`discover/track_record.py` is safe.

If a future caller wants the legacy SELL+TRIM bundled view, they can compute
it from `sell_stats` + `trim_stats` in seconds. Keeping the old bundling as an
extra `legacy_sell_stats` field would be backward-compatibility for nobody.

## Testing

New tests added to `tests/test_track_record.py`:

1. `test_per_model_breakdown_groups_by_opus_model`
   - Insert 6 BUY picks across 2 distinct `opus_model` values (4 + 2 picks).
   - Mock yfinance prices to produce known alpha values.
   - Run `measure_track_record`.
   - Assert: `model_breakdown` has exactly 1 entry (opus_model with n=4),
     the other (n=2) is dropped due to n<3 threshold.
   - Assert: the n=2 picks are still counted in `buy_stats.n_mature == 6`.

2. `test_sharpe_n_threshold`
   - 4 BUY picks (below n=5). Assert `buy_stats.sharpe is None`.
   - Add a 5th. Assert `buy_stats.sharpe is not None` and matches manual
     `mean / stdev` calculation to within 1e-6.

3. `test_sharpe_zero_stdev`
   - 5 BUY picks with identical alpha (`[0.05, 0.05, 0.05, 0.05, 0.05]`).
   - Assert `buy_stats.sharpe is None` (caught by `stdev < 0.001`).

4. `test_hold_alpha_sign_convention`
   - 2 HOLD reviews. Mock yfinance: ticker A returns +10%, SPY +2%
     (HOLD vindicated → alpha = +8%). Ticker B returns -5%, SPY +2%
     (HOLD wrong → alpha = -7%).
   - Assert `hold_stats.mean_alpha_pct ≈ 0.5%`.

5. `test_trim_alpha_sign_convention`
   - 2 TRIM reviews. Ticker A returns -10%, SPY +2% (TRIM right →
     alpha = +12%). Ticker B returns +8%, SPY +2% (TRIM wrong →
     alpha = -6%).
   - Assert `trim_stats.mean_alpha_pct ≈ 3.0%`.

6. `test_format_track_record_block_renders_all_directions`
   - Construct a `TrackRecord` with non-zero counts in all four directions
     plus 2 model-breakdown rows.
   - Assert `format_track_record_block` output contains "Buy track record:",
     "Hold track record:", "Trim track record:", "Sell track record:", and
     "Model breakdown:".

7. `test_format_track_record_block_omits_empty_directions`
   - Construct a `TrackRecord` with zero `hold_stats.n_mature`.
   - Assert "Hold track record:" line is absent from the rendered block.

Existing tests must continue to pass. The `Direction` literal expansion is
backward-compatible at the type level — old callers using `"buy"` / `"sell"`
still match the wider union.

## Risk register

| Risk | Mitigation |
|---|---|
| `sell_stats` semantic change breaks a future caller that imports it. | The only callers today are in `discover/track_record.py` itself; updated atomically. Future callers see the docstring update. |
| HOLD direction adds ~50% more yfinance fetches. | yfinance calls in `_score_pick` use a per-(ticker, date) cache; HOLDs share decision dates with bundled sell calls today, so cache hits dominate. Worst case: ~30 extra calls per `measure_track_record` invocation. |
| Sharpe is intuitively misleading with small samples. | Min-N=5 filter; render `n/a` below threshold. |
| `runs.opus_model` is `None` for very old runs. | Group those under `opus_model = "unknown"` in `model_breakdown`. Drop the entry if `n_mature < 3` like any other. |
| Per-decision Sharpe ≠ annualized Sharpe — user might compare to standard finance Sharpe values (~1-2 for good strategies). | Render label as `"Sharpe (per-decision)"` in the prompt block at least once so the user understands the unit. Don't annualize — the underlying alpha is already over a 90-day measurement window, and annualization assumes IID returns the picks don't have. |

## Dependencies

None. Pure-Python `statistics.stdev` / `statistics.mean` from stdlib.

## Open questions

None — design is fully specified.
