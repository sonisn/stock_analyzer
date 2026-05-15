# Wheel / CSP Action Generation — Design

**Date:** 2026-05-15
**Status:** Approved (pending spec review)
**Author:** snehal.soni

## Problem

The portfolio analyzer recommends covered calls (CC) on stocks you already hold
≥100 shares of (states C/D of the classic wheel). It does not recommend
cash-secured puts (CSPs) on tickers you'd want to own (states A/B). That means
the rebalancer only operates on the back half of the wheel and the user has
no LLM-generated guidance for the front half.

This spec adds **CSP action generation** to the existing rebalancer. Tracking
of open CSPs through to assignment is explicitly out of scope (separate spec
later).

## Goals

1. Add `SELL_PUT` as a `RebalanceAction.action` value and a `CashSecuredPut`
   structured-output model alongside `OptionWrite` in `RebalancePlan`.
2. Define the CSP universe per run from the last N discover-run BUY picks not
   currently held.
3. Premium-harvest posture: 10-20Δ, 30-45 DTE OTM puts.
4. Cash discipline: cap each CSP at 25% of `cash_budget`; keep ≥20% as dry
   powder (sum of all CSP collateral ≤80% of `cash_budget`).
5. Extend the existing options-chain fetcher so it can also return OTM puts.

## Non-goals

- No detection of already-open short put positions.
- No assignment-probability rendering for existing CSPs.
- No CSP roll / close decisions.
- No new cash-data integration. We reuse the existing `cash_budget` parameter
  the user already passes to the rebalancer.
- No iron condor or other multi-leg strategies.
- No state-machine code abstraction; the rebalancer prompt continues to
  recommend actions based on context.

## Architecture overview

```
src/stock_analyzer/
  models/
    rebalance.py            ── CashSecuredPut + extended ActionType + csp_writes
    portfolio.py            ── CspCandidate model
    market.py               ── OptionChain gains `puts: list[OptionQuote]`
  data/
    options_chain.py        ── fetch_chains(..., kind="calls"|"puts"|"both")
                               + provider classes populate `puts`
  db/
    repository.py           ── new fetch_recent_buy_picks helper
  discover/
    csp_eligibility.py      (NEW)
    csp_validation.py       (NEW)
    csp_backfill.py         (NEW)
    csp_render.py           (NEW)
    rebalancer.py           ── prompt instructions for SELL_PUT
  cli/
    rebalance.py            ── wire CSP eligibility into prompt context

tests/
  test_csp_eligibility.py   (NEW)
  test_csp_validation.py    (NEW)
  test_csp_backfill.py      (NEW)
  test_options_chain.py     ── extend to cover puts
  test_pipeline_wiring.py   ── extend to cover SELL_PUT flow
```

## Data flow per rebalance run

1. CLI fetches the **last 3 discover-run BUY picks** via a new
   `db.repository.fetch_recent_buy_picks(session, *, n_runs=3)` helper.
   Returns `list[tuple[str, int, str]]` of `(ticker, rank, run_at)` for every
   pick in the last `n_runs` discover runs.
2. `discover/csp_eligibility.eligible_csp_tickers(picks, holdings, cash_budget,
   denylist)` filters:
   - drop tickers user already holds ≥100 shares of (those route to the CC
     side, not CSP),
   - drop tickers in the existing `OPTIONS_DENYLIST` env var (currently
     `CC_DENYLIST`, **renamed for clarity** — see Migration below),
   - drop tickers with no available put chain (filled in downstream when
     `fetch_chains` returns `source="missing"`).
   Returns `dict[ticker, CspCandidate]`.
3. `data.options_chain.fetch_chains(eligible_tickers, dte_min=30, dte_max=45,
   kind="puts")` fetches OTM put chains.
4. `apply_earnings_filter` is reused unchanged: drops expiries within
   ±7 days of an earnings date (the function operates on `OptionChain.calls`
   today; it now also operates on `.puts`).
5. The rebalancer prompt block lists each eligible ticker's put chain (strikes
   / expiries / deltas / mid premiums) + the user's `cash_budget` + the
   concentration constraints. The existing CC prompt block continues to
   render alongside.
6. Opus emits a `RebalancePlan` whose `actions` may include
   `RebalanceAction(action="SELL_PUT", ticker=..., sizing="2 contracts at $145
   strike, exp 2026-07-18")` and whose `csp_writes: list[CashSecuredPut]`
   carries the structured detail. `option_writes` (existing) continues to
   carry CC detail.
7. `discover/csp_validation.validate_csp_writes(plan, eligible, cash_budget,
   spots)` enforces the 8 validation rules below; invalid `CashSecuredPut`
   entries are dropped with a logged warning and the corresponding
   `SELL_PUT` action is removed from `plan.actions`.
8. `discover/csp_backfill.backfill_csp_writes(plan, spots)` parses the
   `sizing` string of any unmatched `SELL_PUT` action and constructs a
   `CashSecuredPut` if Opus omitted the structured field (mirrors
   `cc_backfill` precisely).
9. `discover/csp_render.render_csp_actions(plan)` produces the report
   action-table rows and a cash-summary line (`"CSPs reserve $X across N
   tickers (Y% of budget)"`).

## Models

### `models/rebalance.py`

```python
class CashSecuredPut(BaseModel):
    """Structured detail for one SELL_PUT action.

    Cash-secured: ``strike * 100 * contracts`` is reserved as collateral for
    potential assignment. Premium is per-share (multiply by 100 for per-
    contract dollars). Put delta is negative; we store the raw signed value
    and compute assignment-probability proxy as ``abs(delta)``.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    strike: float
    expiry: str = Field(..., description="ISO date YYYY-MM-DD.")
    contracts: int = Field(..., gt=0, description="Number of put contracts to sell.")
    est_premium_per_share: float = Field(
        ..., ge=0,
        description="Mid of bid/ask in dollars per share. ×100 = per contract.",
    )
    delta: float = Field(
        ..., ge=-1.0, le=0.0,
        description="Put delta is negative; abs(delta) is the assignment-probability proxy.",
    )
    cash_reserved: float = Field(
        ..., ge=0,
        description="strike × 100 × contracts. Validation cross-checks this.",
    )
    notes: str = ""


# ActionType union extends:
class RebalanceAction(BaseModel):
    ...
    action: Literal["SELL", "TRIM", "ADD", "BUY", "WRITE_CALL", "SELL_PUT"]
    ...


# RebalancePlan gains:
class RebalancePlan(BaseModel):
    ...
    option_writes: list[OptionWrite] = ...  # existing, CC
    csp_writes: list[CashSecuredPut] = Field(
        default_factory=list,
        description=(
            "Parallel to SELL_PUT actions. Each entry MUST have a matching "
            "SELL_PUT in `actions` with the same ticker. Empty list when no "
            "CSPs are recommended."
        ),
    )
```

`__all__` adds `"CashSecuredPut"`.

### `models/portfolio.py`

```python
class CspCandidate(BaseModel):
    """A ticker eligible for CSP recommendations this run.

    The eligibility logic in csp_eligibility.py builds this; the rebalancer
    prompt iterates them. ``max_csp_cash`` is min(per-ticker cap, remaining
    budget) — the upper bound the LLM should size CSPs to."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    last_pick_run_at: str
    last_pick_rank: int
    spot: float
    max_csp_cash: float
```

`__all__` adds `"CspCandidate"`.

### `models/market.py`

```python
class OptionChain(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    spot: float
    asof: datetime
    calls: list[OptionQuote] = Field(default_factory=list)  # OTM calls only
    puts: list[OptionQuote] = Field(default_factory=list)    # OTM puts only — NEW
    source: Literal["tradier", "yfinance", "missing"] = "missing"
```

The added field is `default_factory=list`, so existing constructors that
omit `puts=` continue to work — CC callers don't need to change.

## Options chain plumbing

### `data/options_chain.py`

`fetch_chains` signature gains a `kind` parameter:

```python
def fetch_chains(
    tickers: list[str],
    *,
    dte_min: int,
    dte_max: int,
    kind: Literal["calls", "puts", "both"] = "calls",
) -> dict[str, OptionChain]:
    ...
```

`kind="calls"` is the default — every existing caller stays backward-compatible.

The `YFinanceChain` and `TradierChain` provider classes both gain a matching
parameter or internal branching that selects which side(s) to populate:

- **yfinance**: `Ticker(t).option_chain(expiry)` returns a tuple with `.calls`
  and `.puts` DataFrames. The provider already iterates expiries and filters
  to OTM calls (`strike > spot`); add a symmetric branch for `kind in
  ("puts","both")` that filters OTM puts (`strike < spot`).
- **Tradier**: the `/markets/options/chains` REST endpoint already returns
  both `option_type` (CALL/PUT) for every strike; the provider currently
  discards puts. Switch the filter to keep calls and/or puts based on `kind`.

For `kind="puts"`, the returned `OptionChain` has `calls=[]` and the OTM
puts in `.puts`. For `kind="both"`, both lists are populated.

The existing per-provider error handling (return `OptionChain(... source="missing")`
on any error) is preserved unchanged.

## CSP eligibility — `discover/csp_eligibility.py`

```python
def eligible_csp_tickers(
    picks: list[tuple[str, int, str]],
    *,
    positions: dict[str, dict[str, float]],
    spots: dict[str, float],
    cash_budget: float,
    denylist: tuple[str, ...],
    max_pct_per_ticker: float = 0.25,
) -> dict[str, CspCandidate]:
    """Filter picks to tickers eligible for CSP recommendations.

    Drops:
      - tickers user already holds >= 100 shares of (CC candidates)
      - tickers in the denylist
      - tickers with no recent pick (n_runs argument controls the lookback)
    """
```

`positions` matches the existing `_aggregate_positions` shape in
`cli/rebalance.py`. `spots` is the technicals-stage price dict. `cash_budget`
sets the per-ticker cap. Empty `picks` returns `{}`.

Dedup rule: when a ticker appears in multiple discover runs, keep the
**most recent** (highest `run_at`) and surface its rank in `last_pick_rank`.

## CSP validation — `discover/csp_validation.py`

```python
def validate_csp_writes(
    plan: RebalancePlan,
    *,
    eligible: dict[str, CspCandidate],
    cash_budget: float,
    spots: dict[str, float],
) -> RebalancePlan:
    """Return a new RebalancePlan with invalid CashSecuredPut entries
    (and their corresponding SELL_PUT actions) removed. Logs validation
    warnings for the user."""
```

Rules applied to each `CashSecuredPut`:
1. **Ticker eligible.** `cp.ticker in eligible`.
2. **Cash math.** `abs(cp.strike * 100 * cp.contracts - cp.cash_reserved) < 0.01`.
3. **OTM.** `cp.strike < spots[cp.ticker]`.
4. **Delta band.** `0.10 <= abs(cp.delta) <= 0.25`.
5. **DTE band.** Days between today and `cp.expiry` is in `[30, 45]`.
6. **Matching action.** Exactly one `RebalanceAction(action="SELL_PUT",
   ticker=cp.ticker)` exists in `plan.actions`.
7. **Total CSP cap.** Sum of all `cash_reserved` ≤ `cash_budget * 0.80`.
8. **Per-CSP cap.** Each individual `cash_reserved` ≤ `cash_budget * 0.25`.

Violations result in dropping the offending `CashSecuredPut` and its matching
`SELL_PUT` action from the returned plan; a `logger.warning` describes which
rule fired.

Constants `_DELTA_LOW=0.10`, `_DELTA_HIGH=0.25`, `_DTE_LOW=30`, `_DTE_HIGH=45`,
`_TOTAL_BUDGET_PCT=0.80`, `_PER_CSP_PCT=0.25` live at module top.

## CSP backfill — `discover/csp_backfill.py`

Mirrors `cc_backfill.py`. Parses `RebalanceAction.sizing` strings such as
`"2 contracts at $145 strike, exp 2026-07-18"` to construct a
`CashSecuredPut` when Opus emits a `SELL_PUT` action without a corresponding
entry in `csp_writes`. Regex pattern:

```python
_CSP_SIZING_RE = re.compile(
    r"^\s*(?P<contracts>\d+)\s+contract[s]?\s+at\s+\$(?P<strike>[\d.]+)\s+strike"
    r"(?:,\s+exp\s+(?P<expiry>\d{4}-\d{2}-\d{2}))?",
    re.IGNORECASE,
)
```

When `expiry` is missing the entry is skipped (cannot backfill incomplete
data). `delta` and `est_premium_per_share` cannot be reconstructed from the
sizing string — they default to `delta=-0.15, est_premium_per_share=0.0`
with a `notes="backfilled from sizing; delta/premium not parseable"` so the
downstream validation still catches OTM/delta violations.

## CSP rendering — `discover/csp_render.py`

Two outputs:

1. **Action-table rows** for the structured-output PDF section. Format:
   ```
   SELL_PUT  NVDA  2 contracts × $145 strike (exp 2026-07-18)
             Premium: $2.85/share = $570/contract = $1,140 total
             Cash reserved: $29,000  Delta: -0.15  Assign prob: 15%
   ```
2. **Cash-summary line** appended below the action table:
   ```
   CSP cash reservation: $29,000 across 1 ticker (10.4% of $280,000 budget)
   ```

## DB helper — `db/repository.py`

Add:

```python
def fetch_recent_buy_picks(
    session: Session, *, n_runs: int = 3
) -> list[tuple[str, int, str]]:
    """Return [(ticker, rank, run_at), ...] for every BUY pick in the last
    `n_runs` discover runs, ordered (ticker dedup keeps most-recent run)
    most-recent-first. Used by the CSP eligibility filter."""
```

Mirrors the existing `fetch_recent_holdings_history` pattern but joins
`picks` to `runs` filtered by `kind = 'discover'`.

## Migration — `CC_DENYLIST` → `OPTIONS_DENYLIST`

The existing `CC_DENYLIST` env var becomes `OPTIONS_DENYLIST` because CSP
recommendations share the same denylist (you don't want to write puts on
tickers you've blacklisted for calls). The CC eligibility code's `denylist`
parameter doesn't change shape; only the env-var name moves.

Backward-compat: read `OPTIONS_DENYLIST` first, fall back to `CC_DENYLIST` if
unset. Log a one-time deprecation warning when the old name resolves.

Files touched: `config.py` (env var read), `cli/rebalance.py` (passes
denylist into eligibility), `cc_eligibility.py` (no code change — caller
controls the value), `csp_eligibility.py` (new).

## Rebalancer prompt extension

The prompt's "Options context" section gains a CSP block when
`eligible_csp_tickers` is non-empty:

```
=== CSP CANDIDATES ===
You may sell cash-secured puts on these tickers (drawn from recent discover
picks; you'd own them at the strike). Posture is PREMIUM-HARVEST:
  - Target delta: 0.10-0.25 (low assignment probability)
  - Target DTE:   30-45 days
  - Per-CSP cap:  25% of cash budget ($X)
  - Total cap:    80% of cash budget ($Y)

NVDA (last picked rank 2 on 2026-04-30; spot $158.42):
  2026-07-18 expiry, OTM puts:
    $145 strike  bid/ask 2.80/2.90  delta -0.15  oi 12,400
    $140 strike  bid/ask 1.95/2.05  delta -0.10  oi  8,200
  (... more strikes ...)

GOOGL (last picked rank 3 on 2026-04-30; spot $175.20):
  (... chain ...)

For each CSP you recommend, emit a SELL_PUT action AND a matching
CashSecuredPut entry in csp_writes with the cash_reserved math.
```

The Opus rebalancer already balances multiple action types within a budget;
SELL_PUT joins the same pool.

## Testing

### `tests/test_csp_eligibility.py`
- `test_eligible_csp_filters_held_tickers` — pick that's already held drops out.
- `test_eligible_csp_dedups_to_most_recent_pick` — ticker in 2 runs surfaces with latest run's data.
- `test_eligible_csp_respects_denylist` — denylisted ticker drops.
- `test_eligible_csp_returns_empty_on_no_picks` — empty input → `{}`.
- `test_max_csp_cash_caps_at_per_ticker_pct` — `max_csp_cash = min(0.25 * budget, budget)`.

### `tests/test_csp_validation.py`
One test per rule, each constructing a deliberately-bad `RebalancePlan` and
asserting the offending `CashSecuredPut` + matching `SELL_PUT` are dropped:
- `test_rejects_ineligible_ticker`
- `test_rejects_bad_cash_math`
- `test_rejects_strike_above_spot`
- `test_rejects_delta_out_of_band` (both low and high)
- `test_rejects_dte_out_of_band` (both too short and too long)
- `test_rejects_unmatched_action`
- `test_rejects_total_over_80pct`
- `test_rejects_single_csp_over_25pct`
- `test_accepts_valid_csp` — positive case.

### `tests/test_csp_backfill.py`
- `test_backfill_parses_standard_sizing`
- `test_backfill_skips_missing_expiry`
- `test_backfill_no_op_when_csp_already_present`

### `tests/test_options_chain.py` (extend)
- `test_fetch_chains_kind_calls_returns_only_calls` (regression)
- `test_fetch_chains_kind_puts_returns_only_puts`
- `test_fetch_chains_kind_both_returns_both`
- `test_puts_filter_is_otm_only` (strike < spot)
- `test_yfinance_provider_populates_puts`
- `test_tradier_provider_populates_puts`

### `tests/test_pipeline_wiring.py` (extend)
- `test_sell_put_action_flows_through_pipeline` — construct a discover-run +
  rebalance flow, mock the LLM to emit a SELL_PUT, verify validation runs,
  verify report renders.
- `test_csp_assignment_transitions_to_cc_state` — start with 0 shares of T,
  simulate assignment via inserting 100 shares into positions, verify
  `eligible_holdings` now recognizes T as a CC candidate.

## Risk register

| Risk | Mitigation |
|---|---|
| Reusing `cash_budget` for both BUY/ADD and CSP collateral risks double-booking | Validation rule #7 enforces 80% cap; prompt explicitly states the constraint; Opus already balances action types within a budget. |
| Frozen `OptionChain` previously had only `calls`; consumers may iterate `chain.calls` expecting "all options" | Default `puts=[]` keeps consumers safe. Grep audit verifies no consumer assumes "calls" means "every option". |
| yfinance puts fetching could be flaky for thin tickers | Existing per-ticker error handling returns `source="missing"` chain; downstream renders `UNAVAILABLE`. |
| Backfill cannot reconstruct delta/premium from sizing string | Backfill stamps `delta=-0.15` (mid of premium-harvest band) and `est_premium_per_share=0.0`; validation rule #4 still gates delta. Premium misreporting only affects the rendered report's premium-income line, not action correctness. |
| LLM picks a CSP whose cash_reserved overlaps a BUY action's notional | Validation rule #7 sums CSP cash; BUY notional is separate; Opus must respect the total cash budget across action types. The 80% cap leaves 20% for BUYs and dry powder; an explicit prompt note tells the LLM this. |
| CSP gets assigned, leaving the user holding shares not yet tracked in positions | The next rebalance run's brokerage-state fetch picks them up; CC eligibility then takes over. The transition is tested in `test_csp_assignment_transitions_to_cc_state`. |
| `CC_DENYLIST` → `OPTIONS_DENYLIST` rename breaks user's `.env` | Backward-compat fallback reads `CC_DENYLIST` if `OPTIONS_DENYLIST` unset; deprecation warning logged once. |
| `validate_csp_writes` returning a new `RebalancePlan` requires `Plan` to be re-buildable | `RebalancePlan` is frozen Pydantic; use `plan.model_copy(update={"actions": ..., "csp_writes": ...})`. |

## Dependencies

None. All deps already in `pyproject.toml`.

## Open questions

None — design is fully specified.
