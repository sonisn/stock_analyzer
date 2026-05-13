# Covered-Call Writing — Design Spec

**Date:** 2026-05-13
**Owner:** snehal.soni
**Status:** Design approved, pending implementation plan
**Integration target:** `rebalance-portfolio` pipeline

## Goal

Extend the existing `rebalance-portfolio` pipeline so the single Opus rebalancer pass can recommend selling covered calls against held positions, collect the premium estimate, and reinvest that premium via the same plan's `ADD`/`BUY` actions — all in one coherent decision context.

## Non-goals

- **Trade execution.** The pipeline produces recommendations; the user places trades manually.
- **Multi-leg option strategies.** Covered calls only. No spreads, no cash-secured puts, no the wheel automation.
- **New pipeline.** No `cc-discover` or standalone covered-call CLI. This is purely additive to `rebalance-portfolio`.
- **Real-time options pricing.** End-of-day-ish quotes are sufficient.

## Strategy parameters (defaults, all overridable via env)

| Knob | Default | Meaning |
|---|---|---|
| `CC_ENABLED` | `true` | Master switch. `false` skips all CC logic. |
| `CC_TARGET_DELTA_MIN` | `0.35` | Lower bound for the LLM's strike-selection band. |
| `CC_TARGET_DELTA_MAX` | `0.45` | Upper bound. Aggressive-premium style. |
| `CC_DTE_MIN` | `30` | Min days-to-expiry. |
| `CC_DTE_MAX` | `45` | Max days-to-expiry. |
| `CC_DENYLIST` | `""` | Comma-separated tickers to never write calls against. |
| `CC_MIN_PREMIUM_USD` | `500` | If total expected premium < this, leave as cash (no reinvestment). |
| `CC_SLIPPAGE_BUFFER` | `0.10` | Fraction of premium held back for fill slippage when sizing reinvestment. |

**Style:** aggressive premium (Δ 0.35–0.45, DTE 30–45). Within that band, Opus pushes lower-Δ on conviction HOLDs and higher-Δ on TRIM-leaning positions.

## Architecture

```
rebalance-portfolio
 ├─ preflight              env vars + SnapTrade ping              [existing]
 ├─ holdings_fetch         positions, cash, OPEN OPTION POSITIONS [existing + small add]
 ├─ tax_lots               3yr activities → per-ticker lots       [existing, parallel]
 ├─ cc_eligibility   NEW   shares>=100 ∧ not in denylist ∧
                           shares minus open short-call coverage   [pure Python]
 ├─ option_chains    NEW   SnapTrade primary, yfinance fallback   [parallel]
 ├─ reviewer               Sonnet HOLD/TRIM/SELL per ticker       [existing, parallel]
 ├─ earnings_filter  NEW   drop expiries straddling next earnings [pure Python]
 ├─ rebalancer             Opus — sees holdings, lots, reviewer,
                           chains, eligibility. Emits actions
                           (incl. WRITE_CALL) + option_writes      [existing, extended]
 ├─ premortem              Opus — red-teams stock AND calls       [existing, extended]
 └─ report                 HTML / PDF / email / SQLite incl.
                           new CC sections + persisted option fields [existing, extended]
```

**Key shift:** option chain fetching is data-prep, not a post-step. By the time the rebalancer Opus runs, it has actual strikes, bid/ask, IV, delta, expiry dates — not hypotheticals.

## New / touched files

### New
- `src/stock_analyzer/data/options_chain.py` — `OptionChainProvider` Protocol, `SnapTradeChain` + `YFinanceChain` implementations, `fetch_chains(tickers, dte_min, dte_max) -> dict[str, OptionChain]` orchestrator with per-ticker fallback.
- *(No new agent module — the rebalancer is just extended.)*

### Touched
- `src/stock_analyzer/discover/rebalance_schema.py` — add `WRITE_CALL` to `RebalanceAction.action` literal; add `OptionWrite` model; add `option_writes` list to `RebalancePlan`.
- `src/stock_analyzer/discover/rebalancer.py` — extended system prompt (CC writing rules + premium reinvestment); chain/eligibility context assembly; post-LLM validation step.
- `src/stock_analyzer/discover/premortem.py` — system prompt extended to red-team `WRITE_CALL` actions.
- `src/stock_analyzer/cli/rebalance.py` — invoke `fetch_chains` + eligibility + earnings filter in the parallel data block; thread results into rebalancer.
- `src/stock_analyzer/data/transactions.py` *(or holdings module)* — parse open option positions from SnapTrade for the coverage subtraction.
- `src/stock_analyzer/discover/report_sections.py` — new section types: `PremiumIncome` + `PremiumDeployment`.
- `src/stock_analyzer/discover/report_html.py` / `report_pdf.py` — renderers for the two new sections.
- `src/stock_analyzer/discover/persistence.py` — six new nullable columns on `actions` table (strike, expiry, contracts, est_premium, delta, assignment_prob) with `ALTER TABLE` migration; track-record extension for CC outcomes.
- `src/stock_analyzer/discover/track_record.py` — WRITE_CALL scoring branch (EXPIRED_OTM vs ASSIGNED, opportunity-cost math).
- `src/stock_analyzer/config.py` — new optional CC settings.
- `src/stock_analyzer/preflight.py` — no required new checks; `CC_ENABLED` toggles whether to attempt chain fetch.
- `.env.example` — new `# Covered-call writing (optional)` section with all CC env vars and inline guidance.

## Schemas

### `OptionWrite` (new)
```python
class OptionWrite(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    strike: float
    expiry: str                       # ISO date "YYYY-MM-DD"
    contracts: int                    # must satisfy contracts*100 <= available_shares
    est_premium_per_contract: float   # mid of bid/ask, in dollars
    delta: float
    assignment_probability: float     # Opus may differ from delta
    notes: str = ""                   # one-line rationale
```

### `RebalanceAction` (extended)
```python
class RebalanceAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["SELL", "TRIM", "ADD", "BUY", "WRITE_CALL"]
    ticker: str
    sizing: str   # for WRITE_CALL: "2 contracts $260C 2026-06-20"
```

### `RebalancePlan` (extended)
```python
class RebalancePlan(BaseModel):
    # ...existing five fields...
    option_writes: list[OptionWrite] = Field(default_factory=list)
```

Six new scalar fields total. Well under Anthropic's structured-output complexity ceiling. All prose detail (alternative strikes considered, why this DTE, premium-deployment narrative) stays in `full_text`.

### Options data types (in `data/options_chain.py`)
```python
@dataclass(frozen=True)
class OptionQuote:
    strike: float
    expiry: date
    bid: float
    ask: float
    iv: float | None
    delta: float | None
    open_interest: int | None
    volume: int | None

@dataclass(frozen=True)
class OptionChain:
    ticker: str
    spot: float
    asof: datetime
    calls: list[OptionQuote]                        # OTM only, DTE-band filtered
    source: Literal["snaptrade", "yfinance", "missing"]
```

## Eligibility filter (deterministic)

A ticker is eligible for a `WRITE_CALL` if **all** hold:
- `shares_held >= 100`
- ticker ∉ `CC_DENYLIST`
- `available_for_cc = shares_held - 100 × open_short_call_contracts >= 100`
- reviewer verdict ≠ `SELL` (Opus is told this in-prompt; no Python enforcement)
- at least one chain expiry survives the earnings-blacklist filter

Produces an `EligibleHolding` record per ticker (shares available, max contracts, blacklisted expiries) consumed by the rebalancer.

## Rebalancer prompt extension

Three additions to the existing system prompt:

### (a) Per-ticker context block (assembled in Python)
```
TICKER: NVDA
  Reviewer verdict:        HOLD (high conviction)
  Shares held:             400
  Available for CC:        300 (100 already collateralizing open short call)
  Earnings-blacklist:      2026-05-21 (skip expiries 2026-05-16 .. 2026-05-23)
  Option chain (OTM calls, Δ 0.30–0.50, 30–45 DTE):
    2026-06-20 $250 strike  bid 3.10 / ask 3.30  Δ 0.42  IV 0.31  OI 4210
    2026-06-20 $260 strike  bid 2.20 / ask 2.40  Δ 0.36  IV 0.29  OI 2890
    2026-07-18 $270 strike  bid 2.85 / ask 3.05  Δ 0.38  IV 0.30  OI 1502
    [capped at ~8 rows per ticker]
```
Tickers with no chain show `Option chain: UNAVAILABLE`.

### (b) Covered-call writing rules
- **Target band:** Δ 0.35–0.45, DTE 30–45.
- **Strike within band:** HOLD-high-conviction → toward Δ 0.35; TRIM/low-conviction → toward Δ 0.45.
- **No WRITE_CALL on SELL positions.**
- **Coherence with TRIM:** contracts ≤ (shares_after_trim) // 100.
- **Liquidity guard:** skip strikes with bid < $0.20, OI < 100, or (ask − bid) / mid > 0.15.
- **Annualized yield in prose:** `premium / strike × (365 / DTE)`. Justify in `full_text` if < 8%.
- **Output:** one `WRITE_CALL` action per eligible ticker (max) + matching `OptionWrite` entry.

### (c) Premium reinvestment
- `expected_premium_total = Σ contracts × est_premium_per_contract × 100`
- `deployable = existing_cash + (1 - CC_SLIPPAGE_BUFFER) × expected_premium_total`
- Priority: fund `ADD`s on conviction HOLDs first, then `BUY`s, then cash.
- If `expected_premium_total < CC_MIN_PREMIUM_USD`: leave as cash, state reason in `full_text`.
- Respect existing aggressiveness-mode concentration/position-size limits.
- Show explicit dry-powder math in `full_text` (template provided in prompt).
- Note trade linkages in `full_text` ("If you skip the NVDA write, shrink the AMZN ADD by $340").

## Premortem prompt extension

Add one paragraph after the existing critique guidance:

> For each `WRITE_CALL`, additionally consider: (a) assignment lock-in if the underlying runs 20% past strike; (b) IV crush after near-term earnings or macro events; (c) opportunity cost of capping upside on conviction picks; (d) tax consequences if assignment triggers short-term gain on the underlying.

## Data flow

```
1. holdings_fetch:
     - equity positions, cash                        [existing]
     - open option positions (parse OCC symbols)     [new — small add]

2. parallel block:
     - tax_lots                                       [existing]
     - cc_eligibility (sync, fast)                    [new, pure Python]
     - option_chains (one fetch per eligible ticker) [new, parallel]
     - reviewer (per-ticker Sonnet)                  [existing]

3. earnings_filter (sync, fast)                       [new, pure Python]

4. rebalancer (one Opus call):
     Input includes context blocks for all tickers,
     with chain + eligibility for the eligible subset.
     Output: RebalancePlan with actions (mixed action types)
     and option_writes list.

5. validation (sync, fast):
     - Each WRITE_CALL must have a matching OptionWrite by ticker.
     - Orphan WRITE_CALLs → dropped, logged loud.
     - contracts × 100 > available_for_cc → clamped or dropped.

6. premortem (Opus, sees CC plan)                     [existing, extended]

7. report:
     - render HTML/PDF with new "Premium Income" +
       "Premium → Deployment" sections                [extended]
     - persist actions (with option columns) to SQLite [extended]
     - email subject annotated with premium total      [extended]
```

## Error handling

Graceful degradation everywhere; the pipeline never fails because of CC.

| Failure | Behavior |
|---|---|
| `CC_ENABLED=false` | Skip eligibility/chains/earnings_filter entirely. Rebalancer runs without CC context. |
| SnapTrade chain endpoint missing for tier | Per-ticker fallback to yfinance. INFO log on first miss. |
| yfinance error per ticker | Treat as missing chain. WARN log. |
| Both providers fail for a ticker | `OptionChain.source = "missing"`; Opus sees `UNAVAILABLE` and won't WRITE_CALL on that ticker. |
| Earnings date unknown for a ticker | No blacklist; add `earnings_unknown: true` to context; Opus told to be conservative on DTE. |
| No tickers ≥100 shares | Skip option_chains + earnings_filter. Pipeline runs unchanged. |
| Schema returns WRITE_CALL with no matching OptionWrite | Validation drops the action, surfaces a warning in email summary. |
| Schema returns WRITE_CALL with `contracts × 100 > available_for_cc` | Validation clamps contracts to max allowed, logs loud. |

## Reporting

### Premium Income section (rendered only when `option_writes` non-empty)
```
┌─ Premium Income ────────────────────────────────────────────────┐
│ Ticker  Strike  Expiry      Qty  Premium    Δ      Assign %    │
│ NVDA    $260    2026-06-20   3    $720    0.36    36%          │
│ AAPL    $230    2026-06-20   2    $640    0.40    40%          │
│ AMZN    $215    2026-07-18   1    $190    0.32    32%          │
│ ─────────────────────────────────────────────────────────────  │
│ Gross premium: $1,550   Slippage buffer (10%): -$155            │
└─────────────────────────────────────────────────────────────────┘
```

### Premium → Deployment section (rendered when both `option_writes` and `ADD`/`BUY` actions exist)
```
┌─ Premium → Deployment ──────────────────────────────────────────┐
│ Deployable premium: $1,395                                      │
│ Existing cash:       $850                                       │
│ Total dry powder:  $2,245                                       │
│                                                                  │
│   → ADD VRT  $1,400                                              │
│   → BUY PLTR  $600                                               │
│   → Cash held: $245                                              │
└─────────────────────────────────────────────────────────────────┘
```

The deployment math is computed deterministically in the renderer from `option_writes` + ADD/BUY actions, not parsed out of `full_text`. The prose narrative is preserved for human context.

**Email subject:** existing format gets one annotation: `[Rebalance] 4 actions + $1,550 premium`.

**Color palette:** WRITE_CALL gets a teal/green badge in the action-type legend (final hex during implementation).

## Persistence

### Migration
```sql
ALTER TABLE actions ADD COLUMN strike            REAL;
ALTER TABLE actions ADD COLUMN expiry            TEXT;
ALTER TABLE actions ADD COLUMN contracts         INTEGER;
ALTER TABLE actions ADD COLUMN est_premium       REAL;   -- per contract
ALTER TABLE actions ADD COLUMN delta             REAL;
ALTER TABLE actions ADD COLUMN assignment_prob   REAL;
```

All nullable; legacy rows unaffected. For WRITE_CALL rows, the six columns are populated from the matching `OptionWrite`. SELL/TRIM/ADD/BUY rows leave them NULL.

### Track-record scoring (extension in `track_record.py`)
After `expiry` date passes, the scorer pulls historical spot via yfinance and:
- If `spot_at_expiry < strike`: outcome = `EXPIRED_OTM`, P&L = `+est_premium × contracts × 100`.
- If `spot_at_expiry >= strike`: outcome = `ASSIGNED`, P&L = `(est_premium − max(0, spot_at_expiry − strike)) × contracts × 100`. (Opportunity cost recorded.)
- Compare to SPY return over the same DTE window for fair benchmarking.

This feeds the recurring track-record block in the email: *"Last 12 covered calls: 10 expired OTM, 2 assigned. Net premium captured: $X. vs SPY: +Y%."*

## Testing

New test files (matching the flat `tests/test_*.py` layout):

| File | Coverage |
|---|---|
| `test_options_chain.py` | SnapTrade adapter (mocked client); yfinance adapter (mocked yfinance); both-fail returns `source="missing"`; DTE-band + OTM strike filtering. |
| `test_cc_eligibility.py` | `shares >= 100` floor; denylist; open-short-call subtraction (OCC symbol parsing); earnings-expiry blacklist. |
| `test_cc_schema.py` | `OptionWrite` validation; `RebalancePlan.option_writes` round-trip; legacy plans (no `option_writes`) still parse. |
| `test_rebalance_validation.py` | Orphan `WRITE_CALL` (no matching `OptionWrite`) gets dropped; `contracts × 100 > available` gets clamped; both cases logged. |
| `test_premium_deployment.py` | Deterministic dry-powder math (gross, buffer, deployable, total). |
| `test_track_record.py` (extend) | WRITE_CALL scoring: EXPIRED_OTM vs ASSIGNED branches; opportunity-cost math. |
| `test_pipeline_wiring.py` (extend) | End-to-end with stubbed Opus returning a mixed-action plan; assert email renders both new sections, SQLite persists option columns. |

**Fixtures:** new `tests/fixtures/` folder with canned SnapTrade chain JSON, yfinance chain frame, reviewer output, and Opus rebalance plan with WRITE_CALL + ADD.

**No new test dependencies** — pytest + existing mocking patterns.

## Open questions for implementation phase

1. **Final color hex** for the WRITE_CALL badge in the report palette.
2. **SnapTrade chain endpoint availability** on the user's tier — verified empirically during implementation; yfinance fallback covers the gap if not.
3. **OCC symbol parsing helper** — small utility; place in `data/transactions.py` or a new `data/options_symbols.py` (decide during implementation).
4. **Chain row cap per ticker** in the LLM context — defaulted to ~8 rows; tune during prompt iteration.

## Future work (explicitly out of scope for v1)

- Cash-secured puts ("the wheel").
- Roll suggestions when an existing short call goes ITM.
- Live trade execution via SnapTrade `place_option_strategy`.
- IV-rank screening to time call-writing cycles.
- Dynamic CC style per ticker (LLM picks the delta band, not just the strike within it).
