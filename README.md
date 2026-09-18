# stock-analyzer

A personal portfolio analyzer that runs two pipelines on top of Claude
(Opus + Sonnet for most stages, with Gemini and OpenAI in the mix for
ranker consensus and red-team), market data, and brokerage holdings:

- **`discover-stocks`** — surface 5 medium-term picks from a screened
  universe (S&P 500 snapshot + watchlist + holdings, with news coverage as
  a conviction overlay rather than the candidate pool). Sonnet writes
  per-ticker analyst reports; the ranker runs one high-effort consensus
  round per provider in `DISCOVER_RANKER_PROVIDERS` (default claude,
  gemini, openai) and majority-votes the top-N picks with
  probability-weighted scenarios; a red-team pass (default Gemini, so it
  isn't checking the ranker's own blind spots) writes bear cases; an Opus
  sizer allocates the new capital, weighting by the ranker's consensus
  agreement ratio as well as conviction; a deterministic macro-veto pass
  suppresses high-momentum picks when the FRED regime reads risk-off.
- **`rebalance-portfolio`** — review every brokerage holding with
  Sonnet, then have Opus produce a structured action plan (SELL /
  TRIM / ADD / BUY) with tax-lot guidance. A second Opus pass writes
  a plan-level pre-mortem (adversarial hindsight).

Both pipelines emit a structured **HTML email + PDF**, persist every
run to **SQLite** for cross-run track-record scoring, and dump the
full analysis to the log so you never lose a run to an email failure.

## What's in it

| Stage | Model | What it does |
|---|---|---|
| Universe | — | Sampling frame (S&P 500 snapshot + watchlist + holdings) plus a news-derived conviction overlay |
| Fundamentals / Technicals / EPS revisions / Sector rotation / Macro | — | Parallel data fetches (yfinance, FRED, FinnHub) |
| Track record | — | Score past BUY/HOLD/TRIM/SELL calls vs SPY at fixed 30d + 90d horizons, beta-adjusted |
| Market themes | Sonnet | Identify 3-8 themes grounded in actual price + revision data |
| Screen | — | Hard filters + 0-100 composite score (45 fundamentals / 45 trend / 10 attention) |
| Enrichment (parallel) | — | News, earnings, insider selling, share trades, peers, 10-Q MD&A, transcripts |
| Analyst | Sonnet | Per-ticker analyst report with structured output |
| Ranker | Claude + Gemini + OpenAI (one consensus round each) | Top-N picks with 3 scenarios (bull/base/bear) + EV + agreement ratio |
| Macro veto | — (deterministic) | Suppress high-momentum picks in a risk-off FRED regime |
| Red-team | Gemini (default) | Bear case per pick with fragility rank + watch metric |
| Sizer | Opus | Allocate new capital; flag concentration / correlation; weights by consensus agreement |
| Holdings review | Sonnet | HOLD / TRIM / SELL per position with tax-lot plan |
| Rebalance | Opus | Structured action plan with aggressiveness knob |
| Pre-mortem | Opus | Adversarial hindsight on the rebalance plan |

### Covered-call writing (rebalance pipeline)

When enabled (`CC_ENABLED=1`, default), the rebalancer can recommend
selling covered calls against any held position with ≥ 100 shares.
Opus picks strikes in the Δ 0.35–0.45, DTE 30–45 band (aggressive
premium style), leaning further OTM on high-confidence holdings and
closer to the money on TRIM-leaning ones.

The same Opus pass also deploys the expected premium (minus a 10%
slippage buffer) via `ADD`/`BUY` actions, and may propose
**stub-consolidation** trades — selling sub-100-share stubs to fund
round-lot completions that expand future CC capacity.

Output adds three sections to the rebalance email: **Premium Income**
(per-contract recommendation table), **Round-Lot Coverage**
(stub decomposition for every holding), and **Premium → Deployment**
(dry-powder math).

**Options chain data:** the pipeline uses **Tradier** as the primary chain
provider (real-time bid/ask + Greeks like delta/IV). Set `TRADIER_API_KEY`
in your `.env` to enable — a free Tradier brokerage
account (no funding minimum) gives you a production access token at
[dash.tradier.com](https://dash.tradier.com/). If `TRADIER_API_KEY` is
unset or Tradier is unreachable, the pipeline falls back to **yfinance**
(free, 15–20 min delayed, no Greeks — Opus picks strikes via strike-vs-spot
proxy). Both work; Tradier gives meaningfully better strike selection
because the LLM gets accurate delta values for the Δ 0.35–0.45 band rule.

**IV timing signal:** Opus picks not just WHICH strike but WHETHER to
write right now. The pipeline ships a free realized-volatility proxy —
compute 252-day annualized HV per eligible ticker from yfinance closes,
then compare to current chain IV to label the regime as elevated
(IV/HV ≥ 1.20), average (0.90-1.19), or depressed (< 0.90). Opus
skips writes in depressed regimes unless conviction is HOLD with
confidence ≥ 8.

### Cash-secured puts (rebalance pipeline)

The front half of the wheel. When enabled (`CSP_ENABLED=1`, default), the
rebalancer can recommend selling puts on recent discover picks you don't
hold a round lot of: you're paid premium now and only buy the stock if it
closes below the strike, i.e. below today's price. Candidates are the
current run's picks plus the last `CSP_PICK_LOOKBACK_RUNS` runs', minus
denylisted tickers, picks whose thesis is BROKEN or target already hit,
and tickers with a put already open.

Posture is premium harvest: |Δ| 0.10–0.25, 30–45 DTE, expiries near
earnings removed. Collateral (strike × 100 × contracts) is capped at 25%
of available cash per put and 80% in total; cash already backing open
short puts is excluded first. After the LLM answers, every put is checked
against the fetched chain (strike and expiry must exist; delta and premium
are re-read from it) and contracts are cut to fit the caps. The rebalance
email gets a **Cash-secured puts** table: premium, annualized yield, cash
reserved and cost if assigned. When yfinance is the chain source, put
deltas are estimated from IV (Black-Scholes).

See `.env.example` for the full set of `CC_*`, `CSP_*`, `OPTIONS_DENYLIST`
and `TRADIER_*` knobs.

## Quickstart

```bash
# Install (Python 3.14+, uses uv)
uv sync

# Configure — copy and fill in your keys
cp .env.example .env

# Run
uv run discover-stocks            # find new picks
uv run rebalance-portfolio        # review holdings + plan
uv run analyze-portfolio          # one-off analyst-style report
uv run analyze-insiders           # insider + political trade signals
```

## Required env vars

Minimum to run a discover pipeline:

- `ANTHROPIC_API_KEY` — Claude (used for most stages regardless of ranker config)
- `DISCOVER_OPUS_MODEL` / `DISCOVER_SONNET_MODEL` — model IDs
- `FINNHUB_API_KEY`, `FRED_API_KEY`, `TAVILY_API_KEY` — data providers

The Ranker's consensus vote runs one round per provider in
`DISCOVER_RANKER_PROVIDERS` (default `claude,gemini,openai`) — so by
default you also need `GOOGLE_API_KEY` and `OPENAI_API_KEY`. To go back to
a single-provider ranker (no Gemini/OpenAI keys needed), set
`DISCOVER_RANKER_PROVIDERS=claude` (or `claude,claude,claude` for the old
N-resample-one-model behavior). `DISCOVER_REDTEAM_PROVIDER` (default
`gemini`) and `DISCOVER_FALLBACK_PROVIDER` (default `claude`) may also
need their own keys — see `.env.example`.

For rebalance, additionally:

- `SNAPTRADE_CLIENT_ID`, `SNAPTRADE_CONSUMER_KEY`, `SNAPTRADE_USER_ID`,
  `SNAPTRADE_USER_SECRET` — brokerage holdings + transaction history

For covered-call writing (optional but recommended — better strike picks):

- `TRADIER_API_KEY` — real-time option chains + Greeks. Free with a
  Tradier brokerage account (no funding minimum). Without it, the
  pipeline falls back to delayed yfinance data with no Greeks.
- `TRADIER_BASE_URL` — defaults to production `https://api.tradier.com/v1`.

For email delivery:

- `EMAIL_TO`, `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`,
  `SMTP_FROM`, `SMTP_USE_SSL`

A full annotated list lives in `.env.example`.

### Request pacing (optional)

Yahoo throttles aggressively and everything yfinance-related goes through
`data/yf_gateway.py`, which caps concurrency process-wide, paces requests,
and halves its own rate whenever Yahoo answers `Too Many Requests`
(recovering as calls succeed). The defaults are tuned for a full discover
run; raise or lower them only after reading the `yfinance [discover run]:`
summary logged at the end of a run.

- `YF_MAX_CONCURRENCY` (4) — max in-flight Yahoo requests for the process
- `YF_RATE_LIMIT_PER_MIN` (150) — starting/ceiling request rate
- `YF_MIN_RATE_PER_MIN` (20) — floor the backoff stops at
- `YF_MAX_ATTEMPTS` (4) — tries per call before giving up
- `YF_COOLDOWN_SECONDS` (15) — base global pause after a rate limit
- `FINNHUB_RATE_LIMIT_PER_MIN` (55) — under the 60/min free-tier ceiling
- `DISCOVER_MAX_SCREEN_CANDIDATES` (250) — cap on names that reach the
  per-ticker fundamentals + EPS fetches, after the trend gate

## Architecture

```
src/stock_analyzer/
├── cli/             # Entry points + pipeline orchestration
│   ├── discover.py
│   ├── rebalance.py
│   ├── portfolio.py
│   └── insider.py
├── discover/        # Multi-agent pipeline modules
│   ├── analyst.py / ranker.py / redteam.py / sizer.py
│   ├── reviewer.py / rebalancer.py / premortem.py
│   ├── market_themes.py / track_record.py / tax_lot_helper.py
│   ├── calibration.py                       # grade the ranker's own EV + conviction,
│   │                                          # plus factor-similar past setups
│   ├── score_validation.py                   # grade the screen score vs forward returns
│   ├── output_validation.py                  # sanity-check ranker targets vs price/HV
│   ├── data_reconciliation.py                 # flag disagreeing data sources pre-LLM
│   ├── macro_filter.py                       # deterministic macro-regime veto on picks
│   ├── report.py                            # public re-exports (shim)
│   ├── report_sections.py                   # Section IR + parsers + palettes
│   ├── report_html.py                       # HTML email renderer
│   └── report_pdf.py                        # ReportLab PDF renderer
├── data/            # Provider adapters (yfinance, FinnHub, FRED, SEC EDGAR,
│                    #                    SnapTrade, Tavily, chart-img)
├── agents/          # Standalone agents (insider, news reranker, portfolio)
├── reporting/       # SMTP + analyst-report HTML renderer
├── llm.py           # AgnoAgent factory (Claude + Gemini + OpenAI) + provider fallback
├── http_client.py   # Shared retry / rate-limit HTTP client
└── preflight.py     # Fail-fast startup checks
```

**Hybrid LLM + deterministic-math pattern:** the LLM picks WHICH lots
to sell with reasoning; `tax_lot_helper.py` computes the actual
realized P&L, treatment, and tax dollars. Same pattern for EV (Sizer
gets pre-computed `Σ(p × return)` instead of doing arithmetic itself).
This eliminates the hallucination class where the LLM invents numbers
that don't match the data.

**Anti-hallucination layer:** market themes are validated against
the actual universe + RS data (`_validate_and_correct_themes`); a
verdict auto-repair pass (`_repair_verdict_inconsistencies`) rewrites
SELL/TRIM verdicts that contradict their own prose; structured
Pydantic outputs everywhere so every LLM stage is a field read, not
a regex.

## Outputs

- **HTML email** (`reporting/smtp.py`) with inline chart images via `cid:` refs
- **PDF attachment** (ReportLab) saved locally before send so an SMTP
  outage never costs the report
- **SQLite** (`discover.db`) — every run + candidates + picks +
  holdings reviews persisted for cross-run track-record measurement

## Track record

BUY / HOLD / TRIM / SELL decisions are all scored against SPY. Alpha is
sign-flipped for sells so positive alpha always means "the call was right":

```
BUY  alpha = stock_ret - spy_ret  (stock beat SPY → wise buy)
SELL alpha = spy_ret - stock_ret  (stock lagged SPY → wise sell)
```

Three things the measurement is careful about, because this number is fed
back into the ranker prompt (every consensus round, every provider) as the
system's own accuracy:

- **One horizon per number.** Each decision is measured over *completed*
  fixed windows (30d and 90d) and only aggregated with decisions measured
  over the same window. Blending a 15-day outcome into the same mean as a
  90-day one made the statistic track how recently you'd run the pipeline.
  Decisions younger than 30 days show as "pending" with a live mark.
- **Nothing is silently dropped.** A decision with no forward price data is
  usually a delisting (the worst outcome a BUY can have) or a bad symbol;
  those are counted and reported rather than removed, because dropping them
  deletes the left tail and inflates measured alpha.
- **Beta is not skill.** The screen selects high-beta momentum leaders by
  construction, so every row also carries `beta-adj` alpha
  (`ret - β·spy_ret`), with β estimated only on pre-decision data.

## Grading the system's own forecasts

```bash
uv run validate-screen                      # both checks
uv run validate-screen --what score --horizon 30
```

- **Screen score validation** — mean forward alpha by score quintile plus a
  Spearman information coefficient for every sub-component, computed from
  the scores already stored per run. A flat quintile curve means the
  composite isn't separating winners from losers; a negative IC means that
  component is pointing the wrong way. Measure before re-tuning weights.
- **Ranker calibration** — EV error (realized − EV) at the 270-day horizon,
  mean realized alpha bucketed by the stated conviction score, stated vs
  observed frequency for bull/base/bear, and the nearest past setups by
  factor similarity (from the screen's own `score_breakdown`). Conviction,
  EV, entry price, all three scenario probabilities, and which provider(s)
  voted for each pick are persisted per pick, and the calibration block is
  fed back into the ranker prompt so every round sees the system's own
  scorecard on the next run.

Both read point-in-time values stored at decision time, so neither can leak
the outcome into the feature. Prices are the only thing fetched
retroactively (a historical close is the same number today as it was then).

## Tests

```bash
uv run pytest -q
```

323 tests covering the high-stakes math (tax-lot computation, verdict
auto-repair, direction-aware and horizon-separated track-record alpha,
beta adjustment, score validation, forecast calibration, parsers,
section-dispatch parity HTML/PDF, multi-provider ranker consensus math,
cross-source data reconciliation, macro-veto rules). The full suite runs
in ~9s.

`tests/conftest.py` points `Settings` at no env file and blocks outbound
sockets for the whole suite, so a test can never read your real `.env` or
spend real API quota. A test that genuinely needs the network must be
marked `@pytest.mark.allow_network`.
