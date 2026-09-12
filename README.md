# stock-analyzer

A personal portfolio analyzer that runs two pipelines on top of Claude
(Opus + Sonnet), market data, and brokerage holdings:

- **`discover-stocks`** — surface 5 medium-term picks from a screened
  universe (S&P 500 snapshot + watchlist + holdings, with news coverage as
  a conviction overlay rather than the candidate pool). Sonnet writes per-ticker analyst reports, Opus ranks them
  with probability-weighted scenarios, a red-team Opus pass writes
  bear cases, and an Opus sizer allocates the new capital.
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
| Ranker | Opus | Top-N picks with 3 scenarios (bull/base/bear) + EV |
| Red-team | Opus | Bear case per pick with fragility rank + watch metric |
| Sizer | Opus | Allocate new capital; flag concentration / correlation |
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

See `.env.example` for the full set of `CC_*` and `TRADIER_*` knobs.

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

- `ANTHROPIC_API_KEY` — Claude (or `GOOGLE_API_KEY` if you swap provider)
- `DISCOVER_OPUS_MODEL` / `DISCOVER_SONNET_MODEL` — model IDs
- `FINNHUB_API_KEY`, `FRED_API_KEY`, `TAVILY_API_KEY` — data providers

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
│   ├── calibration.py                       # grade the ranker's own EV + conviction
│   ├── score_validation.py                   # grade the screen score vs forward returns
│   ├── report.py                            # public re-exports (shim)
│   ├── report_sections.py                   # Section IR + parsers + palettes
│   ├── report_html.py                       # HTML email renderer
│   └── report_pdf.py                        # ReportLab PDF renderer
├── data/            # Provider adapters (yfinance, FinnHub, FRED, SEC EDGAR,
│                    #                    SnapTrade, Tavily, chart-img)
├── agents/          # Standalone agents (insider, news reranker, portfolio)
├── reporting/       # SMTP + analyst-report HTML renderer
├── llm.py           # AgnoAgent factory (Claude + Gemini)
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
back into the Opus ranker prompt as the system's own accuracy:

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
  mean realized alpha bucketed by the stated conviction score, and stated
  vs observed frequency for bull/base/bear. Conviction, EV, entry price and
  all three scenario probabilities are persisted per pick, and the
  calibration block is fed back into the ranker prompt so Opus sees its own
  scorecard on the next run.

Both read point-in-time values stored at decision time, so neither can leak
the outcome into the feature. Prices are the only thing fetched
retroactively (a historical close is the same number today as it was then).

## Tests

```bash
uv run pytest -q
```

265 tests covering the high-stakes math (tax-lot computation, verdict
auto-repair, direction-aware and horizon-separated track-record alpha,
beta adjustment, score validation, forecast calibration, parsers,
section-dispatch parity HTML/PDF). The full suite runs in ~3s.

`tests/conftest.py` points `Settings` at no env file and blocks outbound
sockets for the whole suite, so a test can never read your real `.env` or
spend real API quota. A test that genuinely needs the network must be
marked `@pytest.mark.allow_network`.
