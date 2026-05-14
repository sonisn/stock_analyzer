# stock-analyzer

A personal portfolio analyzer that runs two pipelines on top of Claude
(Opus + Sonnet), market data, and brokerage holdings:

- **`discover-stocks`** — surface 5 medium-term picks from a screened
  universe. Sonnet writes per-ticker analyst reports, Opus ranks them
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
| Universe | — | Pull tickers from S&P 500 + watchlist + holdings |
| Fundamentals / Technicals / EPS revisions / Sector rotation / Macro | — | Parallel data fetches (yfinance, FRED, FinnHub) |
| Track record | — | Score past BUY picks + SELL/TRIM calls against SPY |
| Market themes | Sonnet | Identify 3-8 themes grounded in actual price + revision data |
| Screen | — | Hard filters + 0-100 composite score |
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
provider (real-time bid/ask + Greeks like delta/IV, courtesy of ORATS).
Set `TRADIER_API_KEY` in your `.env` to enable — a free Tradier brokerage
account (no funding minimum) gives you a production access token at
[dash.tradier.com](https://dash.tradier.com/). If `TRADIER_API_KEY` is
unset or Tradier is unreachable, the pipeline falls back to **yfinance**
(free, 15–20 min delayed, no Greeks — Opus picks strikes via strike-vs-spot
proxy). Both work; Tradier gives meaningfully better strike selection
because the LLM gets accurate delta values for the Δ 0.35–0.45 band rule.

**IV timing signal:** when `ORATS_API_KEY` is set, the pipeline pulls
per-ticker IV rank (IVR-1y) from ORATS's `/datav2/ivrank` endpoint —
one batched call per rebalance, well within the free tier's 5 req/min.
IVR tells Opus whether IV is elevated (write more aggressively),
average, or depressed (skip the write unless conviction is high). Free
ORATS signup: https://orats.io/data-api.

See `.env.example` for the full set of `CC_*`, `TRADIER_*`, and `ORATS_*` knobs.

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
- `ORATS_API_KEY` — IV rank / percentile per ticker. Free tier (5
  req/min) is plenty for one batched call per rebalance. Without it,
  the LLM picks strikes by delta alone — no IV timing signal.
- `ORATS_BASE_URL` — defaults to `https://api.orats.io/datav2`.

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
│   ├── schemas.py / rebalance_schema.py     # Pydantic structured outputs
│   ├── persistence.py                       # SQLite layer
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

Both BUY picks and SELL/TRIM calls are scored against SPY over a
90-day window. Alpha is sign-flipped for sells so positive alpha
always means "the call was right":

```
BUY  alpha = stock_ret - spy_ret  (stock beat SPY → wise buy)
SELL alpha = spy_ret - stock_ret  (stock lagged SPY → wise sell)
```

Mature decisions (≥14 days old) flow into the aggregate stats; newer
ones show in a separate "pending" bucket. Delisted tickers (no
yfinance price) are dropped from output entirely.

## Tests

```bash
uv run pytest -q
```

66 tests covering the high-stakes math (tax-lot computation,
verdict auto-repair, direction-aware track-record alpha, parsers,
section-dispatch parity HTML/PDF). The full suite runs in ~3s.
