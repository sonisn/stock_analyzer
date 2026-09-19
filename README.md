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
deltas are estimated from IV (Black-Scholes). Cash is tracked per account: each put is placed
in one account that can secure it (`OPTIONS_ACCOUNTS` limits which), and the
cash the plan's own BUYs/ADDs spend is taken out first.

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
- **Ranker calibration** — EV error (realized − EV) after one year for 3-5
  year picks (whose EV is annualized; 270 days for older 6-12 month ones),
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

## Long-term horizon

Every holding and pick is treated as a 3-5 year investment. The ranker
states its bull/base/bear scenarios as annualized returns over 3-5 years
(so its EV is graded against the realized 1-year return), and price moves
alone never trigger a sale. A position down 20% from cost is flagged for a
thesis re-check, and a thesis counts as BROKEN only when analysts are also
cutting estimates. Every sell suggestion names where the money could go: a
recent discover pick you don't hold, or a same-sector swap for a tax-loss
sale.

## One price per ticker

Each brokerage quotes its own price for a position, and those feeds go
stale: on 2026-09-19 the HSA showed BE at $298.61 while the live quote
and the other two accounts said $265.63, overstating the portfolio by
$2,407. Valuation therefore uses the live quote the run already fetched,
falls back to the broker's price only where there is no quote, and puts a
"Check the data" line in the daily email when an account's price sits
more than 2% off the market. Without that, one holding could be worth two
different amounts in the same email, and the error went into
`portfolio_snapshots`, where it permanently skewed the vs-SPY return.

Unrealized P/L is measured only over positions that have both a value and
a cost basis, so a holding transferred in without one is no longer
counted as pure profit.

## Accounts

Cash only funds buys (and put collateral) in its own account, so the
rebalancer sees cash per account and names the account in every BUY/ADD.
Accounts are labelled by name; two with the same name get the institution
(or the id's last 4 characters) appended, so neither overwrites the other.
Non-USD cash balances are skipped rather than summed as dollars.

## Income and adding on dips

The daily email's Portfolio health shows **dividend income** — about how
much the holdings pay per year at current rates, what the last 12 months
paid, and whether it was reinvested automatically or left as cash (with a
suggestion for idle cash) — and **add on weakness**: holdings 15%+ below
their 52-week high whose long-term case is intact (no thesis flag, no
estimate cuts, sector under the cap, under 20% of the portfolio, not
already flagged for a thesis re-check or a loss sale). The rebalancer gets
the same kind of list for its ADD decisions.

## Daily email: long-term views and earnings results

Each holding's block is built in code from freshly fetched data — price,
52-week range, P/E, yield, analyst counts, earnings dates, trend — so the
numbers are current every morning. The one judgment in the block, the 2-3
sentence **long-term view**, is written by the model and stored, then
reused (shown with the date it was written) until it is
`STOCK_VIEW_MAX_AGE_DAYS` old (7), the price has moved
`STOCK_VIEW_MOVE_PCT` (8%) since, or the company has reported. For a 3-5
year case nothing is lost by not re-deriving it daily, and a portfolio of
25 holdings drops from ~50 model calls a morning to a handful.

**News is filtered, not just sorted.** A holding's Yahoo feed is mostly
syndicated commentary about other companies — measured on four holdings
(2026-09-19), 35 of 40 items, and all ten of NVDA's were about AbbVie,
Boeing, Iamgold and CoreWeave. So an item whose *headline* doesn't name
the company is dropped, the list isn't padded back to five, and headlines
an earlier email already carried don't come round again. Ranking the rest
is one batched model call for the whole portfolio (it used to be one per
holding), with a deterministic ranking as the fallback.

When that leaves a stock with nothing to read, the block says so and
shows what the company actually *did* instead, from sources that are
company-specific by construction and free: recent SEC filings (8-K /
10-Q, with links), where analysts have moved next year's EPS, and Form 4
insider activity. On 2026-09-19 that turned NVDA's five filler links into
an 8-K, a 10-Q and "42 up / 0 down" estimate revisions — and surfaced
$270.8M of insider selling at AVGO that no headline mentioned.

When a holding reports, the next day's Portfolio health carries the
result: the EPS line, and — the part that matters for a 3-5 year hold —
which way analysts have moved estimates since. Cuts raise an **EARNINGS
CUT** item for a thesis re-check and are logged to the suggestions ledger
as a REVIEW; steady or rising estimates say the long-term case holds.

## Screen price rules

`DISCOVER_TREND_GATE=soft` (default) only rejects names more than 40% below
their 52-week high; `strict` restores the old uptrend-only gate. Tested on
15 years of S&P 500 weekly prices (320k observations): names passing the
strict gate beat SPY by +3.4% over the next year vs +3.6% for those failing
it (t = 0.6, sign flipping year to year) — the gate narrowed the field
without improving it. (Current index members only, so crashed-and-dropped
names are missing — why the 40% falling-knife rule stays.) When more names
pass than `DISCOVER_MAX_SCREEN_CANDIDATES`, the soft gate keeps those closest
to the screen's ideal entry (10% below the high) rather than the strongest
momentum.

## Quarterly review

The review opens with **Your portfolio vs SPY**: the daily email stores the
portfolio's total value (holdings + cash) each weekday
(`portfolio_snapshots`), and the review chains those into a time-weighted
return — deposits, withdrawals and transfers (from the brokerage's activity
history) are taken out — for last quarter, year to date and since tracking
began, next to SPY over the same dates.

Advice is kept in the `suggestions` table: the daily email's action lines
(sell, tax-loss sale, thesis re-check) and every rebalance action. On the
first trading day of each quarter `uv run quarterly-review` (cron via
`scripts/run_quarterly_review.sh`) emails how last quarter's advice
worked out — each suggestion and discover pick against SPY, sales against
their suggested replacement, and whether you acted on it — followed by
today's portfolio health. `--force` runs it any day; `--print` prints the
HTML instead of emailing.

## Year-end tax planner

On December's first trading day `uv run tax-planner` (cron via
`scripts/run_tax_planner.sh`) emails, for taxable accounts only: realized
gains so far this year (estimated FIFO from each account's own purchase
lots — SnapTrade sells carry no cost basis — with closed option contracts
netted separately and transferred-in shares flagged as basis unknown), the
losses available to harvest and how far they offset those gains plus the
$3,000 of ordinary income, and short-term lots in profit that turn
long-term within 60 days (wait to sell). `--force` / `--print` as usual.

## Stored data and database size

Besides run history, the database keeps small, reusable reference data so
runs don't re-download it:

| Table | What | Growth |
|---|---|---|
| `brokerage_activities` | every SnapTrade activity (compact, ~300 bytes); synced incrementally — only what's new since the last stored date per account | a few hundred rows/year, kept forever |
| `ticker_reference` | sector / industry / name (refreshed after 30 days) and next earnings date (after 3 days, or once it has passed) | one row per stock, overwritten; unused rows dropped after 365 days |
| `stock_views` | the daily email's latest long-term view per stock, plus the headlines already sent | one row per holding, overwritten (links capped at 40); dropped 90 days after the last refresh |
| `portfolio_snapshots`, `suggestions` | daily value, advice ledger | one row per day / per advice |

Tax lots, cash flows and dividends read the stored activity history, so a
purchase older than the brokerage API's window no longer drops out of lot
splits or FIFO matching. History upkeep trims old prose (365 days), compacts
the file (VACUUM) once trimming frees 20% of it, and the monthly
`model-review` email reports the database size and largest tables, flagging
it past `HISTORY_DB_WARN_MB` (50 MB).

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
