# stock-analyzer

A personal portfolio analyzer that runs two pipelines on top of Claude
(Opus + Sonnet for most stages, with Gemini and OpenAI in the mix for
ranker consensus and red-team), market data, and brokerage holdings:

- **`discover-stocks`** — surface 5 medium-term picks from a screened
  universe (every US-listed stock >= $2B that is tradable + watchlist + holdings, with news coverage as
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

Open models over OpenRouter do the **reading**, never the deciding:
GLM-5.3 and GLM-5.3-Flash turn every $2B+ company's latest 10-Q/10-K and
earnings press release into quoted facts for the deciding models (see
[SEC filings, read by open models](#sec-filings-read-by-open-models)),
and GLM-5.3 writes the insider summary and ranks the news.

## What's in it

| Stage | Model | What it does |
|---|---|---|
| Universe | — | Sampling frame (US stocks >= $2B, filtered for tradability, + watchlist + holdings) plus a news-derived conviction overlay |
| Fundamentals / Technicals / EPS revisions / Sector rotation / Macro | — | Parallel data fetches (yfinance, FRED, FinnHub) |
| Track record | — | Score past BUY/HOLD/TRIM/SELL calls vs SPY at fixed 30d + 90d horizons, beta-adjusted |
| Market themes | Sonnet | Identify 3-8 themes grounded in actual price + revision data |
| Screen | — | Hard filters + 0-100 composite score (45 fundamentals / 45 trend / 10 attention) |
| Enrichment (parallel) | — | News, earnings, insider selling, share trades, peers, SEC filing facts (read weekly by GLM-5.3), transcripts |
| Analyst | Sonnet | Per-ticker analyst report with structured output |
| Ranker | Claude + Gemini + OpenAI (one consensus round each) | Top-N picks with 3 scenarios (bull/base/bear) + EV + agreement ratio |
| Macro veto | — (deterministic) | Suppress high-momentum picks in a risk-off FRED regime |
| Red-team | Gemini (default) | Bear case per pick with fragility rank + watch metric |
| Sizer | Opus | Allocate new capital; flag concentration / correlation; weights by consensus agreement |
| Holdings review | Sonnet | HOLD / TRIM / SELL per position with tax-lot plan; sees its last verdict on the holding and must name new evidence to change it |
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
# Install (Python 3.14+, uses uv). `--extra dev` keeps pytest, ruff + ty;
# a plain `uv sync` removes them.
uv sync --extra dev

# Configure — copy and fill in your keys
cp .env.example .env

# Run
uv run discover-stocks            # find new picks
uv run rebalance-portfolio        # review holdings + plan
uv run analyze-portfolio          # one-off analyst-style report
uv run analyze-insiders           # insider + political trade signals
uv run read-filings               # SEC filings -> facts (GLM-5.3 on OpenRouter)
uv run dashboard --open           # one HTML page of everything on record
uv run ops doctor                 # free check of every key, model id and source
uv run stock-analyzer --help      # every command in one list
```

`discover-stocks --help` and `rebalance-portfolio --help` print their
usage without starting a run; with no flag, both start a live (paid) run.
A run whose step fails stops there and exits 1; each step's timing and
summary (or error) is kept in the `pipeline_steps` table.

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

For the SEC filing reader and the helper roles (insider summary, news
ranking):

- `OPENROUTER_API_KEY` — open models over OpenRouter
  (`OPENROUTER_READER_MODEL`=z-ai/glm-5.3, `OPENROUTER_BULK_MODEL`=
  z-ai/glm-5.3-flash, `OPENROUTER_DAILY_CAP_USD`=2 across every process)
- `RERANK_PROVIDER` / `INSIDER_PROVIDER` = `openrouter` with
  `RERANK_MODEL` / `INSIDER_MODEL` = `z-ai/glm-5.3` puts those roles on
  GLM-5.3 (the insider summary falls back to `LLM_PROVIDER` if OpenRouter
  fails or its report names a ticker the sources don't support). Leave
  them empty to keep them on Claude.
- `LLM_PRICES` — prices for any model the built-in table lacks
  (`model=in:out` USD per million tokens); GPT-6 and Gemini Pro are built
  in, so the cost cap and the run totals count them.

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
- `DISCOVER_MAX_SCREEN_CANDIDATES` (600) — cap on names that reach the
  per-ticker fundamentals + EPS fetches, after the trend gate
- `FETCH_CACHE_DAYS` (8) / `FETCH_CACHE_DIR` (`~/.stock_analyzer/cache`) —
  per-ticker fundamentals and EPS revisions (`data/fetch_cache.py`, one
  small JSON file per kind, latest answer only). An answer is dropped the
  day after the company reports (it carries its next earnings date) and
  otherwise kept up to 8 days; price-dependent fields (market cap, P/E,
  FCF yield, target upside) are rescaled to the latest stored close. The
  nightly `earnings-watch` job runs discover's trend gate and cap and
  renews the oldest fifth of those names plus any new or just-reported
  ones (~120-200 requests a night instead of ~1,200), so discover and
  rebalance ask Yahoo only for names they have never seen. Fundamentals
  are one request per name (`info`, with trailing-12-month operating cash
  flow); a cash-burning company costs two more for the dilution check.
  `0` turns it off
- `YF_BARS_DIR` (`~/.stock_analyzer/cache/bars`) — the on-disk daily-bar
  store (`data/bar_store.py`), one Parquet file per symbol. Outside market
  hours, bars synced after the last close are served with no request;
  otherwise only the days since the last stored bar are downloaded, and a
  new dividend or split (which rewrites adjusted history) downloads the
  symbol's whole history again. `off` disables it. Files no run has read
  in 90 days are pruned.

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
│   ├── answer_checks.py                     # rules each stage's answer must keep
│   │                                          # (sent back to the model once if broken)
│   ├── review_memory.py                     # the Reviewer's last verdict per holding
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
├── llm.py           # ModelAgent on Pydantic AI (Claude + Gemini + OpenAI): settings,
│                    #   cost cap, output ceiling, answer checks, provider fallback
├── openrouter.py    # Open models over OpenRouter, also on Pydantic AI: approved
│                    #   hosts only, daily cap, billed-cost ledger
├── pipeline.py      # Step runner for discover/rebalance (step log in pipeline_steps)
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
a regex — returned as the provider's native JSON-schema output (Claude
keeps its thinking; a forced tool call would switch it off) and sent back
to the model once with the validation errors when it doesn't fit.

**Answer rules** (`discover/answer_checks.py`): each deciding stage's
answer is checked against facts the run already holds — the Ranker picks
only from the candidates analysed (once each, as many as asked, ranks
1..n, bull/base/bear scenarios), the Red team writes one bear case per
pick, the Sizer allocates only to the picks, the Reviewer's TRIM/SELL
needs confidence ≥ 7, and the Rebalancer sells, trims or writes calls
only on positions held. A broken answer goes back to the model once with
the problems listed; one still broken is kept and the deterministic
safeguards downstream decide, so a rule never costs a stage its answer.

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

## Fifteen-year factor study

```bash
uv run factor-study                      # both parts (~10 min first time, SEC download)
uv run factor-study --part risk          # prices only
uv run factor-study --part fundamentals --refresh-sec
```

The live record grows by one cohort a month, so it takes years to grade a
screen rule. This study grades them on month-ends since 2012 instead, using
only free data and no LLM calls:

- **Fundamentals** — each of the screen's measures (and its 0-45 points and
  hard rules, computed with `screen.py` itself), plus gross profitability
  and the earnings surprise, read point-in-time from SEC company facts:
  only what had been *filed* by each month-end counts
  (`model/sec_history.py`; files cached 30 days). Market cap uses the
  actually traded close (the bar store's are split- and dividend-adjusted).
  Financials are excluded.
- **Risk** — whether volatility persists, the low-volatility quintiles, and
  monthly equal- vs inverse-volatility-weighted baskets against SPY.

The universe is today's S&P 500, so names that fell out are missing; that
flatters high-volatility and turnaround names most. The report is printed
and saved under `~/.stock_analyzer/reports/`.

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

A price that far off the market is usually the symptom, not the problem.
SnapTrade reports when each broker last refreshed an account, and on
2026-09-20 the HSA's last successful holdings sync was 2026-07-06 — 75
days earlier — so its share counts, cash and transaction history were
frozen at July's values too, not just its prices. Any account that has
not synced in `STALE_SYNC_DAYS` (4, so a long weekend passes quietly) now
leads the daily email's "Decide today" list and the rebalance "At a
glance" flags with the date it went dark and the one fix for it:
reconnect it in SnapTrade. Until then, every number for that account
describes the day it stopped syncing.

## Covered calls are part of every decision

Twelve short calls were open on 2026-09-20 and only the rebalancer knew:
`fetch_open_option_positions` answers "how much call capacity is left",
which is the writer's question. Every *sell* decision has a different
one — how many of these shares are already promised, at what price, and
until when — and nothing was asking it. Three positions (GOOGL, NVDA,
TSLA) were 100% committed while the holdings table showed them as freely
owned.

`fetch_covered_call_obligations` answers the seller's question, and the
daily email now carries it two ways. Every sale-shaped suggestion —
broken thesis, drawdown re-check, tax-loss harvest, past-its-target trim
— gains a clause naming the contracts, the share of the position they
cover and the strike, because selling shares that back a call turns it
naked: the position has to be bought back first, or assignment has to run
its course. And a "Covered calls written" table shows what is promised
with its distance to each strike.

A position within `ASSIGNMENT_WATCH_PCT` (15%) of its lowest strike also
becomes its own decision line, since for a 3-5 year holder assignment is
not a loss but the end of the compounding — and in a taxable account it
realizes the gain on someone else's schedule. On 2026-09-20 that was TSLA
at 10% below a $400 strike, with all 200 shares committed. It is graded
as a REVIEW in the suggestions ledger, not as a trade.

## What the written options earned

Premium shows up nowhere in performance: the shares are valued at the
market, the cash lands as cash, and a called-away position simply
disappears at the strike — so a winner taken by assignment grades as a
sale into strength and the premium that paid for it is invisible. The
quarterly review now carries an "Options written" section from the
activity ledger (`data/options_income.py`), with the net premium per
underlying and the shares that left at each strike.

Three distinctions make the number honest, and each one came from the
real ledger. A contract sold to OPEN is income; one bought to open is a
bet (a long NFLX call lost $8,900 and belongs nowhere near premium).
Which one it is cannot be read from the sign of a single row, only from
which trade came first — folding in arrival order turned four long
positions into "228 short contracts". And a contract opened and closed on
the same day gives no ordering at all, so those are reported apart from
both. Contract counts track the peak short position, not the largest
single fill: NVDA's December $285 calls were sold -3 then -1, which is
four contracts short.

The tax-loss harvester also subtracts promised shares now. Selling shares
that back a call turns it naked, so only the free shares above the
committed ones are harvestable, counted per account — a call written in
one account promises nothing in another.

`OPTIONS_ACCOUNTS` gates covered calls as well as cash-secured puts. It
gated only puts before, so there was no way to keep either out of an
account that cannot trade options. Empty still means every account.

An account outside that list is treated as a blocked opportunity rather
than an absent one. The Schwab HSA holds 73 uncovered BE shares and no
options approval, so suggesting a call there would be an instruction that
cannot be followed — but silence would read as "nothing to do". Add-on
ideas name only accounts that could act, while a separate line says what
the paperwork is worth: *"HSA Brokerage ...263 is not approved for
options, so BE is 27 shares (~$7,172) from a writable lot — the premium
is behind an options application, not behind the market."* One line per
account, since it is one form.

## Turning off new-stock suggestions

`DAILY_EMAIL_NEW_IDEAS=0` stops the daily email proposing stocks you do
not own: the "Ideas for new money" blocks, the "reinvest the proceeds in
X" tail on a sale line, and the same-sector swap named beside a tax-loss
candidate. Every action on a current holding is untouched — sells,
drawdown re-checks, tax-loss candidates, covered-call rolls, assignment
warnings, add-on-weakness.

The reason to use it: those names come from the stored pick pool, so they
are as old as the last `discover-stocks` run and were chosen under
whatever settings were in force then. When the two have drifted apart,
the daily email should not be the thing sourcing new positions.

## A suggested stock gets the same look as a held one

A holding comes with a chart, trend labels, a 52-week range and a
valuation. An idea arrived as a ticker and one sentence — enough to
recognize, not enough to act on. `attach_idea_details` fetches the same
yfinance snapshot a holding's block is built from for every stock the
report proposes buying, and charts are requested for them alongside the
holdings and referenced by the same CID scheme, so the image inlines the
same way.

No model call is involved: an idea is worth a chart, not another round of
tokens. The reason the ranker gave for the pick heads the block, then
price, today's move, the 52-week range, P/E, analyst target, dividend and
the 1/3/6/12-month trends.

## The contracted book

Everything else forward-looking in the pipeline is somebody's opinion:
analyst targets, forward P/E, EPS revisions. Remaining performance
obligations are not. They are signed orders a company has told the SEC it
has not delivered yet, tagged in every 10-Q and 10-K as
`RevenueRemainingPerformanceObligation` and free through the same EDGAR
client `sec_edgar` already uses.

It matters because price and book can say opposite things. On 2026-09-20
the daily email called AVGO's thesis BROKEN — below its 200-day average,
lagging SPY by 17 points, analysts cutting EPS — while its book had gone
$45.0B → $164.6B → $179.2B over two quarters, +552% over a year. POWL was
flagged at -26.4% and offered as a tax-loss sale with its book up 71%.
Both may still be sells; neither should be sold without that in view.

So every sale-shaped decision now carries the book when it has moved more
than `BACKLOG_MATERIAL_PCT` (10%) — in whichever direction it points. A
growing book is the case for waiting; a shrinking one is the strongest
confirmation a sale can have, and leaving that out would make it a
bull-only footnote. A "Contracted book" table lists the holdings that tag
it, fastest-growing first.

Discovery sees it too: a `contracted_book` step runs in the enrichment
block on the screen's survivors, and the figure goes into the Analyst
payload with an instruction to weigh it above analyst targets when the
two disagree — while treating `null` as "not disclosed" rather than
evidence against a name. On a 25-name survivor sample, 10 tagged a book,
led by DELL +200% and CRWD +49% over a year.

Every fact carries the date it was **filed**, so `as_of` gives the book
as it was known on a past date — point-in-time by construction, which
yfinance's forward estimates are not. Coverage is partial by nature:
ANET stopped tagging it in 2022, and banks and retailers never do, so it
is evidence where present and never a filter that penalizes absence. A
company whose newest fact is older than `MAX_FACT_AGE_DAYS` (200) is
treated as no longer reporting it rather than shown a stale book.

## Why this stock, and what the market is rewarding

A reinvestment line read "reinvest the ~$57,770 in A (pick #2,
2026-09-17, Healthcare)" — a ticker and no argument. The argument existed
the whole time: the ranker wrote one when it chose the name, and it was
stored in `run_outputs.ranker_full` and never read back. `pick_headline`
reads that sentence out, so the line now ends *"— Agilent provides
defensive life-sciences exposure with accelerating estimates and a newly
expanded diagnostics footprint via the Biocare acquisition."* It is the
model's own sentence, not a paraphrase: inventing a reason later would be
attributing one it never gave.

Each idea also carries whether its sector is leading or lagging, from the
same six-month rotation data the discover pipeline already computes. A
pick chosen for balance is often deliberately outside what is working, so
both facts belong in one sentence — the choice then reads as a trade-off
rather than an oversight.

## Calls written to be kept

The covered-call bands were 0.35-0.45 delta over 30-45 days: roughly a
40% chance of losing the shares, renewed every six weeks, on positions
held for three to five years. They are now **0.10-0.25 delta over 60-120
days**, with two rules the premium cannot argue with:

- `CC_MIN_UPSIDE_PCT` (15%) is a hard floor under the strike, independent
  of delta — never cap a position closer than that to today's price,
  however rich the bid.
- `CC_MIN_IV_HV_RATIO` (1.0) only writes when the options market is
  paying more than the stock's own realized volatility. Below it the
  upside is being underpaid and the report says so by name rather than
  going quiet: *"NVDA: IV 31% is only 0.80x its realized 39% (depressed)
  — below the 1.00x floor. Wait for a volatile session."*

All of this is enforced in `validate_option_writes`, not asked for in the
prompt. The put path already re-read every number from the chain while
the call path checked eligibility alone, so a 0.60-delta write a month
out would have passed validation untouched.

Longer expiries are preferred but not unboundedly: a contract 400 days
out is rejected the same as one 30 days out, because it caps the position
for a year. Chain rows carry **premium per day** beside the quote, since
"further out pays more" is true per contract and false per day — $400
over 30 days is $13.33/day, while $1,000 over 120 days is $8.33/day.

### Rolling one the stock has caught up with

"Roll it up and out" is the right instruction and a useless one alone: at
which strike, into which expiry, and does it still pay after buying the
near call back? `discover/cc_roll.py` computes it — buy-back priced at
the **ask** and the replacement sold at the **bid**, the sides actually
available, so nothing that only works at mid prices survives. The
replacement must clear the same 15% floor and delta ceiling a new write
would, and lift the strike at least 5%, since raising a cap by a couple
of percent is churn dressed up as risk management.

Among rolls that pay for themselves it picks the **soonest**, not the
largest credit. Credit grows with time to expiry, so ranking on it alone
always answers "sell a 2028 call" — the most money and the longest
surrender of decisions. When nothing inside the writing window covers the
buy-back, the longer-dated one that does is offered with the lock-in
stated rather than buried, and if nothing pays at all it says so instead
of inventing a trade.

The assignment warning in the daily email carries the costed roll, so
today it reads: *"TSLA is 10% below your $400 strike expiring 2026-12-18:
a rally through it calls away 200 shares. Nothing in the usual 120-day
window pays for buying the TSLA $400 call back. Roll the 2 TSLA $400
call(s) up to $600 2027-09-17, 273 days further out for a net credit of
$110. That lifts the cap from +10% to +65% above today's $364.27 and
keeps the shares, capped until 2027-09-17."* Chains are fetched only for
positions already near their strike.

## Where new money buys a second payoff

A part-lot earns nothing: 62 uncovered AVGO shares are 62 shares of
upside, while 100 are a contract. `call_headroom` measures, per account,
how many shares back no call and how far the position is from the next
writable lot, so an add-on idea can say "45 more ARM shares (~$12,402)
would complete a round lot you could write another covered call against".
A position already holding a full uncovered lot becomes its own decision
line, since that is premium available on shares already owned.

Headroom is limited to holdings a call can actually be written on. Before
that filter, SPAXX offered 208 contracts, the 401(k)'s commingled pool
offered 40 shares, and Taronis Technologies — whose SEC registration was
revoked in 2023 — offered one.

A money-market fund is excluded on its own evidence rather than on the
caller remembering to pass a quote type: `is_cash_like` catches a NAV
pinned to $1.00 and a list of sweep symbols. SPAXX came back the moment
it was read through a path that passed no quote types — which the
quarterly review does.

## Point-in-time fundamentals

The forward-return model is price-only on purpose: a historical close is
the same number today as it was then, while yfinance's fundamentals are a
live snapshot with no history, so training on them teaches the model
figures that had not been filed yet.

Wisesheets' `asof:` selector with `asReported=true` removes that. The
check that matters: with as-reported off, `asof:2025-05-01` returns
NVDA's quarter ending 2025-04-27 — filed 2025-05-28, four weeks after the
as-of date. With it on, the same request returns the quarter ending
2025-01-26, filed 2025-02-26, which is what an investor could have read.

`uv run train-model --fundamentals` adds four ratios (gross margin, net
margin, leverage, return on equity) to the weekly training set. Three
details make them usable:

- **Ratios only.** In as-reported mode the newest filing on a date is a
  10-Q for one company and a 10-K for another, so revenue means a quarter
  here and a year there. A margin from inside one filing is comparable.
- **Reported tags only.** The API's own `debt_to_equity` and `roe` are
  calculated fields and come back empty in this mode, as does
  `total_debt`. Across ten holdings, `total_assets` covered 10/10 and
  `total_liabilities` 9/10 while `total_equity` covered 3/10 — so
  leverage is liabilities over assets, and equity is what is left of the
  assets when the filing never tagged it.
- **Monthly sampling, carried forward.** Filings land quarterly, so
  twelve as-of dates a year carry the signal at a twelfth of the request
  budget, and each value is held forward until the next filing — never
  interpolated, never pulled back from a month that had not happened.

Each as-of date costs one request per 100 tickers, so the S&P 500 over
five years is ~300 requests; results are cached per date, so a retrain
spends nothing, and the fetch stops early rather than draining the
month's quota. A ratio a company never tagged becomes that date's median
rather than dropping the company from the training set.

## World markets

The macro context was US-only — FRED's yield curve, VIX and jobs, plus
relative strength against SPY — while the demand behind these holdings is
priced overnight on other exchanges. Taiwan sets the tone for TSM and for
the foundry capacity behind NVDA and AVGO, Korea prices the memory cycle,
the dollar decides what foreign revenue translates to, and copper reads
industrial and grid demand for POWL.

`data/world_markets.py` fetches sixteen markets east to west (Nikkei,
KOSPI, Taiwan, Hang Seng, Shanghai, Sensex, DAX, FTSE, Euro Stoxx, S&P,
Nasdaq, SOX, plus the dollar index, USD/JPY, copper and crude) with
trailing 1d/1mo/6mo/1y changes, all free through the same paced yfinance
gateway. They appear as a table at the end of the daily email's health
block and are appended to the Ranker's macro context.

Each market declares which holdings it actually bears on, so the block
says "Taiwan Weighted +41% over six months — foundry capacity: reads
across to NVDA, TSM" rather than reciting indices. The trailing columns
come first and the overnight move last, deliberately: one session is not
a reason to touch a 3-5 year position. A market down more than 10% over a
year is called out as a regime break.

## Fundamentals as filed

yfinance's fundamentals are derived, undated, and sometimes wrong in ways
the response gives no hint of: on 2026-09-20 it put NVDA's trailing free
cash flow at $41.8B against $127.0B in the filings, and GOOGL's at
$22.7B against $53.3B. Those numbers drive the screen's scoring.

`data/wisesheets.py` fetches the same figures as filed with the SEC —
every value carries its XBRL tag, accession number and a link to the
filing — and `batch_fundamentals` overlays them on the yfinance row.
Anything forward-looking (estimates, price targets, recommendations,
short interest) has no counterpart there and is untouched.

The filed figure wins, with two exceptions. When the two sources are too
far apart to both describe the same company — ANET's 2025-12-31
`NetIncomeLoss` arrives as -$2,556M, turning a 38% net margin into 5% —
neither is trusted: yfinance's value is kept and the disagreement is
reported. And when the newest filing is more than `FILED_MAX_AGE_DAYS`
(150) old, yfinance's trailing figures are the more current answer.

Coverage is US SEC filers, so a foreign private issuer like TSM has
nothing to cross-check against. Those rows get a plausibility check
instead: a debt/equity above 20 means the balance sheet is denominated in
the company's own currency (TSM reads 42.16 because yfinance reports it
in TWD), which is worth knowing before comparing it to a US peer.

Set `WISESHEETS_API_KEY` to enable it. Without the key, or if the API
fails, every number falls back to yfinance exactly as before. The free
plan allows 5,000 requests/month and 100 tickers per request, so a full
S&P 500 pull costs about six.

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

## Earnings standouts and the six-month scorecard

`earnings-watch` runs nightly (10 PM New York, no LLM, no email) and looks
across the whole US market for companies whose results **clearly beat**
(EPS 5%+ and revenue 1%+ over estimates, $250M+ quarterly revenue), that
the market **rewarded** (3%+ better than SPY over the two sessions around
the report) and whose analysts then **raised** next year's EPS estimate
(3%+ over 30 days, checked a week after) — and only when it is **not a
lone blip**: it also beat the quarter before, and revenue is above the
same quarter a year ago (Yahoo's EPS history and quarterly income
statement, fetched only for names that passed everything else). One
quarter rather than a year on purpose: a company that has just turned the
corner is what a six-month idea is looking for, and the daily analysis
digs further once the name is in front of it. One Finnhub earnings-calendar
request lists every reporter; the price reaction comes from the bar store
the revision from Yahoo's `eps_trend` and the track record from Yahoo's
earnings history, so a typical night makes a handful of requests. Only beats and past picks are stored
(`earnings_events`), and a missed night is caught up.

The price is the tiebreak because sources disagree on what "EPS" is:
Finnhub and FMP gave Costco's September 2026 quarter as a miss and a beat.
On its first run TD SYNNEX beat EPS by 20% and revenue by 13% and fell 9%
against SPY — dropped.

For every report it is still following, the job also stores what Wall
Street's analysts did (`analyst_actions`, from Yahoo's upgrade/downgrade
log: firm, rating, old and new price target), from 90 days before the
report on. A standout's row in the email says what they did since —
"5 raised, 1 upgrade, targets +39% avg" — with the actions listed, or
"none" when no analyst has touched it yet.

A confirmed standout shows in the next few daily emails with the numbers,
a chart and a long-term view (one model call per new standout, reused
after), whatever `DAILY_EMAIL_NEW_IDEAS` says, and joins the discover
universe for 60 days — eligible like a watchlist name, with no score
bonus, so the screen judges it on the same terms.

Each standout the email shows (and you don't hold) is recorded in the
suggestions ledger as a `STANDOUT`, graded by the quarterly review, and
graded again six months on in the daily email's scorecard.

The daily email's **scorecard** grades every discover pick and every
standout six months (126 trading days) on against SPY, one row per month,
a ticker picked or shown on several days of the same month counted once.

The same nightly job then stores that day's analyst forecasts for every
tracked stock (`forecast_snapshots`, about 3.5 minutes of paced Yahoo
requests). Yahoo keeps only 90 days of estimate history; this keeps all
of it.

### Insider buying — and what was tested and left out

The history test covers S&P 500 members; the nightly check covers the
S&P 500 plus every tracked stock (holdings, a year of picks, recent screen
survivors, standouts) — about 520 companies. It checks each for
open-market purchases by insiders (Form 4, code P, from Finnhub — about
ten minutes at the free tier's pace) and stores them (`insider_buys`).
When two or more different insiders have each bought $10,000+ within 90
days (the floor drops plan and dividend-reinvestment buys, and made the
history test slightly stronger), the
daily email shows it the day the cluster forms, discover treats the stock
as eligible (no score bonus), and the suggestions ledger records it so
the quarterly review and the six-month scorecard grade it.

Everything added here was tested on 2015-2026 history first (S&P 500
current members, month-ends, point in time by SEC filing date, forward
6- and 12-month return vs SPY, t-statistics corrected for overlapping
windows):

| Signal | 12-month result | Verdict |
|---|---|---|
| 2+ insiders buying $10,000+ each in 90 days | +9.5% vs SPY on average (median +3.0%), against +3.2% (median −1.4%) with none; IC +0.020, t 2.07 at 6 months / 1.85 at 12, same sign both halves | **used** — as eligibility and a graded idea, not a score, until live results confirm it |
| Net share issuance (buybacks vs dilution) | IC −0.030, t −1.2; the heaviest issuers did best | not used |
| Analyst target raises / upgrades (3 months) | IC −0.01 to 0.00, \|t\| < 1.4 | display only |

Current-member samples flatter every group equally (stocks that fell out
of the index are missing), so the gaps between groups are the result,
not the levels.

### Web search: Exa first, Tavily second

News, catalyst, transcript and coverage searches go through one client
(`data/web_search.py`): Exa first (its $10 monthly credit is ~1,400
searches with article text, at most 10 requests a second), Tavily's free
1,000 a month when Exa's credit is spent or a call fails, and Finnhub for
company news when both are out. Nothing is billed past the free tiers.
Each run logs one line of what it searched and spent. If Exa calls fail
with a connection error, check the AdGuard allow rule
(`@@||exa.ai^$important`) first.

`ops doctor` runs Sunday evenings from cron (`scripts/run_doctor.sh`) and
alerts on a revoked key, a failed login, a retired model id, Yahoo
prices or estimates no longer coming back, or a disk running low (under 10% free, or under 50 GB) before Monday's paid runs.

## The universe: every US stock worth $2B+

Discover screens every US-listed stock with a market cap of $2B or more
that passes six business-quality rules, applied by Yahoo's screener itself
(two requests for the whole market instead of one per company):

| Rule | Threshold |
|---|---|
| Revenue growth, latest quarter vs a year before | ≥ 8% |
| Revenue growth, last 12 months | ≥ 8% |
| Total debt / equity | ≤ 2 |
| Operating cash flow, last 12 months | > 0 |
| Free cash flow, last 12 months | > 0 |
| Return on equity | ≥ 10% |

On 2026-09-27 that was 395 of 2,146 (824 with only the first, third and
fourth, the hard filter's own). Then the traps go, judged on the screener's
own fields: OTC listings, anything but common stock (funds, SPACs and
shells, preferreds, warrants, units), prices under $5 for companies under
$10B, under $10M a day of trading, and less than a year of history.
Foreign listings (TSM, ARM) and mid-caps (DINO) are in, which the S&P
indexes exclude; pre-profit companies (OKLO) are not, unless they come in
as a holding, watchlist name, earnings standout or insider cluster.

The nightly `earnings-watch` job rescans (the numbers change every
earnings season) into `~/.stock_analyzer/us_2b_universe.txt`, outside the
repo so the refresh never blocks the auto-update; `uv run ops universe`
does it by hand. The rules are `QUALITY_RULES` in `data/universe_scan.py`.
Without that file a run falls back to the bundled snapshot
(`data/static/us_2b_universe.txt`, 1,895 names taken before the quality
rules, so slow). `DISCOVER_UNIVERSE=sp500` switches back. The
forward-return model still trains on the S&P 500, where it was built and
validated, and the nightly insider check stays on the S&P 500 plus
tracked stocks.

A stock that enters the screen some other way (an earnings standout, an
insider cluster, a holding) meets the same bar in the hard filter:
positive trailing-12-month operating and free cash flow and a return on
equity of 10% or more, besides the growth, debt and analyst rules. (Until
2026-09-27 a company burning cash could pass on 2+ years of runway and
at most 20% dilution; that exception is gone with the quality rules.)

The 25 analysis slots also take at most **5 names per sector and 2 per
industry** (`screen.diversify_shortlist`): on 2026-09-27 seven of the 25
were energy and four of them refiners — one bet on refining margins
analyzed four times. The next-best names take the freed slots, and if too
few sectors pass to fill 25, the skipped names come back in rank order.

Mid-caps behave differently from large caps, so the screen adds two rules
for the bigger universe. A candidate needs **3+ analysts** (estimates and
revisions from one or two are an opinion, not a consensus). And the 25
analysis slots go by each candidate's **percentile within its size band**
(large >= $10B, mid below), raw score breaking ties: on raw points the
first test run gave 19 of 25 slots to non-S&P names, five of them tanker
and dry-bulk shippers riding one trade, at a median 39% volatility.

### Hedge funds' 13F moves (context only)

The nightly job checks 18 long-term, concentrated funds (Berkshire,
Pershing Square, Appaloosa, Baupost, Third Point, Viking, Coatue, Lone
Pine, Duquesne, Himalaya, Tiger, D1, Altimeter, Maverick, Egerton, Akre,
Fundsmith, the Gates Foundation Trust) for a new quarterly 13F (one SEC
request each; a filing is downloaded only when new), maps CUSIPs to
tickers through OpenFIGI (cached; AdGuard needs `@@||openfigi.com^$important`)
and compares each fund's latest two quarters. The daily email lists who
bought or sold your holdings, with the quarter: a 13F is filed up to 45
days after it, so positions can be 4½ months old.

Tested 2026-09-28 on these funds' 13Fs since 2013, bought on the filing
date: one fund's new 2%+ position beat the average S&P 500 stock by 2.1%
over six months (t 1.7), a fund's top holdings not at all, but **two or
more funds buying the same stock in the same quarter** by 3.3% (t 2.0,
positive in both halves). That is borderline, so such a stock becomes a
discover idea source (`fund_consensus`, within 135 days of the filing)
and is flagged in the email — not scored.

## Data frames: Polars

Every frame inside the package is Polars. Daily bars share one shape
(`data/frames.py`: a `date` column plus yfinance's column names), and
yfinance's pandas output is converted where it arrives — the one place
pandas remains. The price cache reads files written before the switch.
The migration was checked against outputs recorded from the pandas code
on frozen inputs — features, the training dataset, walk-forward metrics,
point-in-time fundamentals, technicals, the goal projection, track-record
helpers, reactions: 142 values, all equal to 1e-9.

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

## Plan check: asset location and goal projection

`uv run plan-check` (`--print` to print instead of emailing; the same two
sections are in the quarterly review). No LLM calls.

**Goal projection.** Odds of reaching `GOAL_TARGET_USD` by `GOAL_DATE` (or
in `GOAL_HORIZON_YEARS`, 5), from 10,000 futures built out of 12-month
blocks of the current holdings' own monthly history (15 years from the bar
store; SPY's months stand in for younger stocks, money-market funds count
as cash). The holdings' swings are kept but every month is centred on
`GOAL_EXPECTED_RETURN` (7%/yr) — past returns of stocks held because they
rose are not a forecast. Shows the bad / middle / good case, the same money
in SPY on the same draws, the odds of a 30%+ fall on the way, and the
monthly contribution that would make the odds 75%. Contributions default to
last year's median month of deposits and payroll plan purchases (one-off
lumps like a rollover don't count); `GOAL_MONTHLY_CONTRIBUTION` overrides.

**Asset location.** Accounts are taxable, tax-deferred (Traditional IRA,
401(k)) or tax-free (Roth, HSA). For each holding: what it costs in tax a
year in a taxable account — trailing dividends (REITs at the short-term
rate) plus option premium written there in the last year (short-term).
Suggests swaps that keep the portfolio the same — sell the costly stock in
taxable and buy the cheap one there, the reverse in the IRA — only when
the yearly saving (`ASSET_LOCATION_MIN_DRAG_USD`, $150) pays back the tax
on the taxable sale within `ASSET_LOCATION_MAX_BREAKEVEN_YEARS` (3). A
loss leg carries the wash-sale date: buying the same stock in an IRA within
30 days of a loss sale loses the loss for good. Also flags premium written
in taxable on a stock an IRA holds 100+ shares of.

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
| `fund_positions`, `cusip_tickers` | tracked hedge funds' 13F stock positions (last 4 quarters each) and the CUSIP->ticker cache | ~1,600 rows a quarter, pruned to 4 quarters; cache grows slowly |
| `insider_buys` | open-market purchases (Form 4, code P) at S&P 500 and tracked companies | a few hundred rows a year; dropped after 400 days |
| `analyst_actions` | rating and price-target actions by firm for stocks with a followed earnings report, from 90 days before it | tens of rows per stock |
| `forecast_snapshots` | each weekday night, analysts' consensus for every tracked stock (~175: holdings, a year of picks, recent screen survivors, standouts): EPS and revenue for this and next fiscal year, analyst count, price targets, recommendation — plus short interest (share of float, days to cover), shares outstanding and ownership from the same request — point in time, so revisions and short interest can become model features without look-ahead | ~175 rows per weekday (~4 MB/year); kept forever |
| `filing_facts` | structured facts from each stock's latest 10-Q/10-K (`read-filings`) | two filings per stock acted on, one for the rest (~1,900 stocks, ~5 KB each) |
| `sec_events` | late-filing notices, shelves and offerings, planned insider sales (Form 144) and Schedule 13Ds on holdings; 13Ds across the universe | a few thousand small rows a year |
| `openrouter_spend`, `openrouter_host_checks` | what OpenRouter billed per day/model/stage; each host's weekly known-answer check | a few rows a day |
| `earnings_events` | clear earnings beats anywhere in the market and reports by past picks, with the reaction, the revision and the verdict (`earnings-watch`) | tens of rows a week in earnings season; dropped after 365 days |

Tax lots, cash flows and dividends read the stored activity history, so a
purchase older than the brokerage API's window no longer drops out of lot
splits or FIFO matching. History upkeep trims old prose (365 days), compacts
the file (VACUUM) once trimming frees 20% of it, and the monthly
`model-review` email reports the database size and largest tables, flagging
it past `HISTORY_DB_WARN_MB` (50 MB).

## SEC filings, read by open models

`read-filings` (Saturday 11 PM New York, `scripts/run_filings.sh`, $5 cap)
reads the newest 10-Q / 10-K / 20-F of every stock whose facts can reach
the deciding models — holdings, picks, shortlists, market leaders and
anything that passed the screen's hard filter in the last 90 days (~460 of
the ~1,900 $2B+ stocks; `--all` reads every one). A stock read for the
first time gets its previous filing read too, so its facts say what
changed quarter to quarter. It stores
structured facts in `filing_facts`: guidance, demand, margins, backlog,
liquidity, capital return, key risks, one-offs, reported events
(material weakness, going concern, restatement, investigation...), tone
and what changed since the filing before — every field with a verbatim
quote that is checked against the filing text.

- **Tier A** (holdings, recent picks and shortlists, market leaders) is
  read by `OPENROUTER_READER_MODEL` (GLM-5.3, ~$0.01 a filing); **tier B**
  (the other screen-eligible stocks) by `OPENROUTER_BULK_MODEL`
  (GLM-5.3-Flash, ~$0.002). A bulk read is re-read on GLM-5.3 when it
  flags an event, fails, or when the filing's own XBRL figures show
  operating or net income down 25%+ (`data/income_drop.py`, free).
- Tier A's latest **earnings press release** (8-K item 2.02) is read too —
  a 10-Q seldom states guidance, the release usually does.
- The deciding models (Analyst, Reviewer) get a compact `sec_filing` pack
  and an `earnings_release` in place of the first few thousand characters
  of MD&A they used to see.
- The **screen** takes points off for high-severity red flags in the
  latest filing: going concern -8, material weakness or restatement -4
  (floor -8). Every filing category is stored with each screened
  candidate so `score-attribution` can measure it later.
- A **held** stock's new 10-Q/10-K or material 8-K is read the evening it
  appears (the 6:30 PM snapshot run) and emailed with the drop alerts.
- A 10-K's **risk factors are compared with last year's**
  (`data/text_change.py`, free): the share of sentences carried over
  verbatim (typical 70%) goes to the deciding models as a fact. Tested on
  the S&P 500's 10-Ks since 2012 ("Lazy Prices"): heavily rewritten risk
  factors did precede weaker returns, in both halves, but too weakly to
  score (t 1.6); it is kept with every screened candidate to test again on
  the wider universe.

Useful runs:

```bash
uv run read-filings --dry-run          # fetch and cut every filing, no model calls
uv run read-filings AVGO NVDA          # just these, on GLM-5.3
uv run read-filings --recheck-drops    # re-read stored bulk reads whose income fell
uv run read-filings --spot-check 5     # Claude re-reads 5 random reads (~$0.35), by hand only
```

**Host guardrails** (`openrouter_hosts.py`). The same model at the same
fp8 precision is served by a dozen OpenRouter hosts and they are not
interchangeable — some bill 2-3x more, and two answered a test prompt as
if a place name had been masked. So:

- only **approved hosts** get traffic (`openrouter.APPROVED_HOSTS`); a new
  host gets nothing until it is added there;
- before each weekly read every approved host takes a **known-answer
  check** (a made-up 10-Q, exact quotes required); a host that fails is
  skipped until a later check passes (`openrouter_host_checks`);
- a host whose **quote match** falls below 97% over 30 days, or with over
  10% unusable replies, is skipped automatically;
- a reply containing a redaction placeholder ("[ADDRESS]") is flagged.

The Saturday log ends with a per-host quality table, and `ops doctor`
fails its "OpenRouter hosts" check when any of the above trips.

### Other SEC filings: holding alerts and activist stakes

The 6:30 PM snapshot run also checks each holding for these, from the
same EDGAR filing list it already downloads (`data/sec_events.py`), and
puts them in the evening alert email:

| Filing | Alert | Model |
|---|---|---|
| NT 10-Q / NT 10-K | the company can't file its report on time — a known red flag | none |
| S-3 / S-3ASR | a shelf registration, with the share count's change over a year (XBRL) | none |
| 424B5 | an offering: common stock, at-the-market program, debt or convertible, and how much — plus the share-count change | GLM-5.3, ~$0.005 |
| Form 144 | a planned insider sale of $1M+ (seller, role, shares, value, date), before its Form 4 | none |
| Schedule 13D | a 5%+ holder: who, what percent, and whether they seek changes | GLM-5.3, ~$0.001 |

The **dashboard** shows it all per holding: click a holding for its latest
filing's highlights (summary, guidance, demand, margins, liquidity, reported
events with their quotes, key risks, risk-factor change, what changed since
the filing before), its latest earnings release and its SEC events; a
"SEC events on your holdings" table lists the last 90 days.

Foreign filers are read too: a Canadian 40-F's MD&A is found among its
exhibits (the one titled as an MD&A — not the financial statements that
mention it), and a 20-F's "Operating and Financial Review" by its title.
Shell's and BHP's 20-Fs, which point to page numbers in a separate annual
report, still can't be cut. Every filing is decoded only up to 15 MB — its
prose comes before the exhibits and inline XBRL — so one giant document
can't exhaust memory.

Every night `earnings-watch` also scans EDGAR's daily index for 13Ds on
the whole $2B+ universe. A holder the reader calls activist (board seats,
a sale, a buyback, a strategy change — not an asset manager's routine
"engagement") makes the stock a discover idea source for 60 days
(`activist_13d`, no score bonus). About $5 a year in all.

## Scheduled jobs and upkeep

Cron calls one-line `scripts/run_<job>.sh` shims, which all go through
`scripts/run_job.sh <job> <command...>`: it writes `logs/<job>_YYYYMMDD.log`
and, when the command exits non-zero, emails the log's last 80 lines
(`ops alert`). History upkeep deletes those logs after 30 days.

| Command | What | No LLM |
|---|---|---|
| `ops doctor` | database integrity, `.env` permissions, Finnhub, FRED, SnapTrade, SMTP login, and a free model lookup for every configured (provider, model) — catches a bad key or a retired model id before a run pays for it | ✓ |
| `ops backup` | consistent SQLite copy into `BACKUP_DIR` (default `~/.stock_analyzer/backups`, keep `BACKUP_KEEP`=14), then deletes logs older than `LOG_KEEP_DAYS`=90; uploads it to `BACKUP_REMOTE` when set (below); `scripts/run_backup.sh` runs it nightly | ✓ |
| `scripts/update.sh` | fetches `origin/main`, runs the suite on it in a throwaway worktree, and fast-forwards only if it passes (`scripts/run_update.sh` from cron) | ✓ |

The schedule (New York time):

| When | Script | What | LLM |
|---|---|---|---|
| Weekdays 6:00 AM | `run_ibd.sh` | IBD-style ratings for every $2B+ stock, then the dashboard | — |
| Weekdays 8:30 AM | `run_update.sh` | pull `main` if its tests pass | — |
| **Wednesday 9:30 AM** | `run_portfolio.sh` | the weekly portfolio email (`analyze-portfolio`) | Claude + GLM-5.3 rerank |
| Weekdays 4:15 PM | `run_dashboard.sh` | rebuild the dashboard | — |
| Weekdays 6:30 PM | `run_snapshot.sh` | silent value snapshot; emails only on an unusual drop, a call near its strike, or a held stock's new filing | GLM-5.3 (filings) |
| Weekdays 10:00 PM | `run_earnings_watch.sh` | earnings standouts, insider clusters, cache warm-up | — |
| **Saturday 11:00 PM** | `run_filings.sh` | host checks, then the week's SEC filings | GLM-5.3 / Flash |
| Sunday 8:00 PM | `run_doctor.sh` | `ops doctor` | — |
| Daily 2:00 AM | `run_backup.sh` | database backup, off-site copy | — |
| 1st of month 11:00 AM | `run_model_review.sh` | forward-return model review email | — |
| First week of Jan/Apr/Jul/Oct | `run_quarterly_review.sh` | quarterly suggestions review | — |
| First week of December | `run_tax_planner.sh` | year-end tax plan | — |

`discover-stocks`, `rebalance-portfolio` and `analyze-insiders` are run by
hand. The schedule lives in `scripts/crontab`; install it with `crontab scripts/crontab`.
Ubuntu's cron ignores `CRON_TZ` and the server runs on UTC, so each job is
scheduled at both UTC hours its New York time can fall on (EDT and EST) with
`NY_AT=HH:MM`, and `run_job.sh` runs it only on the firing whose New York hour
matches — no edits at daylight-saving changes:

```cron
# 9:30 AM New York, weekdays
30 13,14 * * 1-5 NY_AT=09:30 /path/to/stock_analyzer/scripts/run_portfolio.sh
```

The insider email (`analyze-insiders`, no longer scheduled: congressional
trades show no edge since the STOCK Act and are disclosed up to 45 days
late; insider buying clusters now come from the nightly job) pairs the
news-based summary with open-market
Form 4 buys and sells filed on your holdings and watchlist (Finnhub, no
LLM). When every news search fails — e.g. the Tavily quota is spent — the
email says so instead of arriving empty, and when every source fails it is
not sent and the job alerts.

Structured LLM stages detect an answer cut off at its output ceiling
(the provider's stop reason) and raise `OutputTruncatedError` with the raw
text, rather than handing a half-written JSON document downstream or
paying for a retry under the same ceiling.


### Off-site backup

The nightly backups sit on a different disk from the database, but in the
same machine. `BACKUP_REMOTE` also sends each one off-site through
[rclone](https://rclone.org), encrypted before it leaves, and keeps
`BACKUP_KEEP` days there; `ops doctor` fails when the newest off-site copy
is over two days old. It's ~7 MB a night, so a free plan is plenty (Box
10 GB, Google Drive 15 GB, Koofr 10 GB, MEGA 20 GB).

One-time setup (Box shown; this machine has no browser):

1. `sudo apt install rclone`
2. `rclone config` → `n` → name `box` → storage `box` → Enter through the
   options → at "Use web browser to automatically authenticate?" answer `n`,
   run the `rclone authorize "box"` it prints on a computer with a browser
   (and rclone), sign in, and paste the token back. (Or connect with
   `ssh -L 53682:localhost:53682 <this machine>`, answer `y`, and open the
   printed link on your own computer.)
3. `rclone config` → `n` → name `box-crypt` → storage `crypt` → remote
   `box:stock-analyzer-backups` → standard filename encryption → let it
   generate the password and the salt. **Save both in a password manager
   off this machine** — without them the copies can't be opened.
4. In `.env`: `BACKUP_REMOTE=box-crypt:` then check with `uv run ops backup`
   and `rclone ls box-crypt:` (Box itself shows only scrambled names).

To restore: `uv run ops restore` downloads the newest off-site copy (or
`ops restore <file>.db`), decrypts it and runs SQLite's integrity check,
into `~/.stock_analyzer/restore`; `--apply` then swaps it in through
SQLite's backup API (safe with the WAL, unlike a file copy), saving the
current database beside it first. On a new machine, set up the same two
rclone remotes with the saved passwords first.
## Tests

```bash
uv run pytest -q -n 4      # -n: spread over 4 workers (pytest-xdist)
uv run ty check src        # type check
```

1,100+ tests covering the high-stakes math (tax-lot computation, verdict
auto-repair, direction-aware and horizon-separated track-record alpha,
beta adjustment, score validation, forecast calibration, parsers,
section-dispatch parity HTML/PDF, multi-provider ranker consensus math,
cross-source data reconciliation, macro-veto rules). The full suite runs
in ~10s (~8s with `-n 4`). CI (`.github/workflows/ci.yml`) runs ruff, ty
and the suite on every push and pull request.

`tests/conftest.py` points `Settings` at no env file and blocks outbound
sockets for the whole suite, so a test can never read your real `.env` or
spend real API quota. Model calls are scripted with
`tests/llm_fakes.script(monkeypatch, replies)` (a Pydantic AI
`FunctionModel` behind the real `ModelAgent`); OpenRouter calls run the
real client against a mock transport (`tests/test_filing_reader._client`),
so tests assert on the request bodies actually sent. A test that genuinely needs the network must be
marked `@pytest.mark.allow_network`.
