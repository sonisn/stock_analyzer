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
# Install (Python 3.14+, uses uv). `--extra dev` keeps pytest + ruff;
# a plain `uv sync` removes them.
uv sync --extra dev

# Configure — copy and fill in your keys
cp .env.example .env

# Run
uv run discover-stocks            # find new picks
uv run rebalance-portfolio        # review holdings + plan
uv run analyze-portfolio          # one-off analyst-style report
uv run analyze-insiders           # insider + political trade signals
uv run ops doctor                 # free check of every key, model id and source
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

## Scheduled jobs and upkeep

Cron calls one-line `scripts/run_<job>.sh` shims, which all go through
`scripts/run_job.sh <job> <command...>`: it writes `logs/<job>_YYYYMMDD.log`
and, when the command exits non-zero, emails the log's last 80 lines
(`ops alert`). History upkeep deletes those logs after 30 days.

| Command | What | No LLM |
|---|---|---|
| `ops doctor` | database integrity, `.env` permissions, Finnhub, FRED, SnapTrade, SMTP login, and a free model lookup for every configured (provider, model) — catches a bad key or a retired model id before a run pays for it | ✓ |
| `ops backup` | consistent SQLite copy into `BACKUP_DIR` (default `~/.stock_analyzer/backups`, keep `BACKUP_KEEP`=14); `scripts/run_backup.sh` runs it nightly | ✓ |
| `scripts/update.sh` | fetches `origin/main`, runs the suite on it in a throwaway worktree, and fast-forwards only if it passes (`scripts/run_update.sh` from cron) | ✓ |

Suggested crontab (New York time):

```cron
CRON_TZ=America/New_York
30 12 * * 1-5 /path/to/stock_analyzer/scripts/run_update.sh
30 13 * * 1-5 /path/to/stock_analyzer/scripts/run_portfolio.sh
35 13 * * 1   /path/to/stock_analyzer/scripts/run_insiders.sh
15 16 * * 1-5 /path/to/stock_analyzer/scripts/run_dashboard.sh
0  2  * * *   /path/to/stock_analyzer/scripts/run_backup.sh
```

The weekly insider email pairs the news-based summary with open-market
Form 4 buys and sells filed on your holdings and watchlist (Finnhub, no
LLM). When every news search fails — e.g. the Tavily quota is spent — the
email says so instead of arriving empty, and when every source fails it is
not sent and the job alerts.

Structured LLM stages detect an answer cut off at its output ceiling
(agno does not pass the provider's stop reason through, so it is read off
the token count) and raise `OutputTruncatedError` with the raw text,
rather than handing a half-written JSON document downstream.

## Tests

```bash
uv run pytest -q
```

860+ tests covering the high-stakes math (tax-lot computation, verdict
auto-repair, direction-aware and horizon-separated track-record alpha,
beta adjustment, score validation, forecast calibration, parsers,
section-dispatch parity HTML/PDF, multi-provider ranker consensus math,
cross-source data reconciliation, macro-veto rules). The full suite runs
in ~30s. CI (`.github/workflows/ci.yml`) runs ruff and the suite on every
push and pull request.

`tests/conftest.py` points `Settings` at no env file and blocks outbound
sockets for the whole suite, so a test can never read your real `.env` or
spend real API quota. A test that genuinely needs the network must be
marked `@pytest.mark.allow_network`.
