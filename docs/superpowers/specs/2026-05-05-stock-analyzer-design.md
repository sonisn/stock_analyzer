# Stock Analyzer — Design Spec

**Date:** 2026-05-05
**Author:** snehal.soni@farohealth.com
**Status:** Approved (pending implementation plan)

---

## 1. Purpose

A daily, pre-market intelligence pipeline that emails three reports to the operator at **07:00 ET on weekdays** (skipping NYSE holidays):

1. **Portfolio Analysis** — per-holding briefing with overnight price action, news catalysts, insider/institutional activity, and a "trending news" section ranked by a deterministic score.
2. **Drawdown Alert** — list of *the operator's holdings* whose pre-market price is more than 5% below the previous close, with Claude-generated explanations of likely causes.
3. **Politician Trade Signal** — recent BUY/SELL disclosures from members of Congress whose 24-month track record beats SPY, with Claude-generated thesis commentary per trade.

All three emails use Claude Sonnet 4.6 (via the `agno` framework) for analysis. The application targets an Ubuntu LXC container on a Proxmox home server, with a fully-functional **ephemeral mode** for ad-hoc execution on a developer's machine without database state.

---

## 2. Architecture: Three Agents + Orchestrator

```
                      ┌──────────────────┐
                      │   Orchestrator   │  ← single systemd timer fires this
                      └────────┬─────────┘
                               │ Phase 1: shared fetch (sequential)
                               ▼
                       SnapTrade • yfinance • SPY snapshot
                               │
                  ┌────────────┼─────────────┐ Phase 2: agents (asyncio.gather)
                  │            │             │
         ┌────────▼──┐  ┌──────▼─────┐  ┌────▼─────────┐
         │ Portfolio │  │ Drawdown   │  │ Politician   │
         │  Agent    │  │  Agent     │  │  Agent       │
         │ (Email 1) │  │ (Email 2)  │  │ (Email 3)    │
         └────────┬──┘  └──────┬─────┘  └────┬─────────┘
                  │            │             │
                  └────────────┼─────────────┘ Phase 3: render + send (sequential)
                               ▼
                       Stalwart SMTP — three emails
```

### Why three agents (not one workflow, not three scripts)

- **One workflow:** simplest but a single failure halts everything; can't parallelize the slow scraping.
- **Three scripts/timers:** maximum isolation but duplicates code, fetches portfolio 3x, harder to deploy atomically.
- **Three agents + orchestrator (chosen):** shared fetch happens once, agents run in parallel (~60s vs ~3min sequential), `asyncio.gather(return_exceptions=True)` gives per-email failure isolation. Right balance of clean SRP and operational simplicity.

---

## 3. Repository Layout

```
stock_analyzer/
├── pyproject.toml                  # uv-managed deps, Python 3.14
├── .env.example                    # template for secrets (no real values)
├── README.md
├── deploy/
│   ├── stock-analyzer.service      # systemd unit
│   ├── stock-analyzer.timer        # systemd timer (07:00 ET, weekdays)
│   └── install.sh                  # idempotent installer for the LXC
├── docs/
│   └── superpowers/specs/          # design specs (this file lives here)
├── src/
│   └── stock_analyzer/
│       ├── __init__.py
│       ├── __main__.py             # Typer CLI entry point
│       ├── orchestrator.py         # top-level pipeline
│       ├── config.py               # pydantic-settings, loads from .env
│       ├── logging.py              # structlog → JSON in prod, pretty in dev
│       ├── agents/
│       │   ├── portfolio_agent.py
│       │   ├── drawdown_agent.py
│       │   └── politician_agent.py
│       ├── tools/                  # agno tools (each focused, reusable)
│       │   ├── snaptrade.py
│       │   ├── market_data.py      # yfinance wrapper + Yahoo trending feed
│       │   ├── news.py             # Finnhub + yfinance news + RSS
│       │   ├── sec_edgar.py        # Form 4 + 13F
│       │   ├── capitol_trades.py   # deterministic — hits bff.capitoltrades.com JSON API
│       │   ├── insider_monkey.py   # agent-driven — Crawl4aiTools fetches markdown
│       │   └── smtp_sender.py      # Stalwart submission
│       ├── persistence/
│       │   ├── db.py               # SQLite connection, raises in ephemeral mode
│       │   ├── models.py           # SQLAlchemy models
│       │   ├── repositories.py     # query helpers (politicians, SPY, runs)
│       │   └── in_memory.py        # ephemeral-mode equivalents
│       ├── analytics/
│       │   ├── politician_scorer.py    # 24-month vs SPY computation
│       │   ├── drawdown_filter.py
│       │   ├── news_ranker.py          # deterministic trending score
│       │   └── cost_tracker.py
│       ├── rendering/
│       │   ├── templates/              # Jinja2 HTML templates
│       │   │   ├── portfolio.html.j2
│       │   │   ├── drawdown.html.j2
│       │   │   └── politician.html.j2
│       │   └── renderer.py
│       └── calendar/
│           └── nyse.py             # is_market_holiday() via pandas-market-calendars
└── tests/
    ├── unit/
    ├── integration/                # vcrpy-recorded fixtures
    ├── e2e/                        # ephemeral + dry-run + mocked Claude
    └── fixtures/
```

**Conventions**

- `src/` layout (modern Python, prevents accidental dev imports).
- Pydantic v2 throughout for config and inter-component contracts.
- SQLAlchemy 2.x + Alembic for schema migrations.
- Jinja2 for HTML emails — Claude returns Pydantic models, the renderer is dumb (no LLM calls).

---

## 4. Data Flow

### Phase 1 — Shared Fetch (sequential, ~15s)

1. `SnapTrade.get_holdings()` → `portfolio: list[Holding]`
2. `market_data.batch_quotes(tickers + ["SPY"])` → previous closes + pre-market quotes
3. `persistence.upsert_spy_close(yesterday)` (skipped in ephemeral mode)
4. `persistence.recompute_politician_scores_if_stale()` (skipped in ephemeral mode)

### Phase 2 — Agents (parallel via `asyncio.gather(return_exceptions=True)`, ~45s)

Each agent receives shared data by reference and uses only the tools it needs (small, predictable tool surface for Claude).

If an agent raises, the others continue. The orchestrator collects results, replaces exceptions with a `FailedReport` placeholder, and the renderer produces a clearly-marked degraded email instead of dropping it silently.

### Phase 3 — Render + Send (sequential, ~5s)

1. Each agent's typed Pydantic result is fed to the matching Jinja2 template.
2. `smtp_sender.send()` submits each email to Stalwart.
3. `runs` row is finalized with per-email status, total token usage, est. cost.

**Idempotency:** the orchestrator checks the `runs` table at startup. If a successful run for today exists and the operator didn't pass `--force`, the run aborts cleanly (prevents accidental double-sends from manual reruns). Disabled in ephemeral mode.

---

## 5. Agent Specifications

All three agents share these settings:

- **Model:** `claude-sonnet-4-6`
- **Prompt caching:** enabled (~50% input cost reduction)
- **Max steps:** 10 per agent (typical observed: 4–6)
- **Retries:** 3 on `anthropic.APIStatusError` 5xx with exponential backoff
- **Output:** Pydantic-typed via agno's `response_model=` (no JSON-from-prose parsing)

### 5.1 Portfolio Agent — Email 1

**Tools:** `news.get_yfinance_news`, `news.get_company_news` (Finnhub), `news.get_trending_tickers`, `sec_edgar.get_recent_form_4`, `sec_edgar.get_recent_13f`, `insider_monkey.search_articles`, `market_data.get_overnight_change`

**System prompt (gist):**
> You are a portfolio analyst. For each holding, write 3–5 bullets covering: overnight price action vs SPY, any insider/institutional activity in the past 7 days, top news catalyst, smart-money commentary if available, and a one-line "what to watch today." Be factual, no hype, no buy/sell recommendations.

**Trending news section** (top of email): the orchestrator pre-computes a deterministic score (no LLM tokens) using:

```
score = (50  if ticker in yahoo_trending_tickers  else 0)
      + (30 * (number_of_my_holdings_mentioned - 1))
      + tier_weight[publisher]                       # 0-30 (Reuters/WSJ/Bloomberg/CNBC = 30)
      + recency_decay(published_at)                  # 0-20, last 24h = 20
```

Top 5 articles surface at the top of the email; the agent adds a single-line context summary to each.

**Response model:**

```python
class TrendingArticle(BaseModel):
    title: str
    publisher: str
    url: HttpUrl
    published_at: datetime
    related_tickers: list[str]
    is_market_wide_trending: bool
    score: float

class HoldingBrief(BaseModel):
    ticker: str
    company_name: str
    pct_change_overnight: float
    bullets: list[str]                # 3–5 short lines
    sources: list[HttpUrl]
    watch_today: str                  # one line

class PortfolioReport(BaseModel):
    as_of: datetime
    trending_news: list[TrendingArticle]   # top 5
    holdings: list[HoldingBrief]
    portfolio_summary: str            # 2–3 sentences
```

### 5.2 Drawdown Agent — Email 2

The orchestrator pre-filters portfolio holdings to those with `pct_change_overnight < -5.0`. If the filtered list is empty, the agent returns a "no drawdowns today" report — the email still ships as a daily heartbeat.

**Tools (only invoked when there's something to analyze):** `news.get_company_news`, `news.get_macro_news`

**System prompt (gist):**
> You receive a list of stocks down >5% in pre-market. For each one, identify the most likely cause (earnings, downgrade, sector move, macro news, no clear catalyst). Be specific about news sources. Flag when no clear catalyst — that's actionable signal too.

**Response model:**

```python
class DrawdownItem(BaseModel):
    ticker: str
    pct_drop: float                           # negative number
    pre_market_price: float
    prev_close: float
    likely_cause: Literal[
        "earnings", "downgrade", "sector",
        "macro", "company_news", "no_clear_catalyst"
    ]
    explanation: str                          # 1–2 sentences
    sources: list[HttpUrl]

class DrawdownReport(BaseModel):
    as_of: datetime
    items: list[DrawdownItem]                 # empty list = no drawdowns
    market_context: str | None
```

### 5.3 Politician Agent — Email 3

Independent of portfolio data. Logic flow:

1. Orchestrator fetches the last 7 days of disclosures via the deterministic CapitolTrades JSON API.
2. Orchestrator refreshes `politician_scores` (skipped if computed today).
3. Orchestrator filters to trades where the politician's 24-month return beats SPY *and* the **`disclosure_date`** is within `POLITICIAN_FRESH_DISCLOSURE_DAYS` (default 2 — disclosure happened today or yesterday). We filter on `disclosure_date` rather than `trade_date` because that is the actionable signal: it's when the public *learned* of the trade, and Congress members can lawfully delay disclosure up to 45 days, so `trade_date` would routinely return zero results.
4. Filtered list is passed to the agent for narrative analysis.

In ephemeral mode, step 2's filter is bypassed; all recent disclosures pass through, with a banner at the top of the email noting *"Ephemeral mode: politician scoring filter disabled."*

**Tools:** `news.get_company_news`, `market_data.get_quote`, `sec_edgar.get_recent_form_4`

**Response model:**

```python
class PoliticianTrade(BaseModel):
    politician_name: str
    politician_party: Literal["D", "R", "I"]
    politician_chamber: Literal["House", "Senate"]
    politician_24mo_alpha_vs_spy: float       # percentage points
    ticker: str
    side: Literal["BUY", "SELL"]
    trade_date: date
    disclosure_date: date
    amount_range: str                         # e.g. "$1,001 - $15,000"
    likely_thesis: str
    aligns_with_insiders: bool | None         # None if no recent Form 4
    sources: list[HttpUrl]

class PoliticianReport(BaseModel):
    as_of: datetime
    buys: list[PoliticianTrade]
    sells: list[PoliticianTrade]
    top_takeaway: str                         # one paragraph
```

---

## 6. Hybrid Scraping Strategy

| Source | Strategy | Why |
|---|---|---|
| **CapitolTrades** | Deterministic — hit `bff.capitoltrades.com/trades` JSON API directly | Stable structured data; precise dollar amounts and dates; no LLM tokens |
| **InsiderMonkey** | Agent-driven via `agno.tools.crawl4ai.Crawl4aiTools` (free, no API key) | Narrative content best summarized by an LLM; structure changes often |
| **SEC EDGAR (Form 4, 13F)** | Deterministic — official JSON API | Authoritative source; rate limit 10 req/s respected via tenacity |
| **yfinance / Yahoo trending** | Deterministic library calls; both unofficial endpoints wrapped in try/except with fallbacks | Library is well-maintained; fallback to Finnhub news + cross-ticker scoring if Yahoo redesigns |
| **Finnhub news** | Deterministic — official free-tier API (60 calls/min) | Stable API |

---

## 7. Persistence — SQLite + SQLAlchemy 2.x + Alembic

Single file at `/var/lib/stock-analyzer/stock_analyzer.db`. Five tables:

```
politicians
  id PK, full_name UNIQUE, party CHECK('D','R','I'),
  chamber CHECK('House','Senate'), state, capitol_trades_id UNIQUE,
  created_at, updated_at

politician_trades                                    -- raw disclosures, source of truth
  id PK, politician_id FK, ticker, side CHECK('BUY','SELL'),
  trade_date, disclosure_date, amount_min_usd, amount_max_usd,
  raw_payload JSON, ingested_at,
  UNIQUE(politician_id, ticker, side, trade_date, disclosure_date),
  INDEX(disclosure_date), INDEX(politician_id)

politician_scores                                    -- recomputed daily
  politician_id PK FK, computed_at, window_start_date, window_end_date,
  total_return_pct, spy_return_pct, alpha_vs_spy_pct, trade_count,
  beats_spy GENERATED ALWAYS AS (alpha_vs_spy_pct > 0) STORED

spy_daily_close
  trade_date PK, close_price, fetched_at

runs                                                 -- one row per orchestrator execution
  id PK, run_date, started_at, completed_at,
  status CHECK('running','success','partial','failed'),
  email_1_status, email_2_status, email_3_status,
  error_log JSON, total_tokens_in, total_tokens_out, est_cost_usd,
  UNIQUE(run_date)
```

### Politician scoring computation (Phase 1, daily)

For each politician with disclosures in the last 24 months:

1. Pull all `politician_trades` from `(today - 24 months)` to `today`.
2. Simulate a portfolio: midpoint of each disclosed amount range, "buy" on `disclosure_date + 1` (when public could have learned), "sell" same way.
3. Compute total return % over the window.
4. Pull SPY return over the same window from `spy_daily_close`.
5. Upsert `politician_scores` row.

### Bootstrap

The very first production run scrapes 24 months of CapitolTrades history (one-time, ~5 minutes) to seed `politician_trades`. Subsequent runs only fetch the last 7 days; the `UNIQUE` constraint makes inserts idempotent.

---

## 8. Configuration & Secrets

Loaded by `pydantic-settings` from `.env` (dev) or systemd `EnvironmentFile=` (prod). Validation at startup; missing/malformed config crashes immediately, before any API call.

### `.env.example`

```ini
# ─── Anthropic / agno ───────────────────────────────────────────
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_MODEL=claude-sonnet-4-6
ANTHROPIC_MAX_TOKENS=4096
ANTHROPIC_PROMPT_CACHING=true

# ─── SnapTrade ──────────────────────────────────────────────────
SNAPTRADE_CLIENT_ID=YOUR_CLIENT_ID
SNAPTRADE_CONSUMER_KEY=YOUR_CONSUMER_KEY
SNAPTRADE_USER_ID=your-registered-user-id
SNAPTRADE_USER_SECRET=your-user-secret

# ─── Finnhub (free tier) ────────────────────────────────────────
FINNHUB_API_KEY=xxxxxxxxxxxxxxx

# ─── Stalwart SMTP ──────────────────────────────────────────────
SMTP_HOST=mail.thesonihub.com
SMTP_PORT=587                  # STARTTLS (or 465 for implicit TLS)
SMTP_USE_TLS=true
SMTP_USERNAME=stock-analyzer@thesonihub.com
SMTP_PASSWORD=app-password-here
SMTP_FROM_ADDRESS=stock-analyzer@thesonihub.com
SMTP_FROM_NAME="Stock Analyzer"
SMTP_TO_ADDRESS=snehal@thesonihub.com

# ─── Storage ────────────────────────────────────────────────────
DATABASE_URL=sqlite:////var/lib/stock-analyzer/stock_analyzer.db
FAILED_EMAILS_DIR=/var/lib/stock-analyzer/failed_emails

# ─── Behavior ───────────────────────────────────────────────────
RUN_TIMEZONE=America/New_York
DRAWDOWN_THRESHOLD_PCT=5.0
POLITICIAN_LOOKBACK_MONTHS=24
POLITICIAN_FRESH_DISCLOSURE_DAYS=2
LOG_LEVEL=INFO
LOG_FORMAT=json                # 'json' for prod, 'pretty' for dev

# ─── Optional flags ─────────────────────────────────────────────
DRY_RUN=false                  # if true, render emails to disk instead of sending
SKIP_NYSE_HOLIDAYS=true
STOCK_ANALYZER_ENV=production  # 'production' or 'development'
```

### Secret handling on the LXC

```
/etc/stock-analyzer/env       # 0600, owned by stock-analyzer:stock-analyzer
/var/lib/stock-analyzer/      # 0700, db + failed_emails live here
/var/log/stock-analyzer/      # 0750, optional stdout/stderr capture
```

`pydantic.SecretStr` ensures secrets never surface in repr/logs. Rotating a key: `sudo -e /etc/stock-analyzer/env`, then the next timer tick picks it up — no rebuild required.

---

## 9. Execution Modes & CLI

### Two modes

| Concern | Production | Ephemeral |
|---|---|---|
| Database | SQLite at `/var/lib/...` | None — in-memory dicts only |
| Alembic migrations | run on `ExecStartPre` | skipped |
| 24-month politician score | computed from `politician_trades` table | filter bypassed; banner in email |
| SPY history | `spy_daily_close` table | fetched live from yfinance (small range) |
| Run audit (`runs` table) | written | logged to stdout/journalctl only |
| Idempotency guard ("already sent today") | enforced via `runs` row | disabled |
| Failed-email replay | written to `failed_emails/` | logged with full payload |
| Email send | normal | unchanged unless `--dry-run` is also passed |

### Mode selection

- Explicit `--ephemeral` flag wins.
- Otherwise auto-detect: `DATABASE_URL` unset/unreachable AND `STOCK_ANALYZER_ENV != "production"` → ephemeral.
- Mode is logged at startup:
  `[INFO] Mode: EPHEMERAL — no DB, politician scoring filter disabled, results not persisted`

### CLI surface (Typer)

```bash
stock-analyzer run                             # default — production mode
stock-analyzer run --ephemeral --dry-run       # most common dev invocation
stock-analyzer run --only=portfolio            # one agent at a time
stock-analyzer run --only=drawdown,politician
stock-analyzer run --date=2026-05-04           # rerun for a past date
stock-analyzer run --force                     # bypass idempotency guard

stock-analyzer health-check                    # validates creds, SnapTrade, SMTP, DB

stock-analyzer db migrate                      # production only
stock-analyzer db backfill-politicians --months=24
stock-analyzer db recompute-scores

stock-analyzer history --limit=10              # last N runs from audit log
```

`stock-analyzer` is exposed as a console script via `pyproject.toml`:

```toml
[project.scripts]
stock-analyzer = "stock_analyzer.__main__:app"
```

### Implementation notes

- `persistence/db.py::get_session()` raises `EphemeralModeError` if called while in ephemeral mode. Guarantees no agent or repository accidentally hits a DB it shouldn't.
- Repositories have ephemeral-mode equivalents in `persistence/in_memory.py` exposing the same interface. Orchestrator wires the right implementation at startup (Strategy pattern).
- Email banner is conditional on `mode == "ephemeral"` in the Jinja2 templates.

---

## 10. Error Handling, Observability, Testing

### Retry policy (tenacity)

| Operation | Retries | Backoff | Per-call timeout |
|---|---|---|---|
| Anthropic API (5xx, 429) | 3 | exp 2→16s + jitter | 60s |
| SnapTrade API | 3 | exp 1→8s | 30s |
| yfinance / Yahoo | 2 | linear 5s | 15s |
| Finnhub | 3 | exp 1→8s | 15s |
| SEC EDGAR | 3 | exp 2→16s (10 req/s limit) | 30s |
| CapitolTrades JSON API | 3 | exp 2→16s | 30s |
| Crawl4ai (InsiderMonkey) | 2 | linear 10s | 60s |
| SMTP send | 5 | exp 5→300s | 30s |

After max retries: operation logged as failed, agent returns partial result with `errors: list[str]` field, email rendered with clearly-marked "could not fetch X" placeholder.

### Observability

- **Logging:** `structlog` → JSON in production (parseable via `journalctl -o cat | jq`), pretty in dev. One log per phase, per tool call, per Claude call. Secrets redacted by `SecretStr`.
- **Metrics emitted as structured events:**
  - `run.duration_seconds` per phase + total
  - `claude.tokens_in`, `claude.tokens_out`, `claude.cost_usd` per agent
  - `tool.calls`, `tool.errors` per tool
  - `email.send_status` per email
- **Cost tracking:** `analytics/cost_tracker.py` writes daily totals to `runs.est_cost_usd`. Visible via `stock-analyzer history`.
- **Alerting:**
  - Total run failure (zero emails sent) → fallback plain-text alert email via minimal-dependency code path.
  - If even that fails → systemd marks the unit failed; `systemctl --failed` surfaces it on next SSH.

### Testing strategy

```
tests/
├── unit/                     # ≥ 90% coverage on analytics/, persistence/, rendering/
├── integration/              # vcrpy cassettes for HTTP, aiosmtpd for SMTP
├── e2e/                      # full pipeline in ephemeral + dry-run; Claude stubbed
└── fixtures/cassettes/       # refreshed monthly via --record-mode=new_episodes
```

- **Unit:** pytest, pure-logic targets — news ranker, drawdown filter, politician scorer, renderers.
- **Integration:** `vcrpy`-recorded HTTP, replayed in CI.
- **End-to-end:** ephemeral + dry-run with Claude swapped for a stub returning canned Pydantic models. Verifies HTML emails render correctly.
- **No live-API tests in CI** — keeps CI fast, free, reliable.
- **Pre-commit:** `ruff` (lint + format), `mypy --strict`, `pytest -q tests/unit/`. Full suite in CI.

---

## 11. Deployment to LXC

### One-time setup

```bash
# As root in the fresh LXC:
apt update && apt install -y python3.14 python3.14-venv git curl
useradd -r -m -d /var/lib/stock-analyzer -s /bin/bash stock-analyzer
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /opt
git clone https://github.com/snehalsoni/stock-analyzer.git
cd stock-analyzer
uv sync --frozen
uv pip install -e .

install -d -m 0755 /etc/stock-analyzer
install -m 0600 -o stock-analyzer .env.example /etc/stock-analyzer/env
$EDITOR /etc/stock-analyzer/env

sudo -u stock-analyzer stock-analyzer db migrate
sudo -u stock-analyzer stock-analyzer db backfill-politicians --months=24

cp deploy/stock-analyzer.service /etc/systemd/system/
cp deploy/stock-analyzer.timer   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now stock-analyzer.timer

systemctl list-timers stock-analyzer
sudo -u stock-analyzer stock-analyzer health-check
```

### `deploy/stock-analyzer.service`

```ini
[Unit]
Description=Stock Analyzer — daily pre-market analysis
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=stock-analyzer
Group=stock-analyzer
EnvironmentFile=/etc/stock-analyzer/env
WorkingDirectory=/opt/stock-analyzer
ExecStartPre=/opt/stock-analyzer/.venv/bin/stock-analyzer db migrate
ExecStart=/opt/stock-analyzer/.venv/bin/stock-analyzer run
StateDirectory=stock-analyzer
StateDirectoryMode=0700
TimeoutStartSec=10min

# Hardening
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=true
PrivateTmp=true
ReadWritePaths=/var/lib/stock-analyzer
RestrictAddressFamilies=AF_INET AF_INET6
SystemCallFilter=@system-service
LockPersonality=true
MemoryDenyWriteExecute=true

# Resource caps
MemoryMax=512M
CPUQuota=100%

[Install]
WantedBy=multi-user.target
```

### `deploy/stock-analyzer.timer`

```ini
[Unit]
Description=Run Stock Analyzer at 07:00 ET on weekdays

[Timer]
OnCalendar=Mon..Fri 07:00 America/New_York
Persistent=false               # don't run a missed schedule (avoid stale 9am send)
RandomizedDelaySec=120         # spread load if other timers fire at 7am

[Install]
WantedBy=timers.target
```

NYSE holiday detection happens **inside the app** via `pandas-market-calendars`, not in the timer. Holiday-skipped days are recorded in `runs` for audit.

### Updates / redeploys

```bash
cd /opt/stock-analyzer
sudo -u stock-analyzer git pull
sudo -u stock-analyzer uv sync --frozen
sudo -u stock-analyzer stock-analyzer db migrate
sudo -u stock-analyzer stock-analyzer health-check
```

### Backup

`/var/lib/stock-analyzer/stock_analyzer.db` is the only stateful file:

```cron
0 4 * * 0  stock-analyzer  sqlite3 /var/lib/stock-analyzer/stock_analyzer.db ".backup /var/lib/stock-analyzer/backups/$(date +\%Y\%m\%d).db"
```

…or include the path in the existing Proxmox backup job.

---

## 12. Cost Profile

- **Anthropic (Sonnet 4.6 + prompt caching):** ~$0.15–0.40 per daily run; ~$50–150/year.
- **SnapTrade:** free tier (single user).
- **Finnhub:** free tier (60 calls/min — well under).
- **SEC EDGAR:** free.
- **Yahoo / yfinance:** free, unofficial.
- **CapitolTrades / InsiderMonkey:** free (scraping public pages).
- **Stalwart SMTP:** self-hosted, free.

**Total expected:** ~$50–150/year for the LLM, $0 for everything else.

---

## 13. Out of Scope (v1)

These are intentionally deferred to keep v1 shippable. Each is a candidate for a v2 spec.

- Multi-user / multi-portfolio support.
- Web UI for browsing run history.
- Mobile push notifications (in addition to email).
- Real-time intraday alerts (this v1 is once-per-day pre-market only).
- Backtesting harness for the politician-scoring methodology.
- LLM-powered trade-execution agent (we deliberately stop at *information*).

---

## 14. Open Questions / Risks

| Risk | Mitigation |
|---|---|
| CapitolTrades `bff` JSON endpoint changes shape | Schema validation via Pydantic; agent failure-isolated; one-line config swap to scrape HTML or upgrade to QuiverQuant ($10/mo) |
| Yahoo `trendingTickers` endpoint disappears | Wrapped in try/except; trending news section degrades gracefully to cross-ticker + publisher-tier scoring only |
| InsiderMonkey blocks Crawl4ai user-agent | Crawl4ai supports rotating UAs; on persistent failure, this section of Email 1 is omitted with a one-line note |
| 24-month politician scoring methodology is naive (assumes midpoint of disclosed range, no transaction costs) | Documented in code; can be refined in v2 without changing the schema |
| Pre-market liquidity is thin → noisy quotes for Email 2 | Drawdown threshold defaulted to 5% (large enough to filter noise); operator can raise via env var |
| Stalwart SMTP server downtime | 5 retries with exp backoff; failed emails written to `failed_emails/` and replayed on the next run |
