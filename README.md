# Stock Analyzer

Daily pre-market intelligence pipeline. At 07:00 ET on weekdays, it sends three Claude-analyzed emails to your inbox:

1. **Portfolio Analysis** — per-holding briefing with overnight price action, news catalysts, insider/institutional activity, and a "trending news" section.
2. **Drawdown Alert** — your holdings down >5% in pre-market, with explanations of likely causes.
3. **Politician Trade Signal** — recent BUY/SELL disclosures from members of Congress whose 24-month track record beats SPY.

Designed for an Ubuntu LXC on a Proxmox home server, with a fully-functional **ephemeral mode** for ad-hoc runs from your laptop without touching the database.

---

## Architecture

```
                      ┌──────────────────┐
                      │   Orchestrator   │  ← single systemd timer fires this
                      └────────┬─────────┘
                               │ Phase 1: shared fetch (sequential)
                               ▼
                       SnapTrade • yfinance • SPY snapshot
                               │
                  ┌────────────┼─────────────┐ Phase 2: agents (parallel)
                  │            │             │
         ┌────────▼──┐  ┌──────▼─────┐  ┌────▼─────────┐
         │ Portfolio │  │ Drawdown   │  │ Politician   │
         │  Agent    │  │  Agent     │  │  Agent       │
         │ (Email 1) │  │ (Email 2)  │  │ (Email 3)    │
         └────────┬──┘  └──────┬─────┘  └────┬─────────┘
                  │            │             │
                  └────────────┼─────────────┘ Phase 3: render + send
                               ▼
                       Stalwart SMTP — three emails
```

**Design principle:** shared data fetched once; agents run in parallel via `asyncio.gather` with `return_exceptions=True` so one agent's failure doesn't block the others.

Full design spec: [`docs/superpowers/specs/2026-05-05-stock-analyzer-design.md`](docs/superpowers/specs/2026-05-05-stock-analyzer-design.md)

---

## Tech Stack

| Layer | Choice |
|---|---|
| Language | Python 3.14 |
| Package manager | `uv` |
| Agent framework | [agno](https://github.com/agno-agi/agno) |
| LLM | Claude Sonnet 4.6 (with prompt caching) |
| Brokerage data | [SnapTrade](https://snaptrade.com) (free tier) |
| Quotes / news | yfinance, Finnhub free tier, MarketWatch + Reuters RSS |
| Insider trades / 13F | SEC EDGAR official JSON API |
| Congressional trades | CapitolTrades public JSON API (deterministic) |
| Hedge-fund commentary | InsiderMonkey via Crawl4ai (agent-driven scraping) |
| Persistence | SQLite + SQLAlchemy 2.x + Alembic |
| Email | SMTP submission to a Stalwart server |
| Templating | Jinja2 (HTML emails) |
| Config | pydantic-settings (`SecretStr` everywhere) |
| Logging | structlog → JSON in production |
| Scheduling | systemd timer (NYSE holidays handled in-app) |
| CLI | Typer |
| Tests | pytest, vcrpy, aiosmtpd |

---

## Quick Start (Development on Your Mac)

```bash
# 1. Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/snehalsoni/stock-analyzer.git
cd stock-analyzer
uv sync

# 3. Configure secrets
cp .env.example .env
$EDITOR .env                          # fill in API keys

# 4. First ad-hoc run (no DB, no SMTP — renders emails to ./out/)
uv run stock-analyzer run --ephemeral --dry-run

# 5. When the output looks good, send real emails (still no DB)
uv run stock-analyzer run --ephemeral

# 6. Sanity check
uv run stock-analyzer health-check
```

You can now iterate on the codebase without touching production state.

---

## CLI Reference

```bash
# Run all three emails (production mode by default)
stock-analyzer run

# Ad-hoc, no DB, no SMTP — emails render to ./out/
stock-analyzer run --ephemeral --dry-run

# Test a subset of agents
stock-analyzer run --only=portfolio
stock-analyzer run --only=drawdown,politician

# Re-run for a past date
stock-analyzer run --date=2026-05-04

# Bypass the "already sent today" idempotency guard
stock-analyzer run --force

# Validate every credential, SnapTrade auth, SMTP, and DB connectivity
stock-analyzer health-check

# Database management (production only)
stock-analyzer db migrate
stock-analyzer db backfill-politicians --months=24
stock-analyzer db recompute-scores

# Inspect recent runs from the audit log
stock-analyzer history --limit=10
```

### Production vs Ephemeral mode

| | Production | Ephemeral |
|---|---|---|
| Database | SQLite | none — in-memory only |
| Migrations | run on every start | skipped |
| Politician scoring filter | applied | bypassed (banner in email) |
| Run audit log | written | stdout only |
| Idempotency guard | enforced | disabled |
| Failed-email replay | written to `failed_emails/` | logged to stdout |

**How mode is chosen:** `--ephemeral` flag wins. Otherwise, if `DATABASE_URL` is unset/unreachable AND `STOCK_ANALYZER_ENV != "production"`, ephemeral is auto-selected. The mode is logged at startup so you always know which one you're in.

---

## Configuration

All configuration is via environment variables. See `.env.example` for the complete list. Key knobs:

| Variable | Default | Purpose |
|---|---|---|
| `ANTHROPIC_MODEL` | `claude-sonnet-4-6` | LLM model |
| `ANTHROPIC_PROMPT_CACHING` | `true` | ~50% input-cost reduction |
| `DRAWDOWN_THRESHOLD_PCT` | `5.0` | Trigger for Email 2 |
| `POLITICIAN_LOOKBACK_MONTHS` | `24` | Window for SPY-beating filter |
| `POLITICIAN_FRESH_DISCLOSURE_DAYS` | `2` | "Disclosed today or yesterday" window (filters on `disclosure_date`, not `trade_date` — Congress can lawfully delay disclosure up to 45 days) |
| `RUN_TIMEZONE` | `America/New_York` | Handles DST automatically |
| `SKIP_NYSE_HOLIDAYS` | `true` | Skip exchange holidays |
| `DRY_RUN` | `false` | Render emails to disk instead of sending |
| `LOG_FORMAT` | `json` | `json` (prod) or `pretty` (dev) |

Secrets (API keys, SMTP password) use `pydantic.SecretStr` — they're never logged or printed.

---

## Deployment to Proxmox LXC

### One-time setup on a fresh Ubuntu LXC

```bash
# As root
apt update && apt install -y python3.14 python3.14-venv git curl
useradd -r -m -d /var/lib/stock-analyzer -s /bin/bash stock-analyzer
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /opt
git clone https://github.com/snehalsoni/stock-analyzer.git
cd stock-analyzer
uv sync --frozen
uv pip install -e .

# Place secrets
install -d -m 0755 /etc/stock-analyzer
install -m 0600 -o stock-analyzer .env.example /etc/stock-analyzer/env
$EDITOR /etc/stock-analyzer/env

# Initialize DB and seed historical data
sudo -u stock-analyzer stock-analyzer db migrate
sudo -u stock-analyzer stock-analyzer db backfill-politicians --months=24

# Install and start the timer
cp deploy/stock-analyzer.service /etc/systemd/system/
cp deploy/stock-analyzer.timer   /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now stock-analyzer.timer

# Verify
systemctl list-timers stock-analyzer
sudo -u stock-analyzer stock-analyzer health-check
```

### Updates

```bash
cd /opt/stock-analyzer
sudo -u stock-analyzer git pull
sudo -u stock-analyzer uv sync --frozen
sudo -u stock-analyzer stock-analyzer db migrate
sudo -u stock-analyzer stock-analyzer health-check
```

The next timer fire picks up the new code automatically.

### Backup

`/var/lib/stock-analyzer/stock_analyzer.db` is the only stateful file. A weekly cron handles it:

```cron
0 4 * * 0  stock-analyzer  sqlite3 /var/lib/stock-analyzer/stock_analyzer.db ".backup /var/lib/stock-analyzer/backups/$(date +\%Y\%m\%d).db"
```

Or fold the path into your existing Proxmox backup job.

---

## Cost Profile

| | Cost |
|---|---|
| Anthropic (Sonnet 4.6 + prompt caching) | **~$50–150 / year** for one daily run |
| SnapTrade, Finnhub, SEC EDGAR, Yahoo/yfinance, CapitolTrades, InsiderMonkey | $0 |
| Stalwart SMTP (self-hosted) | $0 |

Token usage and per-run cost is captured in the `runs` table and visible via `stock-analyzer history`.

---

## Project Layout

```
stock_analyzer/
├── pyproject.toml
├── .env.example
├── README.md
├── deploy/                       # systemd units + installer
├── docs/
│   └── superpowers/specs/        # design specs
├── src/stock_analyzer/
│   ├── __main__.py               # Typer CLI
│   ├── orchestrator.py           # 3-phase pipeline
│   ├── config.py                 # pydantic-settings
│   ├── logging.py                # structlog
│   ├── agents/                   # one Agent per email
│   ├── tools/                    # focused, reusable agno tools
│   ├── persistence/              # SQLAlchemy + ephemeral equivalents
│   ├── analytics/                # pure-logic helpers (no LLM)
│   ├── rendering/                # Jinja2 templates + renderer
│   └── calendar/                 # NYSE holiday helper
└── tests/
    ├── unit/
    ├── integration/              # vcrpy cassettes
    └── e2e/                      # ephemeral + dry-run + Claude stub
```

---

## Development

```bash
# Run unit tests
uv run pytest tests/unit/

# Lint, format, typecheck
uv run ruff check .
uv run ruff format .
uv run mypy --strict src/

# Refresh integration cassettes
uv run pytest tests/integration/ --record-mode=new_episodes

# Full test suite (used in CI)
uv run pytest
```

Pre-commit hooks run `ruff`, `mypy`, and the unit test suite. Integration tests use `vcrpy` cassettes; the end-to-end suite runs in ephemeral mode with Claude stubbed for deterministic output.

---

## Status

**v1 — pre-implementation.** Design approved on 2026-05-05; implementation plan and code generation are next.

### Out of scope for v1

- Multi-user / multi-portfolio support
- Web UI
- Real-time intraday alerts
- Mobile push notifications
- Backtesting harness for the politician-scoring methodology

---

## License

Private — for the operator's personal use.
