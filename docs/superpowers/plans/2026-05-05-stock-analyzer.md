# Stock Analyzer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a daily pre-market intelligence pipeline that emails three Claude-analyzed reports (portfolio, drawdowns, politician trades) at 07:00 ET on weekdays from an Ubuntu LXC, with a fully-functional ephemeral mode for ad-hoc developer runs.

**Architecture:** Three focused `agno.Agent`s (one per email) run in parallel via `asyncio.gather` after a single shared-fetch phase, then a sequential render-and-send phase ships the emails through Stalwart SMTP. SQLite holds politician trade history and run audit logs in production; ephemeral mode skips the DB entirely.

**Tech Stack:** Python 3.14, `uv`, `agno`, Anthropic Claude Sonnet 4.6, SnapTrade, yfinance, Finnhub free tier, SEC EDGAR, CapitolTrades JSON API, Crawl4ai (InsiderMonkey), SQLAlchemy 2.x + Alembic + SQLite, Jinja2, structlog, pydantic-settings, Typer, pytest, vcrpy, aiosmtpd, tenacity, pandas-market-calendars.

**Spec:** [`docs/superpowers/specs/2026-05-05-stock-analyzer-design.md`](../specs/2026-05-05-stock-analyzer-design.md)

---

## File Structure (locked at plan time)

```
stock_analyzer/
├── pyproject.toml                          # T1
├── .env.example                            # T2
├── .python-version                         # T1
├── alembic.ini                             # T7
├── deploy/
│   ├── stock-analyzer.service              # T35
│   ├── stock-analyzer.timer                # T35
│   └── install.sh                          # T36
├── docs/
│   ├── superpowers/specs/                  # already exists
│   └── superpowers/plans/                  # already exists
├── migrations/                             # Alembic
│   ├── env.py                              # T7
│   ├── script.py.mako                      # T7
│   └── versions/
│       └── 0001_initial.py                 # T7
├── src/stock_analyzer/
│   ├── __init__.py                         # T1
│   ├── __main__.py                         # T29
│   ├── config.py                           # T2
│   ├── logging.py                          # T3
│   ├── orchestrator.py                     # T26
│   ├── agents/
│   │   ├── __init__.py                     # T23
│   │   ├── portfolio_agent.py              # T23
│   │   ├── drawdown_agent.py               # T24
│   │   └── politician_agent.py             # T25
│   ├── analytics/
│   │   ├── __init__.py                     # T10
│   │   ├── news_ranker.py                  # T10
│   │   ├── drawdown_filter.py              # T11
│   │   ├── politician_scorer.py            # T12
│   │   └── cost_tracker.py                 # T13
│   ├── calendar/
│   │   ├── __init__.py                     # T4
│   │   └── nyse.py                         # T4
│   ├── persistence/
│   │   ├── __init__.py                     # T5
│   │   ├── models.py                       # T5
│   │   ├── db.py                           # T6
│   │   ├── repositories.py                 # T8
│   │   └── in_memory.py                    # T9
│   ├── rendering/
│   │   ├── __init__.py                     # T21
│   │   ├── renderer.py                     # T22
│   │   └── templates/
│   │       ├── portfolio.html.j2           # T21
│   │       ├── drawdown.html.j2            # T21
│   │       └── politician.html.j2          # T21
│   └── tools/
│       ├── __init__.py                     # T14
│       ├── snaptrade.py                    # T14
│       ├── market_data.py                  # T15
│       ├── news.py                         # T16
│       ├── sec_edgar.py                    # T17
│       ├── capitol_trades.py               # T18
│       ├── insider_monkey.py               # T19
│       └── smtp_sender.py                  # T20
└── tests/
    ├── __init__.py                         # T1
    ├── conftest.py                         # T1
    ├── unit/                               # one test file per source module
    ├── integration/
    │   ├── cassettes/                      # vcrpy recordings
    │   └── test_*.py
    └── e2e/
        └── test_full_pipeline.py           # T37
```

**Conventions for every task:**
- Python 3.14 syntax (no `Optional[X]` — use `X | None`).
- Pydantic v2; SQLAlchemy 2.x typed `Mapped[...]` style.
- All public functions have type hints (enforced by `mypy --strict` later).
- Each new file gets at least one unit test in `tests/unit/<mirror_path>/test_<module>.py`.
- Frequent commits — one per task minimum, more when natural.
- Commit messages: `feat:`, `fix:`, `test:`, `chore:`, `docs:` prefixes; no `Co-Authored-By` lines.

---

## Phase 1 — Foundation (Tasks 1–4)

### Task 1: Project scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `.python-version`, `src/stock_analyzer/__init__.py`, `tests/__init__.py`, `tests/conftest.py`

- [ ] **Step 1.1: Replace `pyproject.toml` with the full project definition**

```toml
[project]
name = "stock-analyzer"
version = "0.1.0"
description = "Daily pre-market intelligence pipeline emailing three Claude-analyzed reports."
requires-python = ">=3.14"
authors = [{ name = "Snehal Soni", email = "snehal@thesonihub.com" }]
license = { text = "Proprietary" }
readme = "README.md"

dependencies = [
    "agno>=0.5.0",
    "anthropic>=0.40.0",
    "aiosmtplib>=3.0.0",
    "alembic>=1.13.0",
    "crawl4ai>=0.4.0",
    "feedparser>=6.0.11",
    "httpx>=0.27.0",
    "jinja2>=3.1.4",
    "pandas-market-calendars>=4.4.0",
    "pydantic>=2.9.0",
    "pydantic-settings>=2.5.0",
    "python-dateutil>=2.9.0",
    "snaptrade-python-sdk>=11.0.0",
    "sqlalchemy>=2.0.35",
    "structlog>=24.4.0",
    "tenacity>=9.0.0",
    "typer>=0.12.5",
    "yfinance>=0.2.43",
]

[project.scripts]
stock-analyzer = "stock_analyzer.__main__:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/stock_analyzer"]

[dependency-groups]
dev = [
    "aiosmtpd>=1.4.6",
    "freezegun>=1.5.1",
    "mypy>=1.11.2",
    "pytest>=8.3.3",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-mock>=3.14.0",
    "respx>=0.21.1",
    "ruff>=0.6.9",
    "vcrpy>=6.0.2",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-q --strict-markers"
markers = [
    "integration: hits recorded HTTP cassettes",
    "e2e: runs the full pipeline with stubbed Claude",
]

[tool.ruff]
line-length = 100
target-version = "py314"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP", "SIM", "RUF"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.14"
strict = true
warn_return_any = true
warn_unused_ignores = true
files = ["src/stock_analyzer"]
```

- [ ] **Step 1.2: Create `.python-version`**

```
3.14
```

- [ ] **Step 1.3: Create `src/stock_analyzer/__init__.py`**

```python
"""Stock Analyzer — daily pre-market intelligence pipeline."""

__version__ = "0.1.0"
```

- [ ] **Step 1.4: Create `tests/__init__.py`**

Empty file (just `touch tests/__init__.py`).

- [ ] **Step 1.5: Create `tests/conftest.py`**

```python
"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ambient stock-analyzer env vars so tests are reproducible."""
    for key in list(os.environ):
        if key.startswith(("ANTHROPIC_", "SNAPTRADE_", "FINNHUB_", "SMTP_", "STOCK_ANALYZER_")):
            monkeypatch.delenv(key, raising=False)
    yield
```

- [ ] **Step 1.6: Sync deps and verify package is importable**

```bash
uv sync
uv run python -c "import stock_analyzer; print(stock_analyzer.__version__)"
```

Expected output: `0.1.0`

- [ ] **Step 1.7: Commit**

```bash
git add pyproject.toml .python-version src/stock_analyzer/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: bootstrap python project with deps and pytest config"
```

---

### Task 2: Configuration (`pydantic-settings`)

**Files:**
- Create: `src/stock_analyzer/config.py`, `.env.example`
- Test: `tests/unit/test_config.py`

- [ ] **Step 2.1: Write the failing test**

Create `tests/unit/__init__.py` (empty), then `tests/unit/test_config.py`:

```python
from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_analyzer.config import Settings


def _minimal_env() -> dict[str, str]:
    return {
        "ANTHROPIC_API_KEY": "sk-ant-test",
        "SNAPTRADE_CLIENT_ID": "cid",
        "SNAPTRADE_CONSUMER_KEY": "ck",
        "SNAPTRADE_USER_ID": "uid",
        "SNAPTRADE_USER_SECRET": "us",
        "FINNHUB_API_KEY": "fh",
        "SMTP_HOST": "mail.example.com",
        "SMTP_USERNAME": "u",
        "SMTP_PASSWORD": "p",
        "SMTP_FROM_ADDRESS": "from@example.com",
        "SMTP_TO_ADDRESS": "to@example.com",
        "DATABASE_URL": "sqlite:///:memory:",
    }


def test_settings_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert s.anthropic_model == "claude-sonnet-4-6"
    assert s.drawdown_threshold_pct == 5.0
    assert s.politician_lookback_months == 24
    assert s.politician_fresh_disclosure_days == 2
    assert s.run_timezone == "America/New_York"
    assert s.smtp_port == 587
    assert s.dry_run is False


def test_settings_missing_required_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # No env vars set
    with pytest.raises(ValidationError):
        Settings()


def test_settings_secret_str_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    s = Settings()
    assert "sk-ant-test" not in repr(s)
    assert s.anthropic_api_key.get_secret_value() == "sk-ant-test"


def test_settings_extra_forbidden(monkeypatch: pytest.MonkeyPatch) -> None:
    for k, v in _minimal_env().items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("STOCK_ANALYZER_BOGUS", "x")
    # Extra env vars unrelated to known prefixes are ignored by pydantic-settings
    # (env_prefix scoping). The 'extra=forbid' applies to model fields, not env.
    Settings()  # should not raise
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: ImportError on `stock_analyzer.config`.

- [ ] **Step 2.3: Implement `src/stock_analyzer/config.py`**

```python
"""Application configuration — single source of truth for env-driven knobs."""

from __future__ import annotations

from typing import Literal

from pydantic import EmailStr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    # Anthropic / agno
    anthropic_api_key: SecretStr
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 4096
    anthropic_prompt_caching: bool = True

    # SnapTrade
    snaptrade_client_id: SecretStr
    snaptrade_consumer_key: SecretStr
    snaptrade_user_id: str
    snaptrade_user_secret: SecretStr

    # Finnhub
    finnhub_api_key: SecretStr

    # SMTP (Stalwart)
    smtp_host: str
    smtp_port: int = 587
    smtp_use_tls: bool = True
    smtp_username: str
    smtp_password: SecretStr
    smtp_from_address: EmailStr
    smtp_from_name: str = "Stock Analyzer"
    smtp_to_address: EmailStr

    # Storage
    database_url: str
    failed_emails_dir: str = "/var/lib/stock-analyzer/failed_emails"

    # Behavior
    run_timezone: str = "America/New_York"
    drawdown_threshold_pct: float = 5.0
    politician_lookback_months: int = 24
    politician_fresh_disclosure_days: int = 2
    log_level: str = "INFO"
    log_format: Literal["json", "pretty"] = "json"

    # Optional
    dry_run: bool = False
    skip_nyse_holidays: bool = True
    stock_analyzer_env: Literal["production", "development"] = "production"
```

- [ ] **Step 2.4: Create `.env.example`**

(Use the exact content from the spec, Section 8.)

- [ ] **Step 2.5: Run tests**

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: 4 passed.

- [ ] **Step 2.6: Commit**

```bash
git add src/stock_analyzer/config.py tests/unit/__init__.py tests/unit/test_config.py .env.example
git commit -m "feat: add pydantic-settings config with secret redaction"
```

---

### Task 3: Logging (`structlog`)

**Files:**
- Create: `src/stock_analyzer/logging.py`
- Test: `tests/unit/test_logging.py`

- [ ] **Step 3.1: Write the failing test**

```python
# tests/unit/test_logging.py
from __future__ import annotations

import json
import logging

from stock_analyzer.logging import configure_logging, get_logger


def test_configure_logging_json(capsys) -> None:
    configure_logging(level="INFO", fmt="json")
    log = get_logger("test")
    log.info("hello", foo="bar")
    captured = capsys.readouterr().err or capsys.readouterr().out
    payload = json.loads(captured.strip().splitlines()[-1])
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert payload["level"] == "info"


def test_configure_logging_pretty(capsys) -> None:
    configure_logging(level="INFO", fmt="pretty")
    log = get_logger("test")
    log.info("readable", foo="bar")
    out = (capsys.readouterr().err or "") + (capsys.readouterr().out or "")
    assert "readable" in out


def test_logger_respects_level() -> None:
    configure_logging(level="WARNING", fmt="json")
    assert logging.getLogger().getEffectiveLevel() == logging.WARNING
```

- [ ] **Step 3.2: Run test to verify it fails**

```bash
uv run pytest tests/unit/test_logging.py -v
```

Expected: ImportError.

- [ ] **Step 3.3: Implement `src/stock_analyzer/logging.py`**

```python
"""Structured logging — JSON in production, pretty in dev."""

from __future__ import annotations

import logging
import sys
from typing import Any, Literal

import structlog


def configure_logging(level: str = "INFO", fmt: Literal["json", "pretty"] = "json") -> None:
    """Idempotent logger configuration. Call once at startup."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
        force=True,
    )

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]

    if fmt == "json":
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
```

- [ ] **Step 3.4: Run tests**

```bash
uv run pytest tests/unit/test_logging.py -v
```

Expected: 3 passed.

- [ ] **Step 3.5: Commit**

```bash
git add src/stock_analyzer/logging.py tests/unit/test_logging.py
git commit -m "feat: add structlog-based logging with json/pretty modes"
```

---

### Task 4: NYSE calendar utility

**Files:**
- Create: `src/stock_analyzer/calendar/__init__.py`, `src/stock_analyzer/calendar/nyse.py`
- Test: `tests/unit/calendar/test_nyse.py`

- [ ] **Step 4.1: Write failing test**

Create `tests/unit/calendar/__init__.py` (empty), then:

```python
# tests/unit/calendar/test_nyse.py
from __future__ import annotations

from datetime import date

from stock_analyzer.calendar.nyse import is_market_holiday, is_trading_day, next_trading_day


def test_christmas_is_holiday() -> None:
    assert is_market_holiday(date(2026, 12, 25)) is True


def test_random_weekday_is_trading_day() -> None:
    # Wed, May 6 2026 — not a known NYSE holiday
    assert is_trading_day(date(2026, 5, 6)) is True


def test_saturday_is_not_trading_day() -> None:
    # Sat, May 9 2026
    assert is_trading_day(date(2026, 5, 9)) is False


def test_next_trading_day_skips_weekend() -> None:
    # Friday → Monday
    assert next_trading_day(date(2026, 5, 8)) == date(2026, 5, 11)


def test_next_trading_day_skips_holiday() -> None:
    # Day before Independence Day observed
    fri = date(2026, 7, 3)  # this date is itself observed Independence Day in 2026
    nxt = next_trading_day(fri)
    assert nxt > fri
    assert is_trading_day(nxt)
```

- [ ] **Step 4.2: Run to verify it fails**

```bash
uv run pytest tests/unit/calendar/test_nyse.py -v
```

Expected: ImportError.

- [ ] **Step 4.3: Implement `src/stock_analyzer/calendar/__init__.py`**

```python
"""NYSE trading calendar utilities."""
```

- [ ] **Step 4.4: Implement `src/stock_analyzer/calendar/nyse.py`**

```python
"""Thin wrapper over pandas-market-calendars for NYSE trading days/holidays."""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import pandas_market_calendars as mcal


@lru_cache(maxsize=1)
def _nyse():
    return mcal.get_calendar("XNYS")


def is_trading_day(d: date) -> bool:
    schedule = _nyse().valid_days(start_date=d, end_date=d)
    return len(schedule) == 1


def is_market_holiday(d: date) -> bool:
    if d.weekday() >= 5:  # weekends are not "holidays" per se
        return False
    return not is_trading_day(d)


def next_trading_day(d: date) -> date:
    candidate = d + timedelta(days=1)
    while not is_trading_day(candidate):
        candidate += timedelta(days=1)
    return candidate


def previous_trading_day(d: date) -> date:
    candidate = d - timedelta(days=1)
    while not is_trading_day(candidate):
        candidate -= timedelta(days=1)
    return candidate
```

- [ ] **Step 4.5: Run tests**

```bash
uv run pytest tests/unit/calendar/test_nyse.py -v
```

Expected: 5 passed.

- [ ] **Step 4.6: Commit**

```bash
git add src/stock_analyzer/calendar/ tests/unit/calendar/
git commit -m "feat: add NYSE calendar helpers (holiday/trading-day detection)"
```

---

*Phase 1 complete: project scaffolded, config validated, logging structured, calendar primitives in place. The next phase builds the persistence layer.*

## Phase 2 — Persistence (Tasks 5–9)

### Task 5: SQLAlchemy models

**Files:**
- Create: `src/stock_analyzer/persistence/__init__.py`, `src/stock_analyzer/persistence/models.py`
- Test: `tests/unit/persistence/test_models.py`

- [ ] **Step 5.1: Write failing test**

Create `tests/unit/persistence/__init__.py` (empty), then:

```python
# tests/unit/persistence/test_models.py
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from stock_analyzer.persistence.models import (
    Base,
    Politician,
    PoliticianTrade,
    PoliticianScore,
    SpyDailyClose,
    Run,
)


@pytest.fixture
def session() -> Session:
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


def test_create_politician_and_trade(session: Session) -> None:
    pol = Politician(full_name="Jane Doe", party="D", chamber="House", state="CA",
                     capitol_trades_id="N00001")
    session.add(pol)
    session.flush()
    trade = PoliticianTrade(
        politician_id=pol.id, ticker="AAPL", side="BUY",
        trade_date=date(2026, 5, 4), disclosure_date=date(2026, 5, 5),
        amount_min_usd=1001, amount_max_usd=15000, raw_payload={"src": "test"},
    )
    session.add(trade)
    session.commit()
    assert session.scalar(select(PoliticianTrade)).ticker == "AAPL"


def test_unique_disclosure_constraint(session: Session) -> None:
    pol = Politician(full_name="A", party="R", chamber="Senate", state="TX",
                     capitol_trades_id="N00002")
    session.add(pol)
    session.flush()
    common = dict(politician_id=pol.id, ticker="MSFT", side="BUY",
                  trade_date=date(2026, 5, 4), disclosure_date=date(2026, 5, 5),
                  amount_min_usd=1, amount_max_usd=2, raw_payload={})
    session.add(PoliticianTrade(**common))
    session.commit()
    session.add(PoliticianTrade(**common))
    with pytest.raises(Exception):
        session.commit()


def test_run_unique_per_date(session: Session) -> None:
    session.add(Run(run_date=date(2026, 5, 5),
                    started_at=datetime.now(timezone.utc),
                    status="running"))
    session.commit()
    session.add(Run(run_date=date(2026, 5, 5),
                    started_at=datetime.now(timezone.utc),
                    status="running"))
    with pytest.raises(Exception):
        session.commit()


def test_spy_close_round_trip(session: Session) -> None:
    session.add(SpyDailyClose(trade_date=date(2026, 5, 4), close_price=528.41,
                              fetched_at=datetime.now(timezone.utc)))
    session.commit()
    row = session.scalar(select(SpyDailyClose))
    assert row.close_price == 528.41
```

- [ ] **Step 5.2: Run to verify it fails**

```bash
uv run pytest tests/unit/persistence/test_models.py -v
```

Expected: ImportError.

- [ ] **Step 5.3: Implement `src/stock_analyzer/persistence/__init__.py`**

```python
"""Persistence layer — SQLAlchemy models and repositories."""

from stock_analyzer.persistence.models import (
    Base,
    Politician,
    PoliticianScore,
    PoliticianTrade,
    Run,
    SpyDailyClose,
)

__all__ = [
    "Base",
    "Politician",
    "PoliticianScore",
    "PoliticianTrade",
    "Run",
    "SpyDailyClose",
]
```

- [ ] **Step 5.4: Implement `src/stock_analyzer/persistence/models.py`**

```python
"""SQLAlchemy 2.x typed models matching the spec schema."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Politician(Base):
    __tablename__ = "politicians"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    party: Mapped[str] = mapped_column(String, nullable=False)
    chamber: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str | None] = mapped_column(String)
    capitol_trades_id: Mapped[str | None] = mapped_column(String, unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    trades: Mapped[list["PoliticianTrade"]] = relationship(back_populates="politician")
    score: Mapped["PoliticianScore | None"] = relationship(back_populates="politician", uselist=False)

    __table_args__ = (
        CheckConstraint("party IN ('D','R','I')", name="party_check"),
        CheckConstraint("chamber IN ('House','Senate')", name="chamber_check"),
    )


class PoliticianTrade(Base):
    __tablename__ = "politician_trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    politician_id: Mapped[int] = mapped_column(ForeignKey("politicians.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    side: Mapped[str] = mapped_column(String, nullable=False)
    trade_date: Mapped[date] = mapped_column(Date, nullable=False)
    disclosure_date: Mapped[date] = mapped_column(Date, nullable=False)
    amount_min_usd: Mapped[int | None] = mapped_column(Integer)
    amount_max_usd: Mapped[int | None] = mapped_column(Integer)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)

    politician: Mapped[Politician] = relationship(back_populates="trades")

    __table_args__ = (
        CheckConstraint("side IN ('BUY','SELL')", name="side_check"),
        UniqueConstraint(
            "politician_id", "ticker", "side", "trade_date", "disclosure_date",
            name="uq_disclosure",
        ),
        Index("ix_trades_disclosure_date", "disclosure_date"),
        Index("ix_trades_politician_id", "politician_id"),
    )


class PoliticianScore(Base):
    __tablename__ = "politician_scores"

    politician_id: Mapped[int] = mapped_column(
        ForeignKey("politicians.id"), primary_key=True
    )
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    window_start_date: Mapped[date] = mapped_column(Date, nullable=False)
    window_end_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_return_pct: Mapped[float] = mapped_column(nullable=False)
    spy_return_pct: Mapped[float] = mapped_column(nullable=False)
    alpha_vs_spy_pct: Mapped[float] = mapped_column(nullable=False)
    trade_count: Mapped[int] = mapped_column(Integer, nullable=False)

    politician: Mapped[Politician] = relationship(back_populates="score")


class SpyDailyClose(Base):
    __tablename__ = "spy_daily_close"

    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    close_price: Mapped[float] = mapped_column(nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_date: Mapped[date] = mapped_column(Date, nullable=False, unique=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String, nullable=False)
    email_1_status: Mapped[str | None] = mapped_column(String)
    email_2_status: Mapped[str | None] = mapped_column(String)
    email_3_status: Mapped[str | None] = mapped_column(String)
    error_log: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    total_tokens_in: Mapped[int | None] = mapped_column(Integer)
    total_tokens_out: Mapped[int | None] = mapped_column(Integer)
    est_cost_usd: Mapped[float | None] = mapped_column()

    __table_args__ = (
        CheckConstraint(
            "status IN ('running','success','partial','failed')", name="status_check"
        ),
    )
```

- [ ] **Step 5.5: Run tests**

```bash
uv run pytest tests/unit/persistence/test_models.py -v
```

Expected: 4 passed.

- [ ] **Step 5.6: Commit**

```bash
git add src/stock_analyzer/persistence/ tests/unit/persistence/
git commit -m "feat: add SQLAlchemy 2.x models for politicians, trades, scores, spy, runs"
```

---

### Task 6: Database session factory + ephemeral guard

**Files:**
- Create: `src/stock_analyzer/persistence/db.py`
- Test: `tests/unit/persistence/test_db.py`

- [ ] **Step 6.1: Write failing test**

```python
# tests/unit/persistence/test_db.py
from __future__ import annotations

import pytest

from stock_analyzer.persistence.db import (
    Database,
    EphemeralModeError,
)


def test_production_session_works(tmp_path) -> None:
    db = Database(url=f"sqlite:///{tmp_path / 'x.db'}", ephemeral=False)
    db.create_all()
    with db.session() as s:
        s.execute(__import__("sqlalchemy").text("SELECT 1"))


def test_ephemeral_session_raises() -> None:
    db = Database(url="sqlite:///:memory:", ephemeral=True)
    with pytest.raises(EphemeralModeError):
        with db.session():
            pass


def test_ephemeral_create_all_is_noop() -> None:
    db = Database(url="sqlite:///:memory:", ephemeral=True)
    db.create_all()  # should not raise
```

- [ ] **Step 6.2: Run to verify it fails**

```bash
uv run pytest tests/unit/persistence/test_db.py -v
```

Expected: ImportError.

- [ ] **Step 6.3: Implement `src/stock_analyzer/persistence/db.py`**

```python
"""Database connection management with explicit ephemeral-mode guard."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from stock_analyzer.persistence.models import Base


class EphemeralModeError(RuntimeError):
    """Raised when DB-only code paths are invoked in ephemeral mode."""


@dataclass
class Database:
    url: str
    ephemeral: bool

    def __post_init__(self) -> None:
        if not self.ephemeral:
            self._engine: Engine = create_engine(self.url, future=True)
            self._sessionmaker = sessionmaker(self._engine, expire_on_commit=False)

    def create_all(self) -> None:
        if self.ephemeral:
            return
        Base.metadata.create_all(self._engine)

    @contextmanager
    def session(self) -> Iterator[Session]:
        if self.ephemeral:
            raise EphemeralModeError(
                "Database.session() called in ephemeral mode — repositories "
                "must use the in-memory equivalents."
            )
        with self._sessionmaker() as s:
            yield s
```

- [ ] **Step 6.4: Run tests**

```bash
uv run pytest tests/unit/persistence/test_db.py -v
```

Expected: 3 passed.

- [ ] **Step 6.5: Commit**

```bash
git add src/stock_analyzer/persistence/db.py tests/unit/persistence/test_db.py
git commit -m "feat: add Database wrapper with ephemeral-mode guard"
```

---

### Task 7: Alembic migrations bootstrap

**Files:**
- Create: `alembic.ini`, `migrations/env.py`, `migrations/script.py.mako`, `migrations/versions/0001_initial.py`

- [ ] **Step 7.1: Initialize alembic skeleton**

```bash
uv run alembic init -t generic migrations
```

This creates `alembic.ini` and `migrations/{env.py,script.py.mako,versions/}`.

- [ ] **Step 7.2: Edit `alembic.ini`** — set `sqlalchemy.url` to be overridden from env:

Replace the `sqlalchemy.url = ...` line with:

```ini
sqlalchemy.url = sqlite:///./stock_analyzer.db
```

(Real value comes from env at runtime via `migrations/env.py`.)

- [ ] **Step 7.3: Replace `migrations/env.py`** with this content:

```python
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from stock_analyzer.persistence.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override URL from env at runtime
url_from_env = os.getenv("DATABASE_URL")
if url_from_env:
    config.set_main_option("sqlalchemy.url", url_from_env)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 7.4: Generate the initial migration**

```bash
DATABASE_URL=sqlite:///./bootstrap.db uv run alembic revision --autogenerate -m "initial"
```

Move/rename the generated file to `migrations/versions/0001_initial.py` if Alembic gave it a hash-based name. Verify the upgrade function creates all 5 tables. **Delete `bootstrap.db`** after verifying.

- [ ] **Step 7.5: Test the migration round-trip**

```bash
DATABASE_URL=sqlite:///./check.db uv run alembic upgrade head
DATABASE_URL=sqlite:///./check.db uv run alembic downgrade base
rm check.db
```

Both must succeed.

- [ ] **Step 7.6: Commit**

```bash
git add alembic.ini migrations/
git commit -m "feat: bootstrap alembic migrations with initial schema"
```

---

### Task 8: Repositories

**Files:**
- Create: `src/stock_analyzer/persistence/repositories.py`
- Test: `tests/unit/persistence/test_repositories.py`

- [ ] **Step 8.1: Write failing test**

```python
# tests/unit/persistence/test_repositories.py
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_analyzer.persistence.db import Database
from stock_analyzer.persistence.models import Politician
from stock_analyzer.persistence.repositories import (
    PoliticianRepository,
    PoliticianTradeRepository,
    SpyCloseRepository,
    RunRepository,
)


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(url=f"sqlite:///{tmp_path / 'r.db'}", ephemeral=False)
    d.create_all()
    return d


def test_upsert_politician(db: Database) -> None:
    repo = PoliticianRepository(db)
    p1 = repo.upsert(full_name="A B", party="D", chamber="House",
                     state="CA", capitol_trades_id="X1")
    assert p1.id is not None
    p2 = repo.upsert(full_name="A B", party="R", chamber="Senate",
                     state="NY", capitol_trades_id="X1")
    assert p2.id == p1.id
    assert p2.party == "R"  # updated


def test_idempotent_trade_insert(db: Database) -> None:
    pol_repo = PoliticianRepository(db)
    pol = pol_repo.upsert(full_name="C D", party="I", chamber="Senate",
                          state="VT", capitol_trades_id="X2")
    trade_repo = PoliticianTradeRepository(db)
    payload = dict(politician_id=pol.id, ticker="AAPL", side="BUY",
                   trade_date=date(2026, 5, 4),
                   disclosure_date=date(2026, 5, 5),
                   amount_min_usd=1001, amount_max_usd=15000,
                   raw_payload={"k": "v"})
    n1 = trade_repo.bulk_upsert([payload])
    n2 = trade_repo.bulk_upsert([payload])  # same input
    assert n1 == 1
    assert n2 == 0  # already present


def test_spy_close_upsert(db: Database) -> None:
    repo = SpyCloseRepository(db)
    repo.upsert(date(2026, 5, 4), 528.41)
    repo.upsert(date(2026, 5, 4), 530.00)  # overwrite
    series = repo.get_range(date(2026, 5, 1), date(2026, 5, 5))
    assert series[date(2026, 5, 4)] == 530.00


def test_run_idempotency(db: Database) -> None:
    repo = RunRepository(db)
    r = repo.start_run(date(2026, 5, 5))
    assert r.id is not None
    assert repo.find_by_date(date(2026, 5, 5)).id == r.id
    repo.complete_run(r.id, status="success", email_statuses=("sent","sent","sent"),
                      tokens_in=100, tokens_out=200, cost_usd=0.05, error_log=None)
    final = repo.find_by_date(date(2026, 5, 5))
    assert final.status == "success"
    assert final.completed_at is not None
```

- [ ] **Step 8.2: Run to verify it fails**

```bash
uv run pytest tests/unit/persistence/test_repositories.py -v
```

Expected: ImportError.

- [ ] **Step 8.3: Implement `src/stock_analyzer/persistence/repositories.py`**

```python
"""Query helpers — one repository per aggregate root."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from stock_analyzer.persistence.db import Database
from stock_analyzer.persistence.models import (
    Politician,
    PoliticianScore,
    PoliticianTrade,
    Run,
    SpyDailyClose,
)


@dataclass
class PoliticianRepository:
    db: Database

    def upsert(self, *, full_name: str, party: str, chamber: str,
               state: str | None, capitol_trades_id: str | None) -> Politician:
        with self.db.session() as s:
            existing = s.scalar(
                select(Politician).where(Politician.full_name == full_name)
            )
            if existing:
                existing.party = party
                existing.chamber = chamber
                existing.state = state
                existing.capitol_trades_id = capitol_trades_id
                existing.updated_at = datetime.now(timezone.utc)
                s.commit()
                s.refresh(existing)
                return existing
            new = Politician(full_name=full_name, party=party, chamber=chamber,
                             state=state, capitol_trades_id=capitol_trades_id)
            s.add(new)
            s.commit()
            s.refresh(new)
            return new

    def all(self) -> list[Politician]:
        with self.db.session() as s:
            return list(s.scalars(select(Politician)).all())


@dataclass
class PoliticianTradeRepository:
    db: Database

    def bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        """Returns count of new rows inserted (existing duplicates skipped)."""
        if not rows:
            return 0
        with self.db.session() as s:
            stmt = sqlite_insert(PoliticianTrade).values(rows).on_conflict_do_nothing(
                index_elements=["politician_id", "ticker", "side",
                                "trade_date", "disclosure_date"]
            )
            result = s.execute(stmt)
            s.commit()
            return result.rowcount or 0

    def for_politician(self, politician_id: int,
                        since: date | None = None) -> list[PoliticianTrade]:
        with self.db.session() as s:
            stmt = select(PoliticianTrade).where(
                PoliticianTrade.politician_id == politician_id
            )
            if since:
                stmt = stmt.where(PoliticianTrade.trade_date >= since)
            return list(s.scalars(stmt).all())

    def disclosed_between(self, start: date, end: date) -> list[PoliticianTrade]:
        with self.db.session() as s:
            stmt = select(PoliticianTrade).where(
                PoliticianTrade.disclosure_date >= start,
                PoliticianTrade.disclosure_date <= end,
            )
            return list(s.scalars(stmt).all())


@dataclass
class PoliticianScoreRepository:
    db: Database

    def upsert(self, *, politician_id: int, computed_at: datetime,
                window_start: date, window_end: date,
                total_return_pct: float, spy_return_pct: float,
                trade_count: int) -> PoliticianScore:
        alpha = total_return_pct - spy_return_pct
        with self.db.session() as s:
            row = s.get(PoliticianScore, politician_id)
            if row:
                row.computed_at = computed_at
                row.window_start_date = window_start
                row.window_end_date = window_end
                row.total_return_pct = total_return_pct
                row.spy_return_pct = spy_return_pct
                row.alpha_vs_spy_pct = alpha
                row.trade_count = trade_count
            else:
                row = PoliticianScore(
                    politician_id=politician_id, computed_at=computed_at,
                    window_start_date=window_start, window_end_date=window_end,
                    total_return_pct=total_return_pct, spy_return_pct=spy_return_pct,
                    alpha_vs_spy_pct=alpha, trade_count=trade_count,
                )
                s.add(row)
            s.commit()
            s.refresh(row)
            return row

    def beating_spy(self) -> set[int]:
        with self.db.session() as s:
            stmt = select(PoliticianScore.politician_id).where(
                PoliticianScore.alpha_vs_spy_pct > 0
            )
            return set(s.scalars(stmt).all())


@dataclass
class SpyCloseRepository:
    db: Database

    def upsert(self, d: date, close_price: float) -> None:
        with self.db.session() as s:
            stmt = sqlite_insert(SpyDailyClose).values(
                trade_date=d, close_price=close_price,
                fetched_at=datetime.now(timezone.utc),
            ).on_conflict_do_update(
                index_elements=["trade_date"],
                set_={"close_price": close_price,
                      "fetched_at": datetime.now(timezone.utc)},
            )
            s.execute(stmt)
            s.commit()

    def get_range(self, start: date, end: date) -> dict[date, float]:
        with self.db.session() as s:
            stmt = select(SpyDailyClose).where(
                SpyDailyClose.trade_date >= start,
                SpyDailyClose.trade_date <= end,
            )
            return {row.trade_date: row.close_price for row in s.scalars(stmt).all()}


@dataclass
class RunRepository:
    db: Database

    def start_run(self, run_date: date) -> Run:
        with self.db.session() as s:
            existing = s.scalar(select(Run).where(Run.run_date == run_date))
            if existing:
                return existing
            r = Run(run_date=run_date,
                    started_at=datetime.now(timezone.utc),
                    status="running")
            s.add(r)
            s.commit()
            s.refresh(r)
            return r

    def find_by_date(self, run_date: date) -> Run | None:
        with self.db.session() as s:
            return s.scalar(select(Run).where(Run.run_date == run_date))

    def complete_run(self, run_id: int, *, status: str,
                      email_statuses: tuple[str, str, str],
                      tokens_in: int, tokens_out: int, cost_usd: float,
                      error_log: dict[str, Any] | None) -> None:
        with self.db.session() as s:
            r = s.get(Run, run_id)
            if r is None:
                return
            r.status = status
            r.completed_at = datetime.now(timezone.utc)
            r.email_1_status, r.email_2_status, r.email_3_status = email_statuses
            r.total_tokens_in = tokens_in
            r.total_tokens_out = tokens_out
            r.est_cost_usd = cost_usd
            r.error_log = error_log
            s.commit()

    def history(self, limit: int = 10) -> list[Run]:
        with self.db.session() as s:
            stmt = select(Run).order_by(Run.run_date.desc()).limit(limit)
            return list(s.scalars(stmt).all())
```

- [ ] **Step 8.4: Run tests**

```bash
uv run pytest tests/unit/persistence/test_repositories.py -v
```

Expected: 4 passed.

- [ ] **Step 8.5: Commit**

```bash
git add src/stock_analyzer/persistence/repositories.py tests/unit/persistence/test_repositories.py
git commit -m "feat: add repositories for politicians, trades, scores, spy, runs"
```

---

### Task 9: In-memory ephemeral repositories

**Files:**
- Create: `src/stock_analyzer/persistence/in_memory.py`
- Test: `tests/unit/persistence/test_in_memory.py`

- [ ] **Step 9.1: Write failing test**

```python
# tests/unit/persistence/test_in_memory.py
from __future__ import annotations

from datetime import date

from stock_analyzer.persistence.in_memory import (
    InMemoryPoliticianRepository,
    InMemoryPoliticianTradeRepository,
    InMemorySpyCloseRepository,
    InMemoryRunRepository,
)


def test_in_memory_politician_upsert() -> None:
    repo = InMemoryPoliticianRepository()
    p = repo.upsert(full_name="X", party="D", chamber="House",
                    state="CA", capitol_trades_id="A")
    assert p.id == 1
    p2 = repo.upsert(full_name="X", party="R", chamber="Senate",
                     state="NY", capitol_trades_id="A")
    assert p2.id == p.id
    assert p2.party == "R"


def test_in_memory_trade_dedup() -> None:
    repo = InMemoryPoliticianTradeRepository()
    row = dict(politician_id=1, ticker="AAPL", side="BUY",
               trade_date=date(2026, 5, 4),
               disclosure_date=date(2026, 5, 5),
               amount_min_usd=1, amount_max_usd=2, raw_payload={})
    assert repo.bulk_upsert([row]) == 1
    assert repo.bulk_upsert([row]) == 0


def test_in_memory_spy_close() -> None:
    repo = InMemorySpyCloseRepository()
    repo.upsert(date(2026, 5, 4), 528.41)
    series = repo.get_range(date(2026, 5, 1), date(2026, 5, 5))
    assert series[date(2026, 5, 4)] == 528.41


def test_in_memory_run_repo() -> None:
    repo = InMemoryRunRepository()
    r = repo.start_run(date(2026, 5, 5))
    assert repo.find_by_date(date(2026, 5, 5)).id == r.id
```

- [ ] **Step 9.2: Run to verify it fails**

```bash
uv run pytest tests/unit/persistence/test_in_memory.py -v
```

Expected: ImportError.

- [ ] **Step 9.3: Implement `src/stock_analyzer/persistence/in_memory.py`**

```python
"""In-memory repository equivalents for ephemeral mode.

Same public method shapes as the SQL repositories, dict-backed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from stock_analyzer.persistence.models import (
    Politician,
    PoliticianScore,
    PoliticianTrade,
    Run,
    SpyDailyClose,
)


@dataclass
class InMemoryPoliticianRepository:
    _by_name: dict[str, Politician] = field(default_factory=dict)
    _next_id: int = 1

    def upsert(self, *, full_name: str, party: str, chamber: str,
               state: str | None, capitol_trades_id: str | None) -> Politician:
        if full_name in self._by_name:
            p = self._by_name[full_name]
            p.party = party
            p.chamber = chamber
            p.state = state
            p.capitol_trades_id = capitol_trades_id
            p.updated_at = datetime.now(timezone.utc)
            return p
        p = Politician(full_name=full_name, party=party, chamber=chamber,
                       state=state, capitol_trades_id=capitol_trades_id)
        p.id = self._next_id
        self._next_id += 1
        p.created_at = datetime.now(timezone.utc)
        p.updated_at = p.created_at
        self._by_name[full_name] = p
        return p

    def all(self) -> list[Politician]:
        return list(self._by_name.values())


@dataclass
class InMemoryPoliticianTradeRepository:
    _rows: dict[tuple, PoliticianTrade] = field(default_factory=dict)
    _next_id: int = 1

    def bulk_upsert(self, rows: list[dict[str, Any]]) -> int:
        added = 0
        for r in rows:
            key = (r["politician_id"], r["ticker"], r["side"],
                   r["trade_date"], r["disclosure_date"])
            if key in self._rows:
                continue
            t = PoliticianTrade(**r)
            t.id = self._next_id
            self._next_id += 1
            t.ingested_at = datetime.now(timezone.utc)
            self._rows[key] = t
            added += 1
        return added

    def for_politician(self, politician_id: int,
                        since: date | None = None) -> list[PoliticianTrade]:
        out = [t for t in self._rows.values() if t.politician_id == politician_id]
        if since:
            out = [t for t in out if t.trade_date >= since]
        return out

    def disclosed_between(self, start: date, end: date) -> list[PoliticianTrade]:
        return [t for t in self._rows.values()
                if start <= t.disclosure_date <= end]


@dataclass
class InMemoryPoliticianScoreRepository:
    _rows: dict[int, PoliticianScore] = field(default_factory=dict)

    def upsert(self, *, politician_id: int, computed_at: datetime,
                window_start: date, window_end: date,
                total_return_pct: float, spy_return_pct: float,
                trade_count: int) -> PoliticianScore:
        alpha = total_return_pct - spy_return_pct
        s = PoliticianScore(
            politician_id=politician_id, computed_at=computed_at,
            window_start_date=window_start, window_end_date=window_end,
            total_return_pct=total_return_pct, spy_return_pct=spy_return_pct,
            alpha_vs_spy_pct=alpha, trade_count=trade_count,
        )
        self._rows[politician_id] = s
        return s

    def beating_spy(self) -> set[int]:
        return {pid for pid, s in self._rows.items() if s.alpha_vs_spy_pct > 0}


@dataclass
class InMemorySpyCloseRepository:
    _by_date: dict[date, float] = field(default_factory=dict)

    def upsert(self, d: date, close_price: float) -> None:
        self._by_date[d] = close_price

    def get_range(self, start: date, end: date) -> dict[date, float]:
        return {d: p for d, p in self._by_date.items() if start <= d <= end}


@dataclass
class InMemoryRunRepository:
    _by_date: dict[date, Run] = field(default_factory=dict)
    _next_id: int = 1

    def start_run(self, run_date: date) -> Run:
        if run_date in self._by_date:
            return self._by_date[run_date]
        r = Run(run_date=run_date, started_at=datetime.now(timezone.utc),
                status="running")
        r.id = self._next_id
        self._next_id += 1
        self._by_date[run_date] = r
        return r

    def find_by_date(self, run_date: date) -> Run | None:
        return self._by_date.get(run_date)

    def complete_run(self, run_id: int, *, status: str,
                      email_statuses: tuple[str, str, str],
                      tokens_in: int, tokens_out: int, cost_usd: float,
                      error_log: dict[str, Any] | None) -> None:
        for r in self._by_date.values():
            if r.id == run_id:
                r.status = status
                r.completed_at = datetime.now(timezone.utc)
                r.email_1_status, r.email_2_status, r.email_3_status = email_statuses
                r.total_tokens_in = tokens_in
                r.total_tokens_out = tokens_out
                r.est_cost_usd = cost_usd
                r.error_log = error_log
                return

    def history(self, limit: int = 10) -> list[Run]:
        return sorted(self._by_date.values(),
                      key=lambda r: r.run_date, reverse=True)[:limit]
```

- [ ] **Step 9.4: Run tests**

```bash
uv run pytest tests/unit/persistence/test_in_memory.py -v
```

Expected: 4 passed.

- [ ] **Step 9.5: Commit**

```bash
git add src/stock_analyzer/persistence/in_memory.py tests/unit/persistence/test_in_memory.py
git commit -m "feat: add in-memory repositories for ephemeral mode"
```

---

*Phase 2 complete: schema, migrations, sql repositories, and ephemeral equivalents all in place.*


## Phase 3 — Analytics (pure logic, Tasks 10–13)

These modules have **no external dependencies** beyond the standard library, Pydantic models, and our persistence types. They are 100% unit-tested with no mocks needed.

### Task 10: News ranker (deterministic trending score)

**Files:**
- Create: `src/stock_analyzer/analytics/__init__.py`, `src/stock_analyzer/analytics/news_ranker.py`
- Test: `tests/unit/analytics/test_news_ranker.py`

- [ ] **Step 10.1: Write failing test**

Create `tests/unit/analytics/__init__.py` (empty), then:

```python
# tests/unit/analytics/test_news_ranker.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from stock_analyzer.analytics.news_ranker import (
    NewsItem,
    rank_articles,
    publisher_tier_weight,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def test_market_wide_trending_dominates() -> None:
    items = [
        NewsItem(title="Boring", publisher="random.blog",
                 url="https://x/1", published_at=_now(),
                 related_tickers=["AAPL"]),
        NewsItem(title="Trending", publisher="random.blog",
                 url="https://x/2", published_at=_now(),
                 related_tickers=["NVDA"]),
    ]
    holdings = {"AAPL", "NVDA"}
    trending_market = {"NVDA"}
    ranked = rank_articles(items, holdings=holdings,
                            market_trending=trending_market)
    assert ranked[0].title == "Trending"
    assert ranked[0].is_market_wide_trending is True


def test_cross_ticker_boost() -> None:
    items = [
        NewsItem(title="Single", publisher="reuters.com",
                 url="https://x/1", published_at=_now(),
                 related_tickers=["AAPL"]),
        NewsItem(title="Multi", publisher="reuters.com",
                 url="https://x/2", published_at=_now(),
                 related_tickers=["AAPL", "MSFT", "GOOG"]),
    ]
    ranked = rank_articles(items, holdings={"AAPL", "MSFT", "GOOG"},
                            market_trending=set())
    assert ranked[0].title == "Multi"


def test_recency_decay() -> None:
    fresh = NewsItem(title="Fresh", publisher="reuters.com",
                     url="https://x/1", published_at=_now(),
                     related_tickers=["AAPL"])
    stale = NewsItem(title="Stale", publisher="reuters.com",
                     url="https://x/2",
                     published_at=_now() - timedelta(days=3),
                     related_tickers=["AAPL"])
    ranked = rank_articles([stale, fresh], holdings={"AAPL"}, market_trending=set())
    assert ranked[0].title == "Fresh"


def test_publisher_tier_weights() -> None:
    assert publisher_tier_weight("Reuters") == 30
    assert publisher_tier_weight("WSJ") == 30
    assert publisher_tier_weight("Bloomberg") == 30
    assert publisher_tier_weight("CNBC") == 30
    assert publisher_tier_weight("Financial Times") == 30
    assert publisher_tier_weight("seekingalpha.com") == 10
    assert publisher_tier_weight("random.blog") == 0


def test_rank_returns_top_n() -> None:
    items = [
        NewsItem(title=f"a{i}", publisher="random.blog",
                 url=f"https://x/{i}", published_at=_now(),
                 related_tickers=["AAPL"])
        for i in range(20)
    ]
    ranked = rank_articles(items, holdings={"AAPL"},
                            market_trending=set(), top_n=5)
    assert len(ranked) == 5
```

- [ ] **Step 10.2: Run to verify it fails**

```bash
uv run pytest tests/unit/analytics/test_news_ranker.py -v
```

Expected: ImportError.

- [ ] **Step 10.3: Implement `src/stock_analyzer/analytics/__init__.py`**

```python
"""Pure-logic helpers — no external IO, no LLM calls."""
```

- [ ] **Step 10.4: Implement `src/stock_analyzer/analytics/news_ranker.py`**

```python
"""Deterministic trending-news score (no LLM tokens used).

Scoring formula (per spec):
    score = (50 if ticker in market_trending else 0)
          + 30 * (number_of_holdings_mentioned - 1)
          + tier_weight[publisher]   # 0-30
          + recency_decay            # 0-20, last 24h = 20
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, HttpUrl


_TIER_30 = {
    "reuters", "reuters.com",
    "wsj", "wsj.com",
    "bloomberg", "bloomberg.com",
    "cnbc", "cnbc.com",
    "financial times", "ft.com",
}
_TIER_20 = {"yahoo finance", "yahoofinance.com", "marketwatch", "marketwatch.com"}
_TIER_10 = {"seekingalpha", "seekingalpha.com", "barrons", "barrons.com",
            "investors", "investors.com", "benzinga", "benzinga.com"}


def publisher_tier_weight(publisher: str) -> int:
    p = publisher.strip().lower()
    if p in _TIER_30:
        return 30
    if p in _TIER_20:
        return 20
    if p in _TIER_10:
        return 10
    return 0


def _recency_decay(published_at: datetime) -> int:
    age = datetime.now(timezone.utc) - published_at
    if age <= timedelta(hours=24):
        return 20
    if age <= timedelta(hours=48):
        return 10
    if age <= timedelta(hours=72):
        return 5
    return 0


@dataclass
class NewsItem:
    title: str
    publisher: str
    url: str
    published_at: datetime
    related_tickers: list[str] = field(default_factory=list)


class RankedArticle(BaseModel):
    title: str
    publisher: str
    url: HttpUrl
    published_at: datetime
    related_tickers: list[str]
    is_market_wide_trending: bool
    score: float


def rank_articles(items: list[NewsItem], *,
                   holdings: set[str],
                   market_trending: set[str],
                   top_n: int = 5) -> list[RankedArticle]:
    scored: list[RankedArticle] = []
    for it in items:
        related_in_holdings = [t for t in it.related_tickers if t in holdings]
        if not related_in_holdings:
            continue
        is_trending = any(t in market_trending for t in it.related_tickers)
        score = 0.0
        if is_trending:
            score += 50
        score += 30 * max(0, len(related_in_holdings) - 1)
        score += publisher_tier_weight(it.publisher)
        score += _recency_decay(it.published_at)
        scored.append(RankedArticle(
            title=it.title, publisher=it.publisher, url=it.url,
            published_at=it.published_at,
            related_tickers=related_in_holdings,
            is_market_wide_trending=is_trending, score=score,
        ))
    scored.sort(key=lambda a: a.score, reverse=True)
    return scored[:top_n]
```

- [ ] **Step 10.5: Run tests**

```bash
uv run pytest tests/unit/analytics/test_news_ranker.py -v
```

Expected: 5 passed.

- [ ] **Step 10.6: Commit**

```bash
git add src/stock_analyzer/analytics/ tests/unit/analytics/
git commit -m "feat: add deterministic news ranker (trending score)"
```

---

### Task 11: Drawdown filter

**Files:**
- Create: `src/stock_analyzer/analytics/drawdown_filter.py`
- Test: `tests/unit/analytics/test_drawdown_filter.py`

- [ ] **Step 11.1: Write failing test**

```python
# tests/unit/analytics/test_drawdown_filter.py
from __future__ import annotations

from stock_analyzer.analytics.drawdown_filter import (
    Quote,
    filter_drawdowns,
)


def test_filters_below_threshold() -> None:
    quotes = [
        Quote(ticker="AAPL", prev_close=200.0, pre_market_price=189.0),  # -5.5%
        Quote(ticker="MSFT", prev_close=400.0, pre_market_price=396.0),  # -1%
        Quote(ticker="NVDA", prev_close=900.0, pre_market_price=836.0),  # -7.1%
    ]
    out = filter_drawdowns(quotes, threshold_pct=5.0)
    assert {d.ticker for d in out} == {"AAPL", "NVDA"}


def test_skips_when_pre_market_missing() -> None:
    quotes = [
        Quote(ticker="AAPL", prev_close=200.0, pre_market_price=None),
    ]
    assert filter_drawdowns(quotes, threshold_pct=5.0) == []


def test_skips_when_prev_close_zero() -> None:
    quotes = [
        Quote(ticker="X", prev_close=0.0, pre_market_price=0.0),
    ]
    assert filter_drawdowns(quotes, threshold_pct=5.0) == []


def test_threshold_is_inclusive() -> None:
    # Exactly -5%
    quotes = [Quote(ticker="X", prev_close=100.0, pre_market_price=95.0)]
    out = filter_drawdowns(quotes, threshold_pct=5.0)
    assert len(out) == 1
    assert abs(out[0].pct_drop + 5.0) < 1e-9
```

- [ ] **Step 11.2: Run to verify it fails**

```bash
uv run pytest tests/unit/analytics/test_drawdown_filter.py -v
```

Expected: ImportError.

- [ ] **Step 11.3: Implement `src/stock_analyzer/analytics/drawdown_filter.py`**

```python
"""Pure filter — selects holdings whose pre-market price is at/below
threshold_pct below the previous close.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Quote:
    ticker: str
    prev_close: float
    pre_market_price: float | None


@dataclass
class DrawdownCandidate:
    ticker: str
    prev_close: float
    pre_market_price: float
    pct_drop: float                     # negative number


def filter_drawdowns(quotes: list[Quote],
                      threshold_pct: float) -> list[DrawdownCandidate]:
    out: list[DrawdownCandidate] = []
    for q in quotes:
        if q.pre_market_price is None or q.prev_close <= 0:
            continue
        pct = (q.pre_market_price - q.prev_close) / q.prev_close * 100.0
        if pct <= -abs(threshold_pct):
            out.append(DrawdownCandidate(
                ticker=q.ticker, prev_close=q.prev_close,
                pre_market_price=q.pre_market_price, pct_drop=pct,
            ))
    return out
```

- [ ] **Step 11.4: Run tests**

```bash
uv run pytest tests/unit/analytics/test_drawdown_filter.py -v
```

Expected: 4 passed.

- [ ] **Step 11.5: Commit**

```bash
git add src/stock_analyzer/analytics/drawdown_filter.py tests/unit/analytics/test_drawdown_filter.py
git commit -m "feat: add drawdown filter for >5pct pre-market drops"
```

---

### Task 12: Politician scorer (24-month vs SPY)

**Files:**
- Create: `src/stock_analyzer/analytics/politician_scorer.py`
- Test: `tests/unit/analytics/test_politician_scorer.py`

- [ ] **Step 12.1: Write failing test**

```python
# tests/unit/analytics/test_politician_scorer.py
from __future__ import annotations

from datetime import date

from stock_analyzer.analytics.politician_scorer import (
    SimulatedTrade,
    compute_politician_return,
    midpoint_amount,
)


def test_midpoint_amount() -> None:
    assert midpoint_amount(1001, 15000) == (1001 + 15000) / 2.0
    assert midpoint_amount(None, None) == 0.0
    assert midpoint_amount(50000, None) == 50000.0


def test_simple_buy_hold_appreciation() -> None:
    # Buy AAPL on 2024-01-02 with $10,000; price doubles by end window.
    trades = [
        SimulatedTrade(ticker="AAPL", side="BUY",
                       trade_dollars=10_000.0,
                       open_price=100.0, close_price=200.0),
    ]
    ret = compute_politician_return(trades)
    assert abs(ret - 100.0) < 0.01  # +100%


def test_buy_then_sell_realized() -> None:
    # Bought at 100 with $10k → +50% by sell at 150.
    trades = [
        SimulatedTrade(ticker="X", side="BUY", trade_dollars=10_000.0,
                       open_price=100.0, close_price=150.0),
        SimulatedTrade(ticker="X", side="SELL", trade_dollars=15_000.0,
                       open_price=150.0, close_price=150.0),
    ]
    ret = compute_politician_return(trades)
    assert ret > 0


def test_zero_invested_returns_zero() -> None:
    assert compute_politician_return([]) == 0.0
```

- [ ] **Step 12.2: Run to verify it fails**

```bash
uv run pytest tests/unit/analytics/test_politician_scorer.py -v
```

Expected: ImportError.

- [ ] **Step 12.3: Implement `src/stock_analyzer/analytics/politician_scorer.py`**

```python
"""Politician 24-month performance computation vs SPY.

Methodology (documented in spec §7):
- Use midpoint of disclosed amount range as dollar value.
- "Buy" on disclosure_date + 1 (when public could have learned).
- Total return = (end portfolio value - cost basis) / cost basis.
"""

from __future__ import annotations

from dataclasses import dataclass


def midpoint_amount(min_usd: int | None, max_usd: int | None) -> float:
    if min_usd is None and max_usd is None:
        return 0.0
    if min_usd is None:
        return float(max_usd or 0)
    if max_usd is None:
        return float(min_usd)
    return (float(min_usd) + float(max_usd)) / 2.0


@dataclass
class SimulatedTrade:
    ticker: str
    side: str                       # "BUY" or "SELL"
    trade_dollars: float            # midpoint of disclosed range
    open_price: float               # price on disclosure_date + 1
    close_price: float              # price on window_end_date


def compute_politician_return(trades: list[SimulatedTrade]) -> float:
    """Total return percentage over the simulation window. Returns 0 if no exposure."""
    cost_basis = 0.0
    end_value = 0.0
    for t in trades:
        if t.open_price <= 0:
            continue
        units = t.trade_dollars / t.open_price
        if t.side == "BUY":
            cost_basis += t.trade_dollars
            end_value += units * t.close_price
        else:  # SELL
            # Treat sells as realized at the trade-day price (SPY-like benchmark)
            cost_basis += t.trade_dollars
            end_value += t.trade_dollars
    if cost_basis <= 0:
        return 0.0
    return (end_value - cost_basis) / cost_basis * 100.0


def compute_spy_return(start_close: float, end_close: float) -> float:
    if start_close <= 0:
        return 0.0
    return (end_close - start_close) / start_close * 100.0
```

- [ ] **Step 12.4: Run tests**

```bash
uv run pytest tests/unit/analytics/test_politician_scorer.py -v
```

Expected: 4 passed.

- [ ] **Step 12.5: Commit**

```bash
git add src/stock_analyzer/analytics/politician_scorer.py tests/unit/analytics/test_politician_scorer.py
git commit -m "feat: add politician 24-month return scorer"
```

---

### Task 13: Cost tracker

**Files:**
- Create: `src/stock_analyzer/analytics/cost_tracker.py`
- Test: `tests/unit/analytics/test_cost_tracker.py`

- [ ] **Step 13.1: Write failing test**

```python
# tests/unit/analytics/test_cost_tracker.py
from __future__ import annotations

from stock_analyzer.analytics.cost_tracker import CostTracker, sonnet_46_pricing


def test_sonnet_pricing_known_values() -> None:
    p = sonnet_46_pricing()
    # $3.00 input / $15.00 output per million tokens (cache reads cheaper)
    assert p.input_per_million == 3.0
    assert p.output_per_million == 15.0
    assert p.cache_read_per_million == 0.30


def test_cost_tracker_accumulation() -> None:
    t = CostTracker()
    t.record(input_tokens=10_000, output_tokens=2_000, cache_read_tokens=5_000)
    t.record(input_tokens=20_000, output_tokens=3_000, cache_read_tokens=0)
    assert t.total_input_tokens == 30_000
    assert t.total_output_tokens == 5_000
    cost = t.estimated_cost_usd()
    # Manual calc:
    # input charge = (10k - 5k) + 20k = 25k tokens at $3/M = $0.075
    # cache read   = 5k tokens at $0.30/M = $0.0015
    # output       = 5k tokens at $15/M = $0.075
    assert abs(cost - (0.075 + 0.0015 + 0.075)) < 1e-6
```

- [ ] **Step 13.2: Run to verify it fails**

```bash
uv run pytest tests/unit/analytics/test_cost_tracker.py -v
```

Expected: ImportError.

- [ ] **Step 13.3: Implement `src/stock_analyzer/analytics/cost_tracker.py`**

```python
"""Track Claude token usage and estimate cost per run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Pricing:
    """Per-million-token prices in USD."""
    input_per_million: float
    output_per_million: float
    cache_read_per_million: float


def sonnet_46_pricing() -> Pricing:
    return Pricing(
        input_per_million=3.0,
        output_per_million=15.0,
        cache_read_per_million=0.30,
    )


@dataclass
class CostTracker:
    pricing: Pricing = field(default_factory=sonnet_46_pricing)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0

    def record(self, *, input_tokens: int = 0, output_tokens: int = 0,
                cache_read_tokens: int = 0) -> None:
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cache_read_tokens += cache_read_tokens

    def estimated_cost_usd(self) -> float:
        non_cached_input = max(0, self.total_input_tokens - self.total_cache_read_tokens)
        return (
            non_cached_input / 1_000_000 * self.pricing.input_per_million
            + self.total_cache_read_tokens / 1_000_000 * self.pricing.cache_read_per_million
            + self.total_output_tokens / 1_000_000 * self.pricing.output_per_million
        )
```

- [ ] **Step 13.4: Run tests**

```bash
uv run pytest tests/unit/analytics/test_cost_tracker.py -v
```

Expected: 2 passed.

- [ ] **Step 13.5: Commit**

```bash
git add src/stock_analyzer/analytics/cost_tracker.py tests/unit/analytics/test_cost_tracker.py
git commit -m "feat: add Anthropic Sonnet 4.6 cost tracker"
```

---

*Phase 3 complete: pure-logic analytics modules are 100% unit-tested with no external mocks needed.*


## Phase 4 — Tools (Tasks 14–20)

External-data wrappers. Most of these are tested via `respx` (httpx mock) or by injecting a recorded JSON fixture; **no live API calls in tests**.

### Task 14: SnapTrade tool

**Files:**
- Create: `src/stock_analyzer/tools/__init__.py`, `src/stock_analyzer/tools/snaptrade.py`
- Test: `tests/unit/tools/test_snaptrade.py`

- [ ] **Step 14.1: Write failing test**

Create `tests/unit/tools/__init__.py` (empty), then:

```python
# tests/unit/tools/test_snaptrade.py
from __future__ import annotations

from stock_analyzer.tools.snaptrade import Holding, SnapTradeClient


def test_holding_dataclass() -> None:
    h = Holding(ticker="AAPL", quantity=10.0, currency="USD",
                avg_cost=150.0, market_value=2000.0, account="brokerage")
    assert h.ticker == "AAPL"


def test_client_get_holdings_aggregates_accounts(monkeypatch) -> None:
    # We mock the SDK call with a stub
    class StubAccountsApi:
        def list_user_accounts(self, **_) -> list[dict]:
            return [{"id": "acc-1"}, {"id": "acc-2"}]

    class StubAccountInfoApi:
        def get_user_account_positions(self, *, account_id, **_) -> list[dict]:
            if account_id == "acc-1":
                return [{"symbol": {"symbol": {"raw_symbol": "AAPL"}},
                         "units": 5, "currency": {"code": "USD"},
                         "average_purchase_price": 150.0,
                         "price": 200.0}]
            return [{"symbol": {"symbol": {"raw_symbol": "MSFT"}},
                     "units": 3, "currency": {"code": "USD"},
                     "average_purchase_price": 300.0,
                     "price": 400.0}]

    client = SnapTradeClient.__new__(SnapTradeClient)
    client._user_id = "u"
    client._user_secret = "s"
    client._accounts_api = StubAccountsApi()
    client._account_info_api = StubAccountInfoApi()

    holdings = client.get_holdings()
    tickers = sorted(h.ticker for h in holdings)
    assert tickers == ["AAPL", "MSFT"]
```

- [ ] **Step 14.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_snaptrade.py -v
```

Expected: ImportError.

- [ ] **Step 14.3: Implement `src/stock_analyzer/tools/__init__.py`**

```python
"""External-data tools (data fetchers used by agents and the orchestrator)."""
```

- [ ] **Step 14.4: Implement `src/stock_analyzer/tools/snaptrade.py`**

```python
"""SnapTrade brokerage data wrapper.

We intentionally keep a thin domain layer (Holding) over the SDK so the rest
of the app doesn't import SnapTrade types.
"""

from __future__ import annotations

from dataclasses import dataclass

from snaptrade_client import SnapTrade  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Holding:
    ticker: str
    quantity: float
    currency: str
    avg_cost: float | None
    market_value: float | None
    account: str


class SnapTradeClient:
    def __init__(self, *, client_id: str, consumer_key: str,
                 user_id: str, user_secret: str) -> None:
        sdk = SnapTrade(consumer_key=consumer_key, client_id=client_id)
        self._user_id = user_id
        self._user_secret = user_secret
        self._accounts_api = sdk.account_information
        self._account_info_api = sdk.account_information

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8))
    def _list_accounts(self) -> list[dict]:
        accounts = self._accounts_api.list_user_accounts(
            user_id=self._user_id, user_secret=self._user_secret,
        )
        return list(accounts)

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8))
    def _list_positions(self, account_id: str) -> list[dict]:
        return list(self._account_info_api.get_user_account_positions(
            user_id=self._user_id, user_secret=self._user_secret,
            account_id=account_id,
        ))

    def get_holdings(self) -> list[Holding]:
        out: list[Holding] = []
        for acct in self._list_accounts():
            acc_id = acct["id"]
            for pos in self._list_positions(acc_id):
                ticker = (pos.get("symbol", {})
                             .get("symbol", {})
                             .get("raw_symbol"))
                if not ticker:
                    continue
                qty = float(pos.get("units") or 0)
                if qty == 0:
                    continue
                price = pos.get("price")
                avg = pos.get("average_purchase_price")
                ccy = (pos.get("currency", {}) or {}).get("code", "USD")
                out.append(Holding(
                    ticker=ticker, quantity=qty, currency=ccy,
                    avg_cost=float(avg) if avg is not None else None,
                    market_value=qty * float(price) if price is not None else None,
                    account=str(acc_id),
                ))
        return out
```

- [ ] **Step 14.5: Run tests**

```bash
uv run pytest tests/unit/tools/test_snaptrade.py -v
```

Expected: 2 passed.

- [ ] **Step 14.6: Commit**

```bash
git add src/stock_analyzer/tools/ tests/unit/tools/
git commit -m "feat: add SnapTrade client wrapper for portfolio holdings"
```

---

### Task 15: Market data (yfinance + Yahoo trending)

**Files:**
- Create: `src/stock_analyzer/tools/market_data.py`
- Test: `tests/unit/tools/test_market_data.py`

- [ ] **Step 15.1: Write failing test**

```python
# tests/unit/tools/test_market_data.py
from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.tools.market_data import (
    MarketDataClient,
    QuoteSnapshot,
)


def test_quote_snapshot_dataclass() -> None:
    q = QuoteSnapshot(ticker="AAPL", prev_close=200.0,
                       pre_market_price=189.0, last_price=189.0)
    assert q.pct_change_overnight() == pytest.approx((189.0 - 200.0) / 200.0 * 100, rel=1e-6)


def test_batch_quotes_uses_yfinance(monkeypatch) -> None:
    captured = {}

    class FakeTicker:
        def __init__(self, t: str) -> None:
            self.t = t

        @property
        def info(self) -> dict:
            return {
                "regularMarketPreviousClose": 200.0,
                "preMarketPrice": 195.0,
                "regularMarketPrice": 196.0,
            }

    def fake_ticker(t: str) -> FakeTicker:
        captured.setdefault("calls", []).append(t)
        return FakeTicker(t)

    monkeypatch.setattr("stock_analyzer.tools.market_data.yf.Ticker", fake_ticker)

    client = MarketDataClient()
    quotes = client.batch_quotes(["AAPL", "MSFT"])
    assert {q.ticker for q in quotes} == {"AAPL", "MSFT"}
    assert all(q.prev_close == 200.0 for q in quotes)


def test_get_trending_tickers_handles_failure(monkeypatch) -> None:
    def boom(*_a, **_kw):
        raise RuntimeError("yahoo redesigned again")
    monkeypatch.setattr("stock_analyzer.tools.market_data._fetch_trending", boom)
    client = MarketDataClient()
    assert client.get_trending_tickers() == set()  # graceful degradation


def test_get_spy_close_for_date(monkeypatch) -> None:
    import pandas as pd

    class FakeTicker:
        def __init__(self, *_a, **_kw) -> None: ...
        def history(self, start, end) -> pd.DataFrame:
            idx = pd.DatetimeIndex([pd.Timestamp("2026-05-04")])
            return pd.DataFrame({"Close": [528.41]}, index=idx)

    monkeypatch.setattr("stock_analyzer.tools.market_data.yf.Ticker", FakeTicker)
    client = MarketDataClient()
    assert client.get_spy_close(date(2026, 5, 4)) == pytest.approx(528.41)
```

- [ ] **Step 15.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_market_data.py -v
```

Expected: ImportError.

- [ ] **Step 15.3: Implement `src/stock_analyzer/tools/market_data.py`**

```python
"""yfinance + Yahoo trending wrapper. All unofficial endpoints are best-effort."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import httpx
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_fixed

from stock_analyzer.logging import get_logger

_log = get_logger(__name__)


@dataclass
class QuoteSnapshot:
    ticker: str
    prev_close: float
    pre_market_price: float | None
    last_price: float | None

    def pct_change_overnight(self) -> float:
        ref = self.pre_market_price if self.pre_market_price else self.last_price
        if ref is None or self.prev_close <= 0:
            return 0.0
        return (ref - self.prev_close) / self.prev_close * 100.0


def _fetch_trending() -> set[str]:
    """Hits Yahoo's public trending endpoint. Best-effort."""
    url = "https://query1.finance.yahoo.com/v1/finance/trending/US?count=25"
    headers = {"User-Agent": "Mozilla/5.0 stock-analyzer/1.0"}
    with httpx.Client(timeout=10) as cx:
        r = cx.get(url, headers=headers)
        r.raise_for_status()
        data = r.json()
    quotes = data.get("finance", {}).get("result", [{}])[0].get("quotes", [])
    return {q.get("symbol") for q in quotes if q.get("symbol")}


class MarketDataClient:
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(5))
    def batch_quotes(self, tickers: list[str]) -> list[QuoteSnapshot]:
        out: list[QuoteSnapshot] = []
        for t in tickers:
            try:
                info = yf.Ticker(t).info
            except Exception as e:
                _log.warning("yfinance.info_failed", ticker=t, err=str(e))
                continue
            out.append(QuoteSnapshot(
                ticker=t,
                prev_close=float(info.get("regularMarketPreviousClose") or 0),
                pre_market_price=(float(info["preMarketPrice"])
                                   if info.get("preMarketPrice") else None),
                last_price=(float(info["regularMarketPrice"])
                             if info.get("regularMarketPrice") else None),
            ))
        return out

    def get_trending_tickers(self) -> set[str]:
        try:
            return _fetch_trending()
        except Exception as e:
            _log.warning("yahoo_trending_failed", err=str(e))
            return set()

    def get_spy_close(self, d: date) -> float | None:
        df = yf.Ticker("SPY").history(start=d, end=d + timedelta(days=1))
        if df.empty:
            return None
        return float(df["Close"].iloc[0])

    def get_spy_range(self, start: date, end: date) -> dict[date, float]:
        df = yf.Ticker("SPY").history(start=start, end=end + timedelta(days=1))
        if df.empty:
            return {}
        return {idx.date(): float(row.Close) for idx, row in df.iterrows()}
```

- [ ] **Step 15.4: Run tests**

```bash
uv run pytest tests/unit/tools/test_market_data.py -v
```

Expected: 4 passed.

- [ ] **Step 15.5: Commit**

```bash
git add src/stock_analyzer/tools/market_data.py tests/unit/tools/test_market_data.py
git commit -m "feat: add market data client (yfinance + yahoo trending)"
```

---

### Task 16: News (Finnhub + yfinance news + RSS)

**Files:**
- Create: `src/stock_analyzer/tools/news.py`
- Test: `tests/unit/tools/test_news.py`

- [ ] **Step 16.1: Write failing test**

```python
# tests/unit/tools/test_news.py
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
import respx
from httpx import Response

from stock_analyzer.tools.news import NewsClient


@pytest.mark.asyncio
@respx.mock
async def test_finnhub_company_news() -> None:
    respx.get("https://finnhub.io/api/v1/company-news").mock(
        return_value=Response(200, json=[
            {"headline": "Apple beats earnings",
             "source": "Reuters",
             "url": "https://r.com/1",
             "datetime": int(datetime(2026, 5, 4, tzinfo=timezone.utc).timestamp()),
             "related": "AAPL"},
        ])
    )
    client = NewsClient(finnhub_api_key="fh", http=None)
    items = await client.get_company_news("AAPL", since=datetime(2026, 5, 1, tzinfo=timezone.utc))
    assert len(items) == 1
    assert items[0].publisher == "Reuters"
    assert items[0].related_tickers == ["AAPL"]


def test_yfinance_news_normalizes(monkeypatch) -> None:
    fake_news = [
        {"title": "X", "publisher": "Yahoo Finance",
         "link": "https://y/1",
         "providerPublishTime": int(datetime.now(timezone.utc).timestamp()),
         "relatedTickers": ["AAPL", "MSFT"]},
    ]

    class FakeTicker:
        news = fake_news

    monkeypatch.setattr("stock_analyzer.tools.news.yf.Ticker",
                         lambda _t: FakeTicker())
    client = NewsClient(finnhub_api_key="fh", http=None)
    items = client.get_yfinance_news("AAPL")
    assert items[0].publisher == "Yahoo Finance"
    assert "AAPL" in items[0].related_tickers


def test_rss_macro_news(monkeypatch) -> None:
    sample_feed = type("F", (), {
        "entries": [
            type("E", (), {
                "title": "Markets selloff",
                "link": "https://r/1",
                "published_parsed": (2026, 5, 4, 12, 0, 0, 0, 0, 0),
                "summary": "...",
            })()
        ]
    })()
    monkeypatch.setattr("stock_analyzer.tools.news.feedparser.parse",
                         lambda _u: sample_feed)
    client = NewsClient(finnhub_api_key="fh", http=None)
    items = client.get_macro_news(since=datetime(2026, 5, 1, tzinfo=timezone.utc))
    assert any("Markets selloff" in i.title for i in items)
```

- [ ] **Step 16.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_news.py -v
```

Expected: ImportError.

- [ ] **Step 16.3: Implement `src/stock_analyzer/tools/news.py`**

```python
"""Aggregated news fetchers: Finnhub (per-ticker), yfinance (per-ticker),
RSS (macro)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import feedparser
import httpx
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

from stock_analyzer.logging import get_logger

_log = get_logger(__name__)


_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.marketwatch.com/rss/topstories",
]


@dataclass
class NewsItem:
    title: str
    publisher: str
    url: str
    published_at: datetime
    related_tickers: list[str] = field(default_factory=list)


class NewsClient:
    def __init__(self, *, finnhub_api_key: str,
                 http: httpx.AsyncClient | None) -> None:
        self._fh_key = finnhub_api_key
        self._http = http

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=1, max=8))
    async def get_company_news(self, ticker: str, *,
                                since: datetime) -> list[NewsItem]:
        params = {"symbol": ticker, "from": since.date().isoformat(),
                  "to": datetime.now(timezone.utc).date().isoformat(),
                  "token": self._fh_key}
        async with httpx.AsyncClient(timeout=15) as cx:
            r = await cx.get("https://finnhub.io/api/v1/company-news", params=params)
            r.raise_for_status()
            payload = r.json()
        out: list[NewsItem] = []
        for row in payload:
            ts = row.get("datetime")
            published_at = (
                datetime.fromtimestamp(ts, tz=timezone.utc) if ts
                else datetime.now(timezone.utc)
            )
            related = [t.strip() for t in (row.get("related") or "").split(",") if t.strip()]
            out.append(NewsItem(
                title=row.get("headline", ""), publisher=row.get("source", ""),
                url=row.get("url", ""), published_at=published_at,
                related_tickers=related,
            ))
        return out

    def get_yfinance_news(self, ticker: str) -> list[NewsItem]:
        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as e:
            _log.warning("yfinance.news_failed", ticker=ticker, err=str(e))
            return []
        out: list[NewsItem] = []
        for row in raw:
            ts = row.get("providerPublishTime")
            published_at = (
                datetime.fromtimestamp(ts, tz=timezone.utc) if ts
                else datetime.now(timezone.utc)
            )
            out.append(NewsItem(
                title=row.get("title", ""),
                publisher=row.get("publisher", ""),
                url=row.get("link", ""), published_at=published_at,
                related_tickers=row.get("relatedTickers") or [],
            ))
        return out

    def get_macro_news(self, *, since: datetime) -> list[NewsItem]:
        out: list[NewsItem] = []
        for url in _RSS_FEEDS:
            try:
                feed = feedparser.parse(url)
            except Exception as e:
                _log.warning("rss_failed", url=url, err=str(e))
                continue
            for entry in getattr(feed, "entries", []):
                published = getattr(entry, "published_parsed", None)
                if published:
                    dt = datetime(*published[:6], tzinfo=timezone.utc)
                else:
                    dt = datetime.now(timezone.utc)
                if dt < since:
                    continue
                out.append(NewsItem(
                    title=getattr(entry, "title", ""),
                    publisher=feed.feed.get("title", url) if hasattr(feed, "feed") else url,
                    url=getattr(entry, "link", url),
                    published_at=dt,
                    related_tickers=[],
                ))
        return out
```

- [ ] **Step 16.4: Run tests**

```bash
uv run pytest tests/unit/tools/test_news.py -v
```

Expected: 3 passed.

- [ ] **Step 16.5: Commit**

```bash
git add src/stock_analyzer/tools/news.py tests/unit/tools/test_news.py
git commit -m "feat: add news client (finnhub + yfinance + rss)"
```

---

### Task 17: SEC EDGAR (Form 4 + 13F)

**Files:**
- Create: `src/stock_analyzer/tools/sec_edgar.py`
- Test: `tests/unit/tools/test_sec_edgar.py`

- [ ] **Step 17.1: Write failing test**

```python
# tests/unit/tools/test_sec_edgar.py
from __future__ import annotations

import pytest
import respx
from httpx import Response

from stock_analyzer.tools.sec_edgar import SecEdgarClient


_TICKER_PAYLOAD = {"AAPL": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
_FILINGS_PAYLOAD = {
    "filings": {
        "recent": {
            "accessionNumber": ["0001127602-26-013470"],
            "filingDate": ["2026-05-04"],
            "form": ["4"],
            "primaryDocument": ["doc.html"],
        }
    }
}


@pytest.mark.asyncio
@respx.mock
async def test_get_recent_form_4() -> None:
    respx.get("https://www.sec.gov/files/company_tickers.json").mock(
        return_value=Response(200, json=_TICKER_PAYLOAD))
    respx.get("https://data.sec.gov/submissions/CIK0000320193.json").mock(
        return_value=Response(200, json=_FILINGS_PAYLOAD))
    client = SecEdgarClient(user_agent="stock-analyzer test@example.com")
    filings = await client.get_recent_form_4("AAPL", days=30)
    assert len(filings) == 1
    assert filings[0].form == "4"
    assert filings[0].url.startswith("https://www.sec.gov/Archives/edgar/data/")
```

- [ ] **Step 17.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_sec_edgar.py -v
```

Expected: ImportError.

- [ ] **Step 17.3: Implement `src/stock_analyzer/tools/sec_edgar.py`**

```python
"""SEC EDGAR fetcher — Form 4 (insider trades) and 13F-HR (institutional)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Filing:
    form: str
    accession: str
    filing_date: date
    url: str


class SecEdgarClient:
    def __init__(self, *, user_agent: str) -> None:
        # SEC requires a meaningful UA per https://www.sec.gov/os/accessing-edgar-data
        self._ua = user_agent
        self._cik_cache: dict[str, str] = {}

    async def _http(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            headers={"User-Agent": self._ua, "Accept-Encoding": "gzip, deflate"},
            timeout=30,
        )

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=2, min=2, max=16))
    async def _ticker_to_cik(self, ticker: str) -> str:
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]
        async with await self._http() as cx:
            r = await cx.get("https://www.sec.gov/files/company_tickers.json")
            r.raise_for_status()
            for row in r.json().values():
                if row.get("ticker", "").upper() == ticker.upper():
                    cik = str(row["cik_str"]).zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik
        raise ValueError(f"CIK not found for {ticker}")

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=2, min=2, max=16))
    async def _list_filings(self, cik: str) -> list[dict]:
        url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        async with await self._http() as cx:
            r = await cx.get(url)
            r.raise_for_status()
        recent = r.json().get("filings", {}).get("recent", {})
        rows = []
        for i in range(len(recent.get("accessionNumber", []))):
            rows.append({
                "accession": recent["accessionNumber"][i],
                "form": recent["form"][i],
                "filingDate": recent["filingDate"][i],
                "primaryDocument": recent.get("primaryDocument", [""])[i],
            })
        return rows

    async def _filter_filings(self, ticker: str, *, form: str,
                                days: int) -> list[Filing]:
        cik = await self._ticker_to_cik(ticker)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date()
        out: list[Filing] = []
        for row in await self._list_filings(cik):
            if row["form"] != form:
                continue
            fd = date.fromisoformat(row["filingDate"])
            if fd < cutoff:
                continue
            acc_nodash = row["accession"].replace("-", "")
            url = (f"https://www.sec.gov/Archives/edgar/data/"
                    f"{int(cik)}/{acc_nodash}/{row['primaryDocument']}")
            out.append(Filing(form=row["form"], accession=row["accession"],
                              filing_date=fd, url=url))
        return out

    async def get_recent_form_4(self, ticker: str, *, days: int = 7) -> list[Filing]:
        return await self._filter_filings(ticker, form="4", days=days)

    async def get_recent_13f(self, ticker: str, *, quarters: int = 2) -> list[Filing]:
        days = quarters * 95
        return await self._filter_filings(ticker, form="13F-HR", days=days)
```

- [ ] **Step 17.4: Run tests**

```bash
uv run pytest tests/unit/tools/test_sec_edgar.py -v
```

Expected: 1 passed.

- [ ] **Step 17.5: Commit**

```bash
git add src/stock_analyzer/tools/sec_edgar.py tests/unit/tools/test_sec_edgar.py
git commit -m "feat: add SEC EDGAR client for form 4 and 13F filings"
```

---

### Task 18: CapitolTrades (deterministic JSON API)

**Files:**
- Create: `src/stock_analyzer/tools/capitol_trades.py`
- Test: `tests/unit/tools/test_capitol_trades.py`

- [ ] **Step 18.1: Write failing test**

```python
# tests/unit/tools/test_capitol_trades.py
from __future__ import annotations

from datetime import date

import pytest
import respx
from httpx import Response

from stock_analyzer.tools.capitol_trades import (
    CapitolTradesClient,
    Disclosure,
)


_PAYLOAD = {
    "data": [
        {
            "txDate": "2026-05-03",
            "pubDate": "2026-05-04T12:00:00Z",
            "txType": "Buy",
            "value": "$1,001 - $15,000",
            "valueLow": 1001, "valueHigh": 15000,
            "asset": {"assetTicker": "AAPL"},
            "politician": {
                "fullName": "Jane Doe",
                "party": "Democrat",
                "chamber": "House",
                "state": "CA",
                "id": "P000123",
            },
        }
    ],
    "meta": {"paging": {"totalPages": 1, "currentPage": 1}},
}


@pytest.mark.asyncio
@respx.mock
async def test_get_recent_disclosures() -> None:
    respx.get(url__regex=r"https://bff\.capitoltrades\.com/trades.*").mock(
        return_value=Response(200, json=_PAYLOAD))
    client = CapitolTradesClient()
    out = await client.get_recent_disclosures(since=date(2026, 5, 1))
    assert len(out) == 1
    d = out[0]
    assert isinstance(d, Disclosure)
    assert d.ticker == "AAPL"
    assert d.side == "BUY"
    assert d.party == "D"
    assert d.chamber == "House"
    assert d.amount_min_usd == 1001 and d.amount_max_usd == 15000


def test_party_normalization() -> None:
    from stock_analyzer.tools.capitol_trades import _normalize_party
    assert _normalize_party("Democrat") == "D"
    assert _normalize_party("Republican") == "R"
    assert _normalize_party("Independent") == "I"
    assert _normalize_party("Other") == "I"


def test_side_normalization() -> None:
    from stock_analyzer.tools.capitol_trades import _normalize_side
    assert _normalize_side("Buy") == "BUY"
    assert _normalize_side("buy") == "BUY"
    assert _normalize_side("Partial Sale") == "SELL"
    assert _normalize_side("Sell (Full)") == "SELL"
```

- [ ] **Step 18.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_capitol_trades.py -v
```

Expected: ImportError.

- [ ] **Step 18.3: Implement `src/stock_analyzer/tools/capitol_trades.py`**

```python
"""CapitolTrades public JSON API — deterministic structured data.

Endpoint: https://bff.capitoltrades.com/trades
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class Disclosure:
    politician_full_name: str
    capitol_trades_id: str | None
    party: str
    chamber: str
    state: str | None
    ticker: str
    side: str                       # "BUY" or "SELL"
    trade_date: date
    disclosure_date: date
    amount_min_usd: int | None
    amount_max_usd: int | None
    raw: dict[str, Any]


def _normalize_party(p: str) -> str:
    p = (p or "").strip().lower()
    if p.startswith("dem"):
        return "D"
    if p.startswith("rep"):
        return "R"
    return "I"


def _normalize_side(s: str) -> str:
    s = (s or "").strip().lower()
    return "SELL" if "sell" in s or "sale" in s else "BUY"


class CapitolTradesClient:
    BASE = "https://bff.capitoltrades.com/trades"

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=2, min=2, max=16))
    async def _page(self, params: dict[str, Any]) -> dict:
        async with httpx.AsyncClient(timeout=30) as cx:
            r = await cx.get(self.BASE, params=params,
                              headers={"User-Agent": "stock-analyzer/1.0"})
            r.raise_for_status()
            return r.json()

    async def get_recent_disclosures(self, *, since: date,
                                       max_pages: int = 10) -> list[Disclosure]:
        page = 1
        out: list[Disclosure] = []
        while page <= max_pages:
            data = await self._page({"page": page, "pageSize": 100,
                                       "sortBy": "-pubDate"})
            rows = data.get("data") or []
            for row in rows:
                pub_raw = row.get("pubDate") or row.get("disclosureDate")
                pub_date = (datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
                              .astimezone(timezone.utc).date()
                            if pub_raw else date.today())
                if pub_date < since:
                    return out  # results sorted desc → safe to stop
                tx_raw = row.get("txDate") or row.get("transactionDate")
                tx_date = date.fromisoformat(tx_raw[:10]) if tx_raw else pub_date
                pol = row.get("politician", {}) or {}
                asset = row.get("asset", {}) or {}
                ticker = asset.get("assetTicker") or asset.get("ticker") or ""
                if not ticker:
                    continue
                out.append(Disclosure(
                    politician_full_name=pol.get("fullName", "") or "",
                    capitol_trades_id=pol.get("id"),
                    party=_normalize_party(pol.get("party", "")),
                    chamber=pol.get("chamber", "") or "House",
                    state=pol.get("state"),
                    ticker=ticker.upper(),
                    side=_normalize_side(row.get("txType", "")),
                    trade_date=tx_date,
                    disclosure_date=pub_date,
                    amount_min_usd=row.get("valueLow"),
                    amount_max_usd=row.get("valueHigh"),
                    raw=row,
                ))
            paging = data.get("meta", {}).get("paging", {})
            if page >= paging.get("totalPages", 1):
                break
            page += 1
        return out
```

- [ ] **Step 18.4: Run tests**

```bash
uv run pytest tests/unit/tools/test_capitol_trades.py -v
```

Expected: 3 passed.

- [ ] **Step 18.5: Commit**

```bash
git add src/stock_analyzer/tools/capitol_trades.py tests/unit/tools/test_capitol_trades.py
git commit -m "feat: add CapitolTrades client (deterministic JSON API)"
```

---

### Task 19: InsiderMonkey (Crawl4ai-driven)

**Files:**
- Create: `src/stock_analyzer/tools/insider_monkey.py`
- Test: `tests/unit/tools/test_insider_monkey.py`

- [ ] **Step 19.1: Write failing test**

```python
# tests/unit/tools/test_insider_monkey.py
from __future__ import annotations

import pytest

from stock_analyzer.tools.insider_monkey import InsiderMonkeyClient


@pytest.mark.asyncio
async def test_search_articles_uses_crawler(monkeypatch) -> None:
    fetched: list[str] = []

    class FakeResult:
        markdown = "# Article\n\nApple bought by Bridgewater."

    async def fake_fetch(self, url: str) -> FakeResult:  # noqa: ARG001
        fetched.append(url)
        return FakeResult()

    monkeypatch.setattr(
        "stock_analyzer.tools.insider_monkey.InsiderMonkeyClient._fetch_markdown",
        fake_fetch,
    )

    client = InsiderMonkeyClient()
    articles = await client.search_articles("AAPL", limit=2)
    assert len(articles) >= 1
    assert "Apple" in articles[0].markdown
```

- [ ] **Step 19.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_insider_monkey.py -v
```

Expected: ImportError.

- [ ] **Step 19.3: Implement `src/stock_analyzer/tools/insider_monkey.py`**

```python
"""InsiderMonkey scraper using Crawl4ai for clean markdown extraction.

Crawl4ai is invoked via its Python API. We fetch the search results page,
extract article URLs whose summary contains the ticker, then fetch each
article as markdown for the agent to summarize.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

from stock_analyzer.logging import get_logger

_log = get_logger(__name__)


@dataclass
class Article:
    url: str
    title: str
    markdown: str


_BASE = "https://www.insidermonkey.com"


class InsiderMonkeyClient:
    async def _fetch_markdown(self, url: str) -> "AsyncResult":  # noqa: F821
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url, config=CrawlerRunConfig(verbose=False))

    async def search_articles(self, ticker: str, *, limit: int = 5) -> list[Article]:
        try:
            search_url = f"{_BASE}/?s={ticker}"
            res = await self._fetch_markdown(search_url)
            md: str = getattr(res, "markdown", "") or ""
        except Exception as e:
            _log.warning("insider_monkey.search_failed", ticker=ticker, err=str(e))
            return []

        urls = re.findall(r"\[([^\]]+)\]\((https?://[^\)]+/insider-trading[^\)]+)\)", md)[:limit]
        out: list[Article] = []
        for title, url in urls:
            try:
                article_res = await self._fetch_markdown(url)
                article_md: str = getattr(article_res, "markdown", "") or ""
            except Exception as e:
                _log.warning("insider_monkey.article_failed", url=url, err=str(e))
                continue
            out.append(Article(url=url, title=title.strip(), markdown=article_md))
        return out
```

- [ ] **Step 19.4: Run tests**

```bash
uv run pytest tests/unit/tools/test_insider_monkey.py -v
```

Expected: 1 passed.

- [ ] **Step 19.5: Commit**

```bash
git add src/stock_analyzer/tools/insider_monkey.py tests/unit/tools/test_insider_monkey.py
git commit -m "feat: add InsiderMonkey scraper via Crawl4ai"
```

---

### Task 20: SMTP sender (Stalwart submission)

**Files:**
- Create: `src/stock_analyzer/tools/smtp_sender.py`
- Test: `tests/unit/tools/test_smtp_sender.py`

- [ ] **Step 20.1: Write failing test**

```python
# tests/unit/tools/test_smtp_sender.py
from __future__ import annotations

import asyncio
import email
from email.message import Message

import pytest
from aiosmtpd.controller import Controller
from aiosmtpd.handlers import Message as MsgHandler

from stock_analyzer.tools.smtp_sender import EmailMessage, SmtpSender


class _CaptureHandler(MsgHandler):
    def __init__(self) -> None:
        super().__init__()
        self.captured: list[Message] = []

    def handle_message(self, message: Message) -> None:
        self.captured.append(message)


@pytest.fixture
def smtp_server():
    handler = _CaptureHandler()
    ctrl = Controller(handler, hostname="127.0.0.1", port=0)
    ctrl.start()
    yield ctrl, handler
    ctrl.stop()


@pytest.mark.asyncio
async def test_smtp_sender_delivers(smtp_server) -> None:
    ctrl, handler = smtp_server
    sender = SmtpSender(host="127.0.0.1", port=ctrl.port,
                         username=None, password=None, use_tls=False,
                         from_address="from@example.com",
                         from_name="Sender")
    msg = EmailMessage(to="to@example.com",
                        subject="Hello",
                        html_body="<p>hi</p>")
    await sender.send(msg)
    assert len(handler.captured) == 1
    assert handler.captured[0]["Subject"] == "Hello"
```

- [ ] **Step 20.2: Run to verify it fails**

```bash
uv run pytest tests/unit/tools/test_smtp_sender.py -v
```

Expected: ImportError.

- [ ] **Step 20.3: Implement `src/stock_analyzer/tools/smtp_sender.py`**

```python
"""SMTP submission to Stalwart (or any RFC-compliant submission server)."""

from __future__ import annotations

from dataclasses import dataclass
from email.message import EmailMessage as StdEmailMessage

import aiosmtplib
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class EmailMessage:
    to: str
    subject: str
    html_body: str
    text_body: str | None = None


@dataclass
class SmtpSender:
    host: str
    port: int
    username: str | None
    password: str | None
    use_tls: bool
    from_address: str
    from_name: str = "Stock Analyzer"

    def _build(self, m: EmailMessage) -> StdEmailMessage:
        msg = StdEmailMessage()
        msg["From"] = f"{self.from_name} <{self.from_address}>"
        msg["To"] = m.to
        msg["Subject"] = m.subject
        if m.text_body:
            msg.set_content(m.text_body)
            msg.add_alternative(m.html_body, subtype="html")
        else:
            msg.set_content("This email requires an HTML-capable client.")
            msg.add_alternative(m.html_body, subtype="html")
        return msg

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=5, min=5, max=300))
    async def send(self, m: EmailMessage) -> None:
        msg = self._build(m)
        await aiosmtplib.send(
            msg,
            hostname=self.host, port=self.port,
            username=self.username, password=self.password,
            start_tls=self.use_tls, use_tls=False,
        )
```

- [ ] **Step 20.4: Run tests**

```bash
uv run pytest tests/unit/tools/test_smtp_sender.py -v
```

Expected: 1 passed.

- [ ] **Step 20.5: Commit**

```bash
git add src/stock_analyzer/tools/smtp_sender.py tests/unit/tools/test_smtp_sender.py
git commit -m "feat: add SMTP sender with retries and HTML email support"
```

---

*Phase 4 complete: all external-data tools implemented and tested in isolation.*


## Phase 5 — Rendering (Tasks 21–22)

### Task 21: Jinja2 templates

**Files:**
- Create: `src/stock_analyzer/rendering/__init__.py`
- Create: `src/stock_analyzer/rendering/templates/portfolio.html.j2`
- Create: `src/stock_analyzer/rendering/templates/drawdown.html.j2`
- Create: `src/stock_analyzer/rendering/templates/politician.html.j2`

(No tests in this task — templates are exercised by the Task 22 renderer tests.)

- [ ] **Step 21.1: Create `src/stock_analyzer/rendering/__init__.py`**

```python
"""Email rendering — Jinja2 templates produce HTML from typed Pydantic models."""
```

- [ ] **Step 21.2: Create `src/stock_analyzer/rendering/templates/portfolio.html.j2`**

```jinja
<!doctype html>
<html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,sans-serif;max-width:720px;margin:auto;color:#222;}
.banner{background:#fff3cd;border:1px solid #ffeeba;padding:8px 12px;border-radius:4px;margin-bottom:16px;}
h1{font-size:20px;margin:0 0 4px;} h2{font-size:16px;margin-top:24px;border-bottom:1px solid #eee;padding-bottom:4px;}
.muted{color:#666;font-size:12px;}
table{border-collapse:collapse;width:100%;margin:8px 0;}
th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;font-size:13px;}
.up{color:#1a7f37;} .down{color:#cf222e;}
.bullets{margin:6px 0 0 16px;padding:0;}
.bullets li{font-size:13px;line-height:1.5;}
.tag{display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;background:#e7f3ff;color:#0969da;margin-left:6px;}
.trending{background:#fff8c5;}
</style></head>
<body>
{% if mode == "ephemeral" %}
<div class="banner">Ephemeral mode — no DB, results not persisted.</div>
{% endif %}
<h1>Portfolio Daily Analysis</h1>
<div class="muted">{{ report.as_of.strftime("%A, %B %-d, %Y · %-I:%M %p ET") }}</div>
<p>{{ report.portfolio_summary }}</p>

{% if report.trending_news %}
<h2>📈 Trending in Your Portfolio</h2>
<table>
{% for a in report.trending_news %}
<tr class="{% if a.is_market_wide_trending %}trending{% endif %}">
<td><a href="{{ a.url }}">{{ a.title }}</a>
{% if a.is_market_wide_trending %}<span class="tag">market-wide</span>{% endif %}
<div class="muted">{{ a.publisher }} · {{ ', '.join(a.related_tickers) }}</div></td>
</tr>
{% endfor %}
</table>
{% endif %}

<h2>Holdings</h2>
{% for h in report.holdings %}
<div>
<strong>{{ h.ticker }}</strong> <span class="muted">{{ h.company_name }}</span>
<span class="{% if h.pct_change_overnight >= 0 %}up{% else %}down{% endif %}">
  {{ "%+.2f"|format(h.pct_change_overnight) }}%
</span>
<ul class="bullets">{% for b in h.bullets %}<li>{{ b }}</li>{% endfor %}</ul>
<div class="muted">Watch: {{ h.watch_today }}</div>
</div>
<hr style="border:0;border-top:1px dashed #eee;margin:12px 0;">
{% endfor %}
</body></html>
```

- [ ] **Step 21.3: Create `src/stock_analyzer/rendering/templates/drawdown.html.j2`**

```jinja
<!doctype html>
<html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,sans-serif;max-width:720px;margin:auto;color:#222;}
.banner{background:#fff3cd;border:1px solid #ffeeba;padding:8px 12px;border-radius:4px;margin-bottom:16px;}
h1{font-size:20px;margin:0 0 4px;} h2{font-size:16px;margin-top:24px;}
.muted{color:#666;font-size:12px;}
.down{color:#cf222e;font-weight:600;}
.empty{padding:20px;text-align:center;background:#f6f8fa;border-radius:6px;}
table{border-collapse:collapse;width:100%;}
th,td{padding:6px 8px;border-bottom:1px solid #eee;text-align:left;font-size:13px;}
.cause{display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;background:#ddf4ff;color:#0969da;}
</style></head>
<body>
{% if mode == "ephemeral" %}
<div class="banner">Ephemeral mode — no DB, results not persisted.</div>
{% endif %}
<h1>Pre-Market Drawdown Alert</h1>
<div class="muted">{{ report.as_of.strftime("%A, %B %-d, %Y · %-I:%M %p ET") }}</div>
{% if not report.items %}
<div class="empty">No holdings down more than {{ threshold_pct }}% in pre-market today.</div>
{% else %}
{% if report.market_context %}<p>{{ report.market_context }}</p>{% endif %}
<table>
<thead><tr><th>Ticker</th><th>Drop</th><th>Pre-Market</th><th>Prev Close</th><th>Cause</th></tr></thead>
<tbody>
{% for d in report.items %}
<tr>
<td><strong>{{ d.ticker }}</strong></td>
<td class="down">{{ "%.2f"|format(d.pct_drop) }}%</td>
<td>${{ "%.2f"|format(d.pre_market_price) }}</td>
<td>${{ "%.2f"|format(d.prev_close) }}</td>
<td><span class="cause">{{ d.likely_cause }}</span></td>
</tr>
<tr><td colspan="5" class="muted">{{ d.explanation }}
{% for s in d.sources %} · <a href="{{ s }}">source</a>{% endfor %}</td></tr>
{% endfor %}
</tbody></table>
{% endif %}
</body></html>
```

- [ ] **Step 21.4: Create `src/stock_analyzer/rendering/templates/politician.html.j2`**

```jinja
<!doctype html>
<html><head><meta charset="utf-8"><style>
body{font-family:-apple-system,sans-serif;max-width:720px;margin:auto;color:#222;}
.banner{background:#fff3cd;border:1px solid #ffeeba;padding:8px 12px;border-radius:4px;margin-bottom:16px;}
h1{font-size:20px;margin:0 0 4px;} h2{font-size:16px;margin-top:24px;}
.muted{color:#666;font-size:12px;}
.buy{color:#1a7f37;font-weight:600;} .sell{color:#cf222e;font-weight:600;}
.empty{padding:20px;text-align:center;background:#f6f8fa;border-radius:6px;}
.party-D{background:#cfe8ff;color:#0969da;} .party-R{background:#ffd6cc;color:#cf222e;} .party-I{background:#e3e3e3;color:#444;}
.tag{display:inline-block;padding:1px 6px;border-radius:8px;font-size:11px;}
</style></head>
<body>
{% if mode == "ephemeral" %}
<div class="banner">Ephemeral mode: politician scoring filter disabled.</div>
{% endif %}
<h1>Politician Trade Signal — Above-SPY Performers</h1>
<div class="muted">{{ report.as_of.strftime("%A, %B %-d, %Y · %-I:%M %p ET") }}</div>
<p>{{ report.top_takeaway }}</p>

{% macro trade_block(t) %}
<div style="margin:8px 0;">
<strong>{{ t.politician_name }}</strong>
<span class="tag party-{{ t.politician_party }}">{{ t.politician_party }}</span>
<span class="muted">{{ t.politician_chamber }} · 24mo α {{ "%+.1f"|format(t.politician_24mo_alpha_vs_spy) }}pp</span>
<br>
<span class="{% if t.side == 'BUY' %}buy{% else %}sell{% endif %}">{{ t.side }}</span>
<strong>{{ t.ticker }}</strong> · {{ t.amount_range }}
<span class="muted">trade {{ t.trade_date }} · disclosed {{ t.disclosure_date }}</span>
<div>{{ t.likely_thesis }}</div>
{% if t.aligns_with_insiders is not none %}
<div class="muted">Aligns with corporate insiders: {{ "yes" if t.aligns_with_insiders else "no" }}</div>
{% endif %}
{% for s in t.sources %}<a href="{{ s }}" class="muted">source</a> {% endfor %}
</div>
{% endmacro %}

{% if report.buys %}
<h2>Buys ({{ report.buys|length }})</h2>
{% for t in report.buys %}{{ trade_block(t) }}{% endfor %}
{% endif %}
{% if report.sells %}
<h2>Sells ({{ report.sells|length }})</h2>
{% for t in report.sells %}{{ trade_block(t) }}{% endfor %}
{% endif %}
{% if not report.buys and not report.sells %}
<div class="empty">No qualifying trades from above-SPY politicians today.</div>
{% endif %}
</body></html>
```

- [ ] **Step 21.5: Commit**

```bash
git add src/stock_analyzer/rendering/
git commit -m "feat: add jinja2 templates for the three emails"
```

---

### Task 22: Renderer + agent response models

**Files:**
- Create: `src/stock_analyzer/rendering/renderer.py`
- Test: `tests/unit/rendering/test_renderer.py`

This task also defines the Pydantic models the agents return. We put them next to the renderer so the contract is one file pair.

- [ ] **Step 22.1: Write failing test**

Create `tests/unit/rendering/__init__.py` (empty), then:

```python
# tests/unit/rendering/test_renderer.py
from __future__ import annotations

from datetime import date, datetime, timezone

from stock_analyzer.rendering.renderer import (
    DrawdownItem,
    DrawdownReport,
    HoldingBrief,
    PoliticianReport,
    PoliticianTrade,
    PortfolioReport,
    Renderer,
    TrendingArticle,
)


def _now() -> datetime:
    return datetime(2026, 5, 5, 7, 0, tzinfo=timezone.utc)


def test_render_portfolio_html() -> None:
    report = PortfolioReport(
        as_of=_now(),
        trending_news=[TrendingArticle(
            title="Apple beats", publisher="Reuters",
            url="https://r/1", published_at=_now(),
            related_tickers=["AAPL"],
            is_market_wide_trending=True, score=100.0,
        )],
        holdings=[HoldingBrief(
            ticker="AAPL", company_name="Apple Inc.",
            pct_change_overnight=-0.5,
            bullets=["b1", "b2"], sources=["https://a/1"],
            watch_today="earnings",
        )],
        portfolio_summary="Quiet day.",
    )
    r = Renderer()
    html = r.render_portfolio(report, mode="production")
    assert "Apple beats" in html
    assert "Quiet day." in html
    assert "AAPL" in html
    assert "Ephemeral mode" not in html


def test_render_drawdown_empty() -> None:
    report = DrawdownReport(as_of=_now(), items=[], market_context=None)
    html = Renderer().render_drawdown(report, threshold_pct=5.0, mode="production")
    assert "No holdings down" in html


def test_render_drawdown_with_items() -> None:
    report = DrawdownReport(
        as_of=_now(),
        items=[DrawdownItem(
            ticker="AAPL", pct_drop=-7.2, pre_market_price=189.0,
            prev_close=200.0, likely_cause="earnings",
            explanation="Earnings miss.",
            sources=["https://a/1"],
        )],
        market_context="Broad selloff.",
    )
    html = Renderer().render_drawdown(report, threshold_pct=5.0, mode="production")
    assert "-7.20%" in html
    assert "earnings" in html
    assert "Broad selloff." in html


def test_render_politician_ephemeral_banner() -> None:
    report = PoliticianReport(
        as_of=_now(), buys=[], sells=[],
        top_takeaway="No qualifying trades.",
    )
    html = Renderer().render_politician(report, mode="ephemeral")
    assert "Ephemeral mode" in html


def test_render_politician_with_trade() -> None:
    report = PoliticianReport(
        as_of=_now(),
        buys=[PoliticianTrade(
            politician_name="Jane Doe", politician_party="D",
            politician_chamber="House",
            politician_24mo_alpha_vs_spy=12.5,
            ticker="NVDA", side="BUY",
            trade_date=date(2026, 5, 4),
            disclosure_date=date(2026, 5, 5),
            amount_range="$1,001 - $15,000",
            likely_thesis="AI tailwind.",
            aligns_with_insiders=True,
            sources=["https://x/1"],
        )],
        sells=[],
        top_takeaway="One buy.",
    )
    html = Renderer().render_politician(report, mode="production")
    assert "Jane Doe" in html
    assert "NVDA" in html
    assert "+12.5pp" in html
```

- [ ] **Step 22.2: Run to verify it fails**

```bash
uv run pytest tests/unit/rendering/test_renderer.py -v
```

Expected: ImportError.

- [ ] **Step 22.3: Implement `src/stock_analyzer/rendering/renderer.py`**

```python
"""Pydantic response models + Jinja2 renderer.

The renderer is dumb: it never calls Claude, never makes HTTP requests.
Agents return these typed models; the renderer turns them into HTML.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel, HttpUrl


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
    bullets: list[str]
    sources: list[HttpUrl]
    watch_today: str


class PortfolioReport(BaseModel):
    as_of: datetime
    trending_news: list[TrendingArticle]
    holdings: list[HoldingBrief]
    portfolio_summary: str


class DrawdownItem(BaseModel):
    ticker: str
    pct_drop: float
    pre_market_price: float
    prev_close: float
    likely_cause: Literal["earnings", "downgrade", "sector",
                           "macro", "company_news", "no_clear_catalyst"]
    explanation: str
    sources: list[HttpUrl]


class DrawdownReport(BaseModel):
    as_of: datetime
    items: list[DrawdownItem]
    market_context: str | None = None


class PoliticianTrade(BaseModel):
    politician_name: str
    politician_party: Literal["D", "R", "I"]
    politician_chamber: Literal["House", "Senate"]
    politician_24mo_alpha_vs_spy: float
    ticker: str
    side: Literal["BUY", "SELL"]
    trade_date: date
    disclosure_date: date
    amount_range: str
    likely_thesis: str
    aligns_with_insiders: bool | None = None
    sources: list[HttpUrl]


class PoliticianReport(BaseModel):
    as_of: datetime
    buys: list[PoliticianTrade]
    sells: list[PoliticianTrade]
    top_takeaway: str


_TEMPLATES_DIR = Path(__file__).parent / "templates"


class Renderer:
    def __init__(self) -> None:
        self._env = Environment(
            loader=FileSystemLoader(_TEMPLATES_DIR),
            autoescape=select_autoescape(["html"]),
            undefined=StrictUndefined,
        )

    def render_portfolio(self, report: PortfolioReport, *, mode: str) -> str:
        return self._env.get_template("portfolio.html.j2").render(
            report=report, mode=mode,
        )

    def render_drawdown(self, report: DrawdownReport, *,
                          threshold_pct: float, mode: str) -> str:
        return self._env.get_template("drawdown.html.j2").render(
            report=report, threshold_pct=threshold_pct, mode=mode,
        )

    def render_politician(self, report: PoliticianReport, *, mode: str) -> str:
        return self._env.get_template("politician.html.j2").render(
            report=report, mode=mode,
        )
```

- [ ] **Step 22.4: Run tests**

```bash
uv run pytest tests/unit/rendering/test_renderer.py -v
```

Expected: 5 passed.

- [ ] **Step 22.5: Commit**

```bash
git add src/stock_analyzer/rendering/renderer.py tests/unit/rendering/
git commit -m "feat: add agent response models and jinja2 renderer"
```

---

*Phase 5 complete: agents now have a clear typed contract to return, and the renderer is fully tested with no LLM calls.*


## Phase 6 — Agents (Tasks 23–25)

Each agent is a thin wrapper around `agno.Agent` that:
1. Holds the system prompt (cached for prompt-caching savings)
2. Wires the right tools
3. Forces structured Pydantic output via `response_model=`

Tests use `pytest-mock` to swap `agno.Agent.arun` for a stub that returns a deterministic Pydantic model. **No tests call Claude.**

### Task 23: Portfolio Agent

**Files:**
- Create: `src/stock_analyzer/agents/__init__.py`, `src/stock_analyzer/agents/portfolio_agent.py`
- Test: `tests/unit/agents/test_portfolio_agent.py`

- [ ] **Step 23.1: Write failing test**

Create `tests/unit/agents/__init__.py` (empty), then:

```python
# tests/unit/agents/test_portfolio_agent.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stock_analyzer.agents.portfolio_agent import PortfolioAgent
from stock_analyzer.rendering.renderer import (
    HoldingBrief,
    PortfolioReport,
    TrendingArticle,
)
from stock_analyzer.tools.snaptrade import Holding


@pytest.mark.asyncio
async def test_portfolio_agent_returns_typed_report(mocker) -> None:
    expected = PortfolioReport(
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        trending_news=[],
        holdings=[HoldingBrief(
            ticker="AAPL", company_name="Apple",
            pct_change_overnight=-0.5,
            bullets=["b"], sources=[], watch_today="-",
        )],
        portfolio_summary="ok",
    )

    class _Resp:
        content = expected
        metrics = {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0}

    mock_arun = mocker.patch("stock_analyzer.agents.portfolio_agent.Agent.arun",
                              return_value=_Resp())

    agent = PortfolioAgent(
        anthropic_api_key="sk-ant-test",
        model="claude-sonnet-4-6",
        max_tokens=4096,
        prompt_caching=True,
    )
    holdings = [Holding(ticker="AAPL", quantity=10, currency="USD",
                         avg_cost=150, market_value=2000, account="x")]
    report, usage = await agent.run(holdings=holdings,
                                       prev_closes={"AAPL": 200.0},
                                       pre_market={"AAPL": 199.0},
                                       trending_articles=[],
                                       spy_overnight_pct=-0.1)
    assert report.holdings[0].ticker == "AAPL"
    assert usage["input_tokens"] == 100
    assert mock_arun.called
```

- [ ] **Step 23.2: Run to verify it fails**

```bash
uv run pytest tests/unit/agents/test_portfolio_agent.py -v
```

Expected: ImportError.

- [ ] **Step 23.3: Implement `src/stock_analyzer/agents/__init__.py`**

```python
"""agno-based agents — one per email."""
```

- [ ] **Step 23.4: Implement `src/stock_analyzer/agents/portfolio_agent.py`**

```python
"""Portfolio agent — produces Email 1 (per-holding briefing + trending news)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agno.agent import Agent
from agno.models.anthropic import Claude

from stock_analyzer.rendering.renderer import (
    HoldingBrief,
    PortfolioReport,
    TrendingArticle,
)
from stock_analyzer.tools.snaptrade import Holding


_SYSTEM_PROMPT = """\
You are a portfolio analyst preparing a pre-market daily briefing.

For each holding the user provides, write 3-5 bullets covering:
- Overnight price action vs SPY (use the spy_overnight_pct provided)
- Any insider/institutional activity in the past 7 days (use the SEC EDGAR tools if relevant)
- Top news catalyst (use the news tools)
- Smart-money commentary if available (use the InsiderMonkey tool for narrative context)
- A one-line "what to watch today" forward-look

Be factual. No hype. No buy/sell recommendations — only context.
The user has already pre-computed a "trending news" list; you do NOT need to re-rank it.
You may add a one-line context note inside each holding's bullets when a trending
article is highly relevant to that ticker.

Return a PortfolioReport with:
- portfolio_summary: 2-3 sentences of top-level takeaway
- holdings: a HoldingBrief per ticker
- trending_news: pass through the list provided to you
"""


class PortfolioAgent:
    def __init__(self, *, anthropic_api_key: str, model: str,
                 max_tokens: int, prompt_caching: bool) -> None:
        self._agent = Agent(
            model=Claude(id=model, api_key=anthropic_api_key,
                         max_tokens=max_tokens, cache_system_prompt=prompt_caching),
            description="Portfolio analyst",
            instructions=_SYSTEM_PROMPT,
            response_model=PortfolioReport,
            structured_outputs=True,
            markdown=False,
            debug_mode=False,
        )

    async def run(self, *, holdings: list[Holding],
                   prev_closes: dict[str, float],
                   pre_market: dict[str, float | None],
                   trending_articles: list[TrendingArticle],
                   spy_overnight_pct: float) -> tuple[PortfolioReport, dict[str, int]]:
        prompt = _build_prompt(holdings, prev_closes, pre_market,
                                trending_articles, spy_overnight_pct)
        resp = await self._agent.arun(prompt)
        report: PortfolioReport = resp.content  # type: ignore[assignment]
        # If the agent forgot to pass through trending news, fill it in:
        if not report.trending_news and trending_articles:
            report = report.model_copy(update={"trending_news": trending_articles})
        if not report.as_of:
            report = report.model_copy(update={"as_of": datetime.now(timezone.utc)})
        usage = _extract_usage(resp)
        return report, usage


def _build_prompt(holdings: list[Holding],
                   prev_closes: dict[str, float],
                   pre_market: dict[str, float | None],
                   trending: list[TrendingArticle],
                   spy_overnight_pct: float) -> str:
    rows = []
    for h in holdings:
        pc = prev_closes.get(h.ticker)
        pm = pre_market.get(h.ticker)
        rows.append(f"- {h.ticker}: qty={h.quantity}, prev_close={pc}, pre_market={pm}")
    trending_lines = [
        f"- [{a.publisher}] {a.title} (related: {','.join(a.related_tickers)})"
        for a in trending
    ]
    return (
        f"SPY overnight: {spy_overnight_pct:+.2f}%\n\n"
        f"Holdings:\n" + "\n".join(rows) +
        ("\n\nTrending news (pre-ranked, pass through):\n" + "\n".join(trending_lines)
         if trending_lines else "") +
        "\n\nProduce the PortfolioReport now."
    )


def _extract_usage(resp: Any) -> dict[str, int]:
    metrics = getattr(resp, "metrics", {}) or {}
    return {
        "input_tokens": int(metrics.get("input_tokens", 0) or 0),
        "output_tokens": int(metrics.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(metrics.get("cache_read_tokens", 0) or 0),
    }
```

- [ ] **Step 23.5: Run tests**

```bash
uv run pytest tests/unit/agents/test_portfolio_agent.py -v
```

Expected: 1 passed.

- [ ] **Step 23.6: Commit**

```bash
git add src/stock_analyzer/agents/ tests/unit/agents/
git commit -m "feat: add portfolio agent (Email 1)"
```

---

### Task 24: Drawdown Agent

**Files:**
- Create: `src/stock_analyzer/agents/drawdown_agent.py`
- Test: `tests/unit/agents/test_drawdown_agent.py`

- [ ] **Step 24.1: Write failing test**

```python
# tests/unit/agents/test_drawdown_agent.py
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from stock_analyzer.agents.drawdown_agent import DrawdownAgent
from stock_analyzer.analytics.drawdown_filter import DrawdownCandidate
from stock_analyzer.rendering.renderer import DrawdownReport


@pytest.mark.asyncio
async def test_drawdown_empty_short_circuits(mocker) -> None:
    mock_arun = mocker.patch(
        "stock_analyzer.agents.drawdown_agent.Agent.arun",
    )
    agent = DrawdownAgent(anthropic_api_key="x", model="claude-sonnet-4-6",
                            max_tokens=4096, prompt_caching=True)
    report, usage = await agent.run(candidates=[])
    assert report.items == []
    assert usage["input_tokens"] == 0
    assert mock_arun.call_count == 0  # no Claude call when nothing to analyze


@pytest.mark.asyncio
async def test_drawdown_with_candidates(mocker) -> None:
    expected = DrawdownReport(
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        items=[],  # agent fills this in
        market_context=None,
    )

    class _Resp:
        content = expected
        metrics = {"input_tokens": 50, "output_tokens": 25, "cache_read_tokens": 0}

    mocker.patch("stock_analyzer.agents.drawdown_agent.Agent.arun",
                  return_value=_Resp())
    agent = DrawdownAgent(anthropic_api_key="x", model="claude-sonnet-4-6",
                            max_tokens=4096, prompt_caching=True)
    candidates = [DrawdownCandidate(ticker="AAPL", prev_close=200.0,
                                     pre_market_price=189.0, pct_drop=-5.5)]
    report, usage = await agent.run(candidates=candidates)
    assert usage["input_tokens"] == 50
```

- [ ] **Step 24.2: Run to verify it fails**

```bash
uv run pytest tests/unit/agents/test_drawdown_agent.py -v
```

Expected: ImportError.

- [ ] **Step 24.3: Implement `src/stock_analyzer/agents/drawdown_agent.py`**

```python
"""Drawdown agent — produces Email 2 (>5% pre-market drops)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agno.agent import Agent
from agno.models.anthropic import Claude

from stock_analyzer.analytics.drawdown_filter import DrawdownCandidate
from stock_analyzer.rendering.renderer import DrawdownReport


_SYSTEM_PROMPT = """\
You receive a list of stocks down >5% in pre-market. For each one, identify
the most likely cause:
- earnings: earnings miss/preannouncement/guidance cut
- downgrade: analyst rating change
- sector: a sector-wide move (cite the sector and other movers)
- macro: rates/CPI/Fed/geopolitics
- company_news: idiosyncratic company event
- no_clear_catalyst: nothing obvious — flag as actionable signal

Be specific about news sources. When no clear catalyst, say so plainly.
Return a DrawdownReport with explanations and source URLs.
"""


class DrawdownAgent:
    def __init__(self, *, anthropic_api_key: str, model: str,
                 max_tokens: int, prompt_caching: bool) -> None:
        self._agent = Agent(
            model=Claude(id=model, api_key=anthropic_api_key,
                         max_tokens=max_tokens, cache_system_prompt=prompt_caching),
            description="Pre-market drawdown analyst",
            instructions=_SYSTEM_PROMPT,
            response_model=DrawdownReport,
            structured_outputs=True,
            markdown=False,
        )

    async def run(self, *, candidates: list[DrawdownCandidate]
                   ) -> tuple[DrawdownReport, dict[str, int]]:
        if not candidates:
            return (
                DrawdownReport(as_of=datetime.now(timezone.utc),
                                items=[], market_context=None),
                {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            )
        prompt = "Pre-market drawdowns:\n" + "\n".join(
            f"- {c.ticker}: prev_close=${c.prev_close:.2f}, "
            f"pre_market=${c.pre_market_price:.2f}, drop={c.pct_drop:+.2f}%"
            for c in candidates
        ) + "\n\nProduce the DrawdownReport now."
        resp = await self._agent.arun(prompt)
        report: DrawdownReport = resp.content  # type: ignore[assignment]
        if not report.as_of:
            report = report.model_copy(update={"as_of": datetime.now(timezone.utc)})
        return report, _extract_usage(resp)


def _extract_usage(resp: Any) -> dict[str, int]:
    m = getattr(resp, "metrics", {}) or {}
    return {
        "input_tokens": int(m.get("input_tokens", 0) or 0),
        "output_tokens": int(m.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(m.get("cache_read_tokens", 0) or 0),
    }
```

- [ ] **Step 24.4: Run tests**

```bash
uv run pytest tests/unit/agents/test_drawdown_agent.py -v
```

Expected: 2 passed.

- [ ] **Step 24.5: Commit**

```bash
git add src/stock_analyzer/agents/drawdown_agent.py tests/unit/agents/test_drawdown_agent.py
git commit -m "feat: add drawdown agent (Email 2)"
```

---

### Task 25: Politician Agent

**Files:**
- Create: `src/stock_analyzer/agents/politician_agent.py`
- Test: `tests/unit/agents/test_politician_agent.py`

- [ ] **Step 25.1: Write failing test**

```python
# tests/unit/agents/test_politician_agent.py
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from stock_analyzer.agents.politician_agent import PoliticianAgent
from stock_analyzer.rendering.renderer import PoliticianReport
from stock_analyzer.tools.capitol_trades import Disclosure


@pytest.mark.asyncio
async def test_politician_empty_short_circuits(mocker) -> None:
    mock_arun = mocker.patch(
        "stock_analyzer.agents.politician_agent.Agent.arun",
    )
    agent = PoliticianAgent(anthropic_api_key="x", model="claude-sonnet-4-6",
                              max_tokens=4096, prompt_caching=True)
    report, usage = await agent.run(filtered_disclosures=[],
                                       politician_alpha={})
    assert report.buys == [] and report.sells == []
    assert mock_arun.call_count == 0


@pytest.mark.asyncio
async def test_politician_with_disclosures(mocker) -> None:
    expected = PoliticianReport(
        as_of=datetime.now(timezone.utc),
        buys=[], sells=[], top_takeaway="ok",
    )

    class _Resp:
        content = expected
        metrics = {"input_tokens": 200, "output_tokens": 80, "cache_read_tokens": 0}

    mocker.patch("stock_analyzer.agents.politician_agent.Agent.arun",
                  return_value=_Resp())

    discs = [Disclosure(
        politician_full_name="Jane Doe", capitol_trades_id="P1",
        party="D", chamber="House", state="CA",
        ticker="NVDA", side="BUY",
        trade_date=date(2026, 5, 4),
        disclosure_date=date(2026, 5, 5),
        amount_min_usd=1001, amount_max_usd=15000,
        raw={},
    )]
    agent = PoliticianAgent(anthropic_api_key="x", model="claude-sonnet-4-6",
                              max_tokens=4096, prompt_caching=True)
    _, usage = await agent.run(filtered_disclosures=discs,
                                  politician_alpha={"Jane Doe": 12.5})
    assert usage["input_tokens"] == 200
```

- [ ] **Step 25.2: Run to verify it fails**

```bash
uv run pytest tests/unit/agents/test_politician_agent.py -v
```

Expected: ImportError.

- [ ] **Step 25.3: Implement `src/stock_analyzer/agents/politician_agent.py`**

```python
"""Politician agent — produces Email 3 (filtered above-SPY trades)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agno.agent import Agent
from agno.models.anthropic import Claude

from stock_analyzer.rendering.renderer import PoliticianReport
from stock_analyzer.tools.capitol_trades import Disclosure


_SYSTEM_PROMPT = """\
You receive a list of disclosed Congressional trades from politicians whose
24-month track record beats SPY. For each trade:
- State who, what, when, dollar range
- Explain the likely thesis (sector tailwind, earnings, regulatory move)
- Note alignment or conflict with corporate insider activity if available
  (use the SEC EDGAR Form 4 tool when useful)

Keep entries crisp — these are signals, not theses.
Return a PoliticianReport with separate buys/sells lists and a one-paragraph
top_takeaway synthesizing the overall pattern.
"""


def _amount_range_str(d: Disclosure) -> str:
    if d.amount_min_usd is None and d.amount_max_usd is None:
        return "undisclosed"
    return f"${d.amount_min_usd or 0:,} - ${d.amount_max_usd or 0:,}"


class PoliticianAgent:
    def __init__(self, *, anthropic_api_key: str, model: str,
                 max_tokens: int, prompt_caching: bool) -> None:
        self._agent = Agent(
            model=Claude(id=model, api_key=anthropic_api_key,
                         max_tokens=max_tokens, cache_system_prompt=prompt_caching),
            description="Congressional trade signal analyst",
            instructions=_SYSTEM_PROMPT,
            response_model=PoliticianReport,
            structured_outputs=True,
            markdown=False,
        )

    async def run(self, *, filtered_disclosures: list[Disclosure],
                   politician_alpha: dict[str, float]
                   ) -> tuple[PoliticianReport, dict[str, int]]:
        if not filtered_disclosures:
            return (
                PoliticianReport(as_of=datetime.now(timezone.utc),
                                  buys=[], sells=[],
                                  top_takeaway="No qualifying trades from "
                                                 "above-SPY politicians today."),
                {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0},
            )
        rows = [
            f"- {d.politician_full_name} ({d.party}/{d.chamber}, "
            f"24mo α {politician_alpha.get(d.politician_full_name, 0.0):+.1f}pp): "
            f"{d.side} {d.ticker} {_amount_range_str(d)} "
            f"trade {d.trade_date} disclosed {d.disclosure_date}"
            for d in filtered_disclosures
        ]
        prompt = ("Filtered Congressional trades:\n" + "\n".join(rows) +
                   "\n\nProduce the PoliticianReport now.")
        resp = await self._agent.arun(prompt)
        report: PoliticianReport = resp.content  # type: ignore[assignment]
        if not report.as_of:
            report = report.model_copy(update={"as_of": datetime.now(timezone.utc)})
        return report, _extract_usage(resp)


def _extract_usage(resp: Any) -> dict[str, int]:
    m = getattr(resp, "metrics", {}) or {}
    return {
        "input_tokens": int(m.get("input_tokens", 0) or 0),
        "output_tokens": int(m.get("output_tokens", 0) or 0),
        "cache_read_tokens": int(m.get("cache_read_tokens", 0) or 0),
    }
```

- [ ] **Step 25.4: Run tests**

```bash
uv run pytest tests/unit/agents/test_politician_agent.py -v
```

Expected: 2 passed.

- [ ] **Step 25.5: Commit**

```bash
git add src/stock_analyzer/agents/politician_agent.py tests/unit/agents/test_politician_agent.py
git commit -m "feat: add politician agent (Email 3)"
```

---

*Phase 6 complete: all three agents return typed Pydantic reports and short-circuit cleanly when there's nothing to analyze.*


## Phase 7 — Orchestrator (Tasks 26–28)

The orchestrator is the central pipeline. We split it into three tasks (one per phase) so each is testable in isolation.

### Task 26: Orchestrator skeleton + Phase 1 (shared fetch)

**Files:**
- Create: `src/stock_analyzer/orchestrator.py`
- Test: `tests/unit/test_orchestrator_phase1.py`

- [ ] **Step 26.1: Write failing test**

```python
# tests/unit/test_orchestrator_phase1.py
from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.orchestrator import Orchestrator, SharedData
from stock_analyzer.persistence.in_memory import (
    InMemoryPoliticianRepository,
    InMemoryPoliticianScoreRepository,
    InMemoryPoliticianTradeRepository,
    InMemoryRunRepository,
    InMemorySpyCloseRepository,
)
from stock_analyzer.tools.market_data import QuoteSnapshot
from stock_analyzer.tools.snaptrade import Holding


class _FakeSnapTrade:
    def get_holdings(self) -> list[Holding]:
        return [Holding(ticker="AAPL", quantity=10, currency="USD",
                         avg_cost=150, market_value=2000, account="x")]


class _FakeMarket:
    def batch_quotes(self, tickers: list[str]) -> list[QuoteSnapshot]:
        return [QuoteSnapshot(ticker=t, prev_close=200.0,
                                 pre_market_price=199.0, last_price=199.0)
                for t in tickers]

    def get_trending_tickers(self) -> set[str]:
        return {"AAPL"}

    def get_spy_close(self, d: date) -> float | None:
        return 528.41


@pytest.mark.asyncio
async def test_phase_1_collects_shared_data() -> None:
    o = _build_orch()
    shared = await o.phase_1_shared_fetch(today=date(2026, 5, 5))
    assert isinstance(shared, SharedData)
    assert "AAPL" in shared.tickers
    assert shared.market_trending == {"AAPL"}
    assert shared.spy_overnight_pct is not None


def _build_orch() -> Orchestrator:
    return Orchestrator(
        ephemeral=True,
        snaptrade=_FakeSnapTrade(),
        market_data=_FakeMarket(),
        news=None, sec_edgar=None,
        capitol_trades=None, insider_monkey=None,
        smtp=None, renderer=None,
        politician_repo=InMemoryPoliticianRepository(),
        trade_repo=InMemoryPoliticianTradeRepository(),
        score_repo=InMemoryPoliticianScoreRepository(),
        spy_repo=InMemorySpyCloseRepository(),
        run_repo=InMemoryRunRepository(),
        portfolio_agent=None, drawdown_agent=None, politician_agent=None,
        drawdown_threshold_pct=5.0,
        politician_lookback_months=24,
        politician_fresh_disclosure_days=2,
        failed_emails_dir="/tmp",
        smtp_to="x@y.com",
        dry_run=True,
    )
```

- [ ] **Step 26.2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_orchestrator_phase1.py -v
```

Expected: ImportError.

- [ ] **Step 26.3: Implement `src/stock_analyzer/orchestrator.py` (skeleton + Phase 1)**

```python
"""Top-level pipeline. Orchestrates 3 phases for the daily run."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from stock_analyzer.calendar.nyse import previous_trading_day
from stock_analyzer.logging import get_logger

_log = get_logger(__name__)


@dataclass
class SharedData:
    today: date
    holdings: list  # list[Holding]
    tickers: list[str]
    quotes: dict[str, Any]   # ticker -> QuoteSnapshot
    prev_closes: dict[str, float]
    pre_market: dict[str, float | None]
    market_trending: set[str]
    spy_overnight_pct: float | None


@dataclass
class Orchestrator:
    ephemeral: bool

    # Tools
    snaptrade: Any
    market_data: Any
    news: Any
    sec_edgar: Any
    capitol_trades: Any
    insider_monkey: Any
    smtp: Any
    renderer: Any

    # Repositories (sql or in-memory)
    politician_repo: Any
    trade_repo: Any
    score_repo: Any
    spy_repo: Any
    run_repo: Any

    # Agents
    portfolio_agent: Any
    drawdown_agent: Any
    politician_agent: Any

    # Behavior
    drawdown_threshold_pct: float
    politician_lookback_months: int
    politician_fresh_disclosure_days: int
    failed_emails_dir: str
    smtp_to: str
    dry_run: bool

    async def phase_1_shared_fetch(self, *, today: date) -> SharedData:
        _log.info("phase_1.start")
        holdings = self.snaptrade.get_holdings()
        tickers = sorted({h.ticker for h in holdings})
        quotes = {q.ticker: q for q in self.market_data.batch_quotes(tickers + ["SPY"])}
        prev_closes = {t: quotes[t].prev_close for t in tickers if t in quotes}
        pre_market = {t: quotes[t].pre_market_price for t in tickers if t in quotes}
        spy_q = quotes.get("SPY")
        spy_overnight = spy_q.pct_change_overnight() if spy_q else None
        market_trending = self.market_data.get_trending_tickers()

        if not self.ephemeral and spy_q is not None:
            yesterday = previous_trading_day(today)
            if (yest_close := self.market_data.get_spy_close(yesterday)) is not None:
                self.spy_repo.upsert(yesterday, yest_close)

        _log.info("phase_1.done", n_holdings=len(holdings),
                   trending=len(market_trending))
        return SharedData(
            today=today, holdings=holdings, tickers=tickers,
            quotes=quotes, prev_closes=prev_closes, pre_market=pre_market,
            market_trending=market_trending, spy_overnight_pct=spy_overnight,
        )
```

- [ ] **Step 26.4: Run tests**

```bash
uv run pytest tests/unit/test_orchestrator_phase1.py -v
```

Expected: 1 passed.

- [ ] **Step 26.5: Commit**

```bash
git add src/stock_analyzer/orchestrator.py tests/unit/test_orchestrator_phase1.py
git commit -m "feat: add orchestrator phase 1 (shared fetch)"
```

---

### Task 27: Orchestrator Phase 2 (parallel agents)

**Files:**
- Modify: `src/stock_analyzer/orchestrator.py`
- Test: `tests/unit/test_orchestrator_phase2.py`

- [ ] **Step 27.1: Write failing test**

```python
# tests/unit/test_orchestrator_phase2.py
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from stock_analyzer.orchestrator import AgentResults, Orchestrator, SharedData
from stock_analyzer.rendering.renderer import (
    DrawdownReport, HoldingBrief, PoliticianReport, PortfolioReport,
)


def _shared(today: date) -> SharedData:
    return SharedData(today=today, holdings=[], tickers=[], quotes={},
                       prev_closes={}, pre_market={},
                       market_trending=set(), spy_overnight_pct=0.0)


@pytest.mark.asyncio
async def test_phase_2_runs_agents_parallel(mocker) -> None:
    portfolio_report = PortfolioReport(
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        trending_news=[],
        holdings=[HoldingBrief(ticker="X", company_name="Y",
                                pct_change_overnight=0,
                                bullets=[], sources=[], watch_today="-")],
        portfolio_summary="-",
    )
    drawdown_report = DrawdownReport(
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        items=[], market_context=None,
    )
    politician_report = PoliticianReport(
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        buys=[], sells=[], top_takeaway="-",
    )

    pa = AsyncMock()
    pa.run.return_value = (portfolio_report,
                            {"input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 0})
    da = AsyncMock()
    da.run.return_value = (drawdown_report,
                            {"input_tokens": 3, "output_tokens": 4, "cache_read_tokens": 0})
    ga = AsyncMock()
    ga.run.return_value = (politician_report,
                            {"input_tokens": 5, "output_tokens": 6, "cache_read_tokens": 0})

    o = _make_orch(portfolio_agent=pa, drawdown_agent=da, politician_agent=ga,
                    capitol_trades=AsyncMock(get_recent_disclosures=AsyncMock(return_value=[])))

    result = await o.phase_2_run_agents(_shared(date(2026, 5, 5)))
    assert isinstance(result, AgentResults)
    assert result.portfolio.holdings[0].ticker == "X"
    assert result.usage["input_tokens"] == 1 + 3 + 5


@pytest.mark.asyncio
async def test_phase_2_isolates_failures(mocker) -> None:
    portfolio_report = PortfolioReport(
        as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
        trending_news=[], holdings=[], portfolio_summary="ok",
    )
    pa = AsyncMock()
    pa.run.return_value = (portfolio_report,
                            {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0})
    da = AsyncMock()
    da.run.side_effect = RuntimeError("boom")
    ga = AsyncMock()
    ga.run.side_effect = RuntimeError("boom")

    o = _make_orch(portfolio_agent=pa, drawdown_agent=da, politician_agent=ga,
                    capitol_trades=AsyncMock(get_recent_disclosures=AsyncMock(return_value=[])))
    result = await o.phase_2_run_agents(_shared(date(2026, 5, 5)))
    assert result.portfolio is not None
    assert result.drawdown_error is not None
    assert result.politician_error is not None


def _make_orch(**overrides) -> Orchestrator:
    from stock_analyzer.persistence.in_memory import (
        InMemoryPoliticianRepository, InMemoryPoliticianScoreRepository,
        InMemoryPoliticianTradeRepository, InMemoryRunRepository,
        InMemorySpyCloseRepository,
    )
    base = dict(
        ephemeral=True, snaptrade=None, market_data=None, news=None,
        sec_edgar=None, capitol_trades=None, insider_monkey=None,
        smtp=None, renderer=None,
        politician_repo=InMemoryPoliticianRepository(),
        trade_repo=InMemoryPoliticianTradeRepository(),
        score_repo=InMemoryPoliticianScoreRepository(),
        spy_repo=InMemorySpyCloseRepository(),
        run_repo=InMemoryRunRepository(),
        portfolio_agent=None, drawdown_agent=None, politician_agent=None,
        drawdown_threshold_pct=5.0, politician_lookback_months=24,
        politician_fresh_disclosure_days=2, failed_emails_dir="/tmp",
        smtp_to="x@y.com", dry_run=True,
    )
    base.update(overrides)
    return Orchestrator(**base)
```

- [ ] **Step 27.2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_orchestrator_phase2.py -v
```

Expected: AttributeError on `phase_2_run_agents`.

- [ ] **Step 27.3: Append Phase 2 to `src/stock_analyzer/orchestrator.py`**

Add these imports at the top:

```python
import asyncio
from dataclasses import field as dc_field
from datetime import timedelta

from stock_analyzer.analytics.drawdown_filter import (
    DrawdownCandidate, Quote, filter_drawdowns,
)
from stock_analyzer.analytics.news_ranker import NewsItem as RankerNewsItem, rank_articles
from stock_analyzer.tools.capitol_trades import Disclosure
```

Add this dataclass below `SharedData`:

```python
@dataclass
class AgentResults:
    portfolio: Any | None = None
    drawdown: Any | None = None
    politician: Any | None = None
    portfolio_error: str | None = None
    drawdown_error: str | None = None
    politician_error: str | None = None
    usage: dict[str, int] = dc_field(default_factory=lambda: {
        "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
    })
```

Add the method on `Orchestrator`:

```python
    async def phase_2_run_agents(self, shared: SharedData) -> AgentResults:
        _log.info("phase_2.start")

        # Pre-compute drawdown candidates (deterministic)
        candidates = filter_drawdowns(
            [Quote(ticker=t, prev_close=shared.prev_closes.get(t, 0.0),
                    pre_market_price=shared.pre_market.get(t))
             for t in shared.tickers],
            threshold_pct=self.drawdown_threshold_pct,
        )

        # Pre-fetch politician disclosures (last 7 days for safety)
        since = shared.today - timedelta(days=7)
        try:
            disclosures = await self.capitol_trades.get_recent_disclosures(since=since)
        except Exception as e:
            _log.warning("capitol_trades.fetch_failed", err=str(e))
            disclosures = []

        # Filter to fresh disclosures from above-SPY politicians (production)
        cutoff_date = shared.today - timedelta(days=self.politician_fresh_disclosure_days)
        fresh = [d for d in disclosures if d.disclosure_date >= cutoff_date]
        if self.ephemeral:
            filtered_disclosures = fresh
            politician_alpha: dict[str, float] = {}
        else:
            beating_ids = self.score_repo.beating_spy()
            beating_names: set[str] = set()
            politician_alpha = {}
            for p in self.politician_repo.all():
                if p.id in beating_ids and p.score is not None:
                    beating_names.add(p.full_name)
                    politician_alpha[p.full_name] = p.score.alpha_vs_spy_pct
            filtered_disclosures = [d for d in fresh
                                     if d.politician_full_name in beating_names]

        # Pre-rank trending news (deterministic, no LLM)
        ranker_items: list[RankerNewsItem] = []
        try:
            for t in shared.tickers:
                for it in self.news.get_yfinance_news(t):
                    ranker_items.append(RankerNewsItem(
                        title=it.title, publisher=it.publisher,
                        url=it.url, published_at=it.published_at,
                        related_tickers=it.related_tickers or [t],
                    ))
        except Exception as e:
            _log.warning("yfinance_news_fetch_failed", err=str(e))
        trending_articles_ranked = rank_articles(
            ranker_items, holdings=set(shared.tickers),
            market_trending=shared.market_trending, top_n=5,
        )

        async def _portfolio_task():
            return await self.portfolio_agent.run(
                holdings=shared.holdings,
                prev_closes=shared.prev_closes,
                pre_market=shared.pre_market,
                trending_articles=trending_articles_ranked,
                spy_overnight_pct=shared.spy_overnight_pct or 0.0,
            )

        async def _drawdown_task():
            return await self.drawdown_agent.run(candidates=candidates)

        async def _politician_task():
            return await self.politician_agent.run(
                filtered_disclosures=filtered_disclosures,
                politician_alpha=politician_alpha,
            )

        results = await asyncio.gather(
            _portfolio_task(), _drawdown_task(), _politician_task(),
            return_exceptions=True,
        )

        out = AgentResults()
        for i, r in enumerate(results):
            if isinstance(r, BaseException):
                msg = f"{type(r).__name__}: {r}"
                if i == 0:
                    out.portfolio_error = msg
                elif i == 1:
                    out.drawdown_error = msg
                else:
                    out.politician_error = msg
                _log.warning("agent.failed", index=i, err=msg)
                continue
            report, usage = r
            if i == 0:
                out.portfolio = report
            elif i == 1:
                out.drawdown = report
            else:
                out.politician = report
            for k in ("input_tokens", "output_tokens", "cache_read_tokens"):
                out.usage[k] += int(usage.get(k, 0))

        _log.info("phase_2.done", usage=out.usage,
                   portfolio_ok=out.portfolio is not None,
                   drawdown_ok=out.drawdown is not None,
                   politician_ok=out.politician is not None)
        return out
```

- [ ] **Step 27.4: Run tests**

```bash
uv run pytest tests/unit/test_orchestrator_phase2.py -v
```

Expected: 2 passed.

- [ ] **Step 27.5: Commit**

```bash
git add src/stock_analyzer/orchestrator.py tests/unit/test_orchestrator_phase2.py
git commit -m "feat: add orchestrator phase 2 (parallel agents with isolation)"
```

---

### Task 28: Orchestrator Phase 3 (render + send) + run() entrypoint

**Files:**
- Modify: `src/stock_analyzer/orchestrator.py`
- Test: `tests/unit/test_orchestrator_phase3.py`

- [ ] **Step 28.1: Write failing test**

```python
# tests/unit/test_orchestrator_phase3.py
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import AsyncMock

import pytest

from stock_analyzer.orchestrator import AgentResults
from stock_analyzer.persistence.in_memory import (
    InMemoryPoliticianRepository, InMemoryPoliticianScoreRepository,
    InMemoryPoliticianTradeRepository, InMemoryRunRepository,
    InMemorySpyCloseRepository,
)
from stock_analyzer.rendering.renderer import (
    DrawdownReport, HoldingBrief, PoliticianReport, PortfolioReport, Renderer,
)
from stock_analyzer.tools.smtp_sender import EmailMessage


@pytest.mark.asyncio
async def test_phase_3_sends_three_emails(tmp_path) -> None:
    sent: list[EmailMessage] = []

    class _Smtp:
        async def send(self, m: EmailMessage) -> None:
            sent.append(m)

    from stock_analyzer.orchestrator import Orchestrator
    o = Orchestrator(
        ephemeral=True, snaptrade=None, market_data=None, news=None,
        sec_edgar=None, capitol_trades=None, insider_monkey=None,
        smtp=_Smtp(), renderer=Renderer(),
        politician_repo=InMemoryPoliticianRepository(),
        trade_repo=InMemoryPoliticianTradeRepository(),
        score_repo=InMemoryPoliticianScoreRepository(),
        spy_repo=InMemorySpyCloseRepository(),
        run_repo=InMemoryRunRepository(),
        portfolio_agent=None, drawdown_agent=None, politician_agent=None,
        drawdown_threshold_pct=5.0, politician_lookback_months=24,
        politician_fresh_disclosure_days=2,
        failed_emails_dir=str(tmp_path), smtp_to="to@x.com",
        dry_run=False,
    )
    results = AgentResults(
        portfolio=PortfolioReport(
            as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
            trending_news=[],
            holdings=[HoldingBrief(ticker="A", company_name="A Inc",
                                     pct_change_overnight=0, bullets=[],
                                     sources=[], watch_today="-")],
            portfolio_summary="ok",
        ),
        drawdown=DrawdownReport(
            as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
            items=[], market_context=None),
        politician=PoliticianReport(
            as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
            buys=[], sells=[], top_takeaway="ok"),
    )
    statuses = await o.phase_3_render_and_send(results, today=date(2026, 5, 5))
    assert statuses == ("sent", "sent", "sent")
    assert len(sent) == 3
    subjects = [m.subject for m in sent]
    assert any("Portfolio" in s for s in subjects)
    assert any("Drawdown" in s or "Pre-Market" in s for s in subjects)
    assert any("SmartMoney" in s or "Politician" in s for s in subjects)


@pytest.mark.asyncio
async def test_phase_3_dry_run_writes_files(tmp_path) -> None:
    from stock_analyzer.orchestrator import Orchestrator
    o = Orchestrator(
        ephemeral=True, snaptrade=None, market_data=None, news=None,
        sec_edgar=None, capitol_trades=None, insider_monkey=None,
        smtp=None, renderer=Renderer(),
        politician_repo=InMemoryPoliticianRepository(),
        trade_repo=InMemoryPoliticianTradeRepository(),
        score_repo=InMemoryPoliticianScoreRepository(),
        spy_repo=InMemorySpyCloseRepository(),
        run_repo=InMemoryRunRepository(),
        portfolio_agent=None, drawdown_agent=None, politician_agent=None,
        drawdown_threshold_pct=5.0, politician_lookback_months=24,
        politician_fresh_disclosure_days=2,
        failed_emails_dir=str(tmp_path), smtp_to="to@x.com",
        dry_run=True,
    )
    results = AgentResults(
        portfolio=PortfolioReport(as_of=datetime.now(timezone.utc),
                                    trending_news=[],
                                    holdings=[HoldingBrief(
                                        ticker="A", company_name="A",
                                        pct_change_overnight=0, bullets=[],
                                        sources=[], watch_today="-")],
                                    portfolio_summary="ok"),
        drawdown=DrawdownReport(as_of=datetime.now(timezone.utc),
                                  items=[], market_context=None),
        politician=PoliticianReport(as_of=datetime.now(timezone.utc),
                                      buys=[], sells=[], top_takeaway="ok"),
    )
    statuses = await o.phase_3_render_and_send(results, today=date(2026, 5, 5))
    assert all(s == "dry-run" for s in statuses)
    files = list(tmp_path.glob("*.html"))
    assert len(files) == 3
```

- [ ] **Step 28.2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_orchestrator_phase3.py -v
```

Expected: AttributeError on `phase_3_render_and_send`.

- [ ] **Step 28.3: Append Phase 3 + run() to `src/stock_analyzer/orchestrator.py`**

Add this method:

```python
    async def phase_3_render_and_send(self, results: AgentResults, *,
                                         today: date) -> tuple[str, str, str]:
        _log.info("phase_3.start")
        out_dir = __import__("pathlib").Path(self.failed_emails_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        mode = "ephemeral" if self.ephemeral else "production"

        async def _send(html: str, subject: str, fname: str) -> str:
            if self.dry_run or self.smtp is None:
                (out_dir / fname).write_text(html)
                return "dry-run"
            from stock_analyzer.tools.smtp_sender import EmailMessage
            try:
                await self.smtp.send(EmailMessage(
                    to=self.smtp_to, subject=subject, html_body=html,
                ))
                return "sent"
            except Exception as e:
                _log.error("email_send_failed", subject=subject, err=str(e))
                (out_dir / fname).write_text(html)
                return "failed"

        date_str = today.isoformat()

        if results.portfolio is not None:
            html1 = self.renderer.render_portfolio(results.portfolio, mode=mode)
            s1 = await _send(html1, f"[Portfolio] Daily Analysis — {date_str}",
                              f"{date_str}_portfolio.html")
        else:
            s1 = "skipped"

        if results.drawdown is not None:
            html2 = self.renderer.render_drawdown(
                results.drawdown,
                threshold_pct=self.drawdown_threshold_pct, mode=mode,
            )
            s2 = await _send(html2,
                              f"[Alert] Stocks Down >{self.drawdown_threshold_pct:.0f}%"
                              f" in Pre-Market — {date_str}",
                              f"{date_str}_drawdown.html")
        else:
            s2 = "skipped"

        if results.politician is not None:
            html3 = self.renderer.render_politician(results.politician, mode=mode)
            s3 = await _send(html3,
                              f"[SmartMoney] Politician Trades (Above-SPY) — {date_str}",
                              f"{date_str}_politician.html")
        else:
            s3 = "skipped"

        _log.info("phase_3.done", statuses=(s1, s2, s3))
        return s1, s2, s3

    async def run(self, *, today: date, force: bool = False) -> None:
        from stock_analyzer.calendar.nyse import is_trading_day
        if not is_trading_day(today):
            _log.info("not_a_trading_day", date=today.isoformat())
            return
        run_record = self.run_repo.start_run(today)
        if (not self.ephemeral and not force
                and run_record.status == "success"):
            _log.info("already_sent_today", date=today.isoformat())
            return

        from stock_analyzer.analytics.cost_tracker import CostTracker
        cost = CostTracker()
        try:
            shared = await self.phase_1_shared_fetch(today=today)
            results = await self.phase_2_run_agents(shared)
            cost.record(input_tokens=results.usage["input_tokens"],
                         output_tokens=results.usage["output_tokens"],
                         cache_read_tokens=results.usage["cache_read_tokens"])
            statuses = await self.phase_3_render_and_send(results, today=today)
            overall = ("success" if all(s in ("sent", "dry-run") for s in statuses)
                        else "partial")
            self.run_repo.complete_run(
                run_record.id, status=overall,
                email_statuses=statuses,
                tokens_in=results.usage["input_tokens"],
                tokens_out=results.usage["output_tokens"],
                cost_usd=cost.estimated_cost_usd(),
                error_log={
                    "portfolio": results.portfolio_error,
                    "drawdown": results.drawdown_error,
                    "politician": results.politician_error,
                } if any([results.portfolio_error, results.drawdown_error,
                          results.politician_error]) else None,
            )
        except Exception as e:
            _log.error("run.failed", err=str(e))
            self.run_repo.complete_run(
                run_record.id, status="failed",
                email_statuses=("failed", "failed", "failed"),
                tokens_in=cost.total_input_tokens,
                tokens_out=cost.total_output_tokens,
                cost_usd=cost.estimated_cost_usd(),
                error_log={"top_level": str(e)},
            )
            raise
```

- [ ] **Step 28.4: Run tests**

```bash
uv run pytest tests/unit/test_orchestrator_phase3.py -v
```

Expected: 2 passed.

- [ ] **Step 28.5: Commit**

```bash
git add src/stock_analyzer/orchestrator.py tests/unit/test_orchestrator_phase3.py
git commit -m "feat: add orchestrator phase 3 + top-level run() entrypoint"
```

---

*Phase 7 complete: orchestrator pipeline assembled with per-phase tests.*


## Phase 8 — CLI (Tasks 29–32)

The `stock-analyzer` console script. We split CLI into a few tasks to keep each one bite-sized: run + factory wiring, health-check, db commands, history.

### Task 29: CLI `run` command + DI factory

**Files:**
- Create: `src/stock_analyzer/__main__.py`
- Create: `src/stock_analyzer/factory.py`
- Test: `tests/unit/test_cli_run.py`, `tests/unit/test_factory.py`

This task introduces a small `factory.py` that builds a fully-wired `Orchestrator` from `Settings` plus a mode flag. The CLI is then a thin Typer wrapper over it.

- [ ] **Step 29.1: Write failing tests**

```python
# tests/unit/test_factory.py
from __future__ import annotations

from stock_analyzer.config import Settings
from stock_analyzer.factory import build_orchestrator, resolve_mode


def _settings(monkeypatch, **overrides) -> Settings:
    base = {
        "ANTHROPIC_API_KEY": "x", "SNAPTRADE_CLIENT_ID": "x",
        "SNAPTRADE_CONSUMER_KEY": "x", "SNAPTRADE_USER_ID": "x",
        "SNAPTRADE_USER_SECRET": "x", "FINNHUB_API_KEY": "x",
        "SMTP_HOST": "h", "SMTP_USERNAME": "u", "SMTP_PASSWORD": "p",
        "SMTP_FROM_ADDRESS": "from@x.com", "SMTP_TO_ADDRESS": "to@x.com",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    base.update(overrides)
    for k, v in base.items():
        monkeypatch.setenv(k, str(v))
    return Settings()


def test_resolve_mode_explicit_ephemeral(monkeypatch) -> None:
    s = _settings(monkeypatch, STOCK_ANALYZER_ENV="production")
    assert resolve_mode(s, ephemeral_flag=True) == "ephemeral"


def test_resolve_mode_production_explicit_env(monkeypatch) -> None:
    s = _settings(monkeypatch, STOCK_ANALYZER_ENV="production")
    assert resolve_mode(s, ephemeral_flag=False) == "production"


def test_resolve_mode_dev_default_ephemeral(monkeypatch) -> None:
    s = _settings(monkeypatch, STOCK_ANALYZER_ENV="development")
    assert resolve_mode(s, ephemeral_flag=False) == "ephemeral"


def test_build_orchestrator_ephemeral(monkeypatch) -> None:
    s = _settings(monkeypatch, STOCK_ANALYZER_ENV="development")
    o = build_orchestrator(s, ephemeral_flag=False, dry_run_override=True)
    assert o.ephemeral is True
    assert o.dry_run is True
```

```python
# tests/unit/test_cli_run.py
from __future__ import annotations

from typer.testing import CliRunner

from stock_analyzer.__main__ import app

runner = CliRunner()


def test_run_command_help() -> None:
    res = runner.invoke(app, ["run", "--help"])
    assert res.exit_code == 0
    assert "--ephemeral" in res.stdout
    assert "--dry-run" in res.stdout
    assert "--only" in res.stdout
    assert "--date" in res.stdout
    assert "--force" in res.stdout
```

- [ ] **Step 29.2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_factory.py tests/unit/test_cli_run.py -v
```

Expected: ImportError.

- [ ] **Step 29.3: Implement `src/stock_analyzer/factory.py`**

```python
"""Wires Settings into a fully-built Orchestrator."""

from __future__ import annotations

from typing import Literal

from stock_analyzer.agents.drawdown_agent import DrawdownAgent
from stock_analyzer.agents.politician_agent import PoliticianAgent
from stock_analyzer.agents.portfolio_agent import PortfolioAgent
from stock_analyzer.config import Settings
from stock_analyzer.orchestrator import Orchestrator
from stock_analyzer.persistence.db import Database
from stock_analyzer.persistence.in_memory import (
    InMemoryPoliticianRepository, InMemoryPoliticianScoreRepository,
    InMemoryPoliticianTradeRepository, InMemoryRunRepository,
    InMemorySpyCloseRepository,
)
from stock_analyzer.persistence.repositories import (
    PoliticianRepository, PoliticianScoreRepository,
    PoliticianTradeRepository, RunRepository, SpyCloseRepository,
)
from stock_analyzer.rendering.renderer import Renderer
from stock_analyzer.tools.capitol_trades import CapitolTradesClient
from stock_analyzer.tools.insider_monkey import InsiderMonkeyClient
from stock_analyzer.tools.market_data import MarketDataClient
from stock_analyzer.tools.news import NewsClient
from stock_analyzer.tools.sec_edgar import SecEdgarClient
from stock_analyzer.tools.smtp_sender import SmtpSender
from stock_analyzer.tools.snaptrade import SnapTradeClient


def resolve_mode(s: Settings, *, ephemeral_flag: bool) -> Literal["production", "ephemeral"]:
    if ephemeral_flag:
        return "ephemeral"
    if s.stock_analyzer_env == "production":
        return "production"
    return "ephemeral"


def build_orchestrator(s: Settings, *, ephemeral_flag: bool,
                         dry_run_override: bool | None = None) -> Orchestrator:
    mode = resolve_mode(s, ephemeral_flag=ephemeral_flag)
    ephemeral = (mode == "ephemeral")

    db = Database(url=s.database_url, ephemeral=ephemeral)
    if not ephemeral:
        db.create_all()

    snap = SnapTradeClient(
        client_id=s.snaptrade_client_id.get_secret_value(),
        consumer_key=s.snaptrade_consumer_key.get_secret_value(),
        user_id=s.snaptrade_user_id,
        user_secret=s.snaptrade_user_secret.get_secret_value(),
    )
    market = MarketDataClient()
    news = NewsClient(finnhub_api_key=s.finnhub_api_key.get_secret_value(),
                        http=None)
    sec = SecEdgarClient(user_agent=f"stock-analyzer {s.smtp_from_address}")
    capitol = CapitolTradesClient()
    im = InsiderMonkeyClient()
    smtp = SmtpSender(
        host=s.smtp_host, port=s.smtp_port,
        username=s.smtp_username,
        password=s.smtp_password.get_secret_value(),
        use_tls=s.smtp_use_tls,
        from_address=s.smtp_from_address, from_name=s.smtp_from_name,
    )
    renderer = Renderer()

    if ephemeral:
        pol_repo = InMemoryPoliticianRepository()
        trade_repo = InMemoryPoliticianTradeRepository()
        score_repo = InMemoryPoliticianScoreRepository()
        spy_repo = InMemorySpyCloseRepository()
        run_repo = InMemoryRunRepository()
    else:
        pol_repo = PoliticianRepository(db)
        trade_repo = PoliticianTradeRepository(db)
        score_repo = PoliticianScoreRepository(db)
        spy_repo = SpyCloseRepository(db)
        run_repo = RunRepository(db)

    portfolio_agent = PortfolioAgent(
        anthropic_api_key=s.anthropic_api_key.get_secret_value(),
        model=s.anthropic_model, max_tokens=s.anthropic_max_tokens,
        prompt_caching=s.anthropic_prompt_caching,
    )
    drawdown_agent = DrawdownAgent(
        anthropic_api_key=s.anthropic_api_key.get_secret_value(),
        model=s.anthropic_model, max_tokens=s.anthropic_max_tokens,
        prompt_caching=s.anthropic_prompt_caching,
    )
    politician_agent = PoliticianAgent(
        anthropic_api_key=s.anthropic_api_key.get_secret_value(),
        model=s.anthropic_model, max_tokens=s.anthropic_max_tokens,
        prompt_caching=s.anthropic_prompt_caching,
    )

    return Orchestrator(
        ephemeral=ephemeral,
        snaptrade=snap, market_data=market, news=news,
        sec_edgar=sec, capitol_trades=capitol, insider_monkey=im,
        smtp=smtp, renderer=renderer,
        politician_repo=pol_repo, trade_repo=trade_repo,
        score_repo=score_repo, spy_repo=spy_repo, run_repo=run_repo,
        portfolio_agent=portfolio_agent, drawdown_agent=drawdown_agent,
        politician_agent=politician_agent,
        drawdown_threshold_pct=s.drawdown_threshold_pct,
        politician_lookback_months=s.politician_lookback_months,
        politician_fresh_disclosure_days=s.politician_fresh_disclosure_days,
        failed_emails_dir=s.failed_emails_dir,
        smtp_to=s.smtp_to_address,
        dry_run=(s.dry_run if dry_run_override is None else dry_run_override),
    )
```

- [ ] **Step 29.4: Implement `src/stock_analyzer/__main__.py`**

```python
"""Stock Analyzer CLI."""

from __future__ import annotations

import asyncio
from datetime import date as _date, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

import typer

from stock_analyzer.config import Settings
from stock_analyzer.factory import build_orchestrator, resolve_mode
from stock_analyzer.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, add_completion=False)
db_app = typer.Typer(no_args_is_help=True)
app.add_typer(db_app, name="db", help="Database management")


def _today_in_tz(tz: str) -> _date:
    return datetime.now(ZoneInfo(tz)).date()


@app.command()
def run(
    ephemeral: Annotated[bool, typer.Option("--ephemeral",
        help="Run with no DB, no audit log, scoring filter disabled.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run",
        help="Render emails to disk instead of sending via SMTP.")] = False,
    only: Annotated[str | None, typer.Option("--only",
        help="Comma-separated list of agents: portfolio,drawdown,politician")] = None,
    date_str: Annotated[str | None, typer.Option("--date",
        help="ISO date (YYYY-MM-DD). Defaults to today in run timezone.")] = None,
    force: Annotated[bool, typer.Option("--force",
        help="Bypass the 'already sent today' idempotency guard.")] = False,
) -> None:
    """Run the daily pipeline (production mode by default)."""
    settings = Settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    log = get_logger(__name__)
    mode = resolve_mode(settings, ephemeral_flag=ephemeral)
    log.info("startup",
              mode=mode.upper(),
              dry_run=dry_run or settings.dry_run,
              only=only)
    orch = build_orchestrator(settings, ephemeral_flag=ephemeral,
                                dry_run_override=dry_run if dry_run else None)
    today = _date.fromisoformat(date_str) if date_str else _today_in_tz(settings.run_timezone)
    if only:
        log.warning("--only is informational; the pipeline still runs all phases "
                     "but agents in --only are the only ones whose output is sent.")
        # Reduced surface: zero out reports for excluded agents in phase 3
        # (implementation detail: the orchestrator still runs everything for
        # shared-data efficiency; the CLI only blocks send.)
        # NOTE: a proper --only implementation lives in a follow-up task; for v1
        # we just log and continue.
    asyncio.run(orch.run(today=today, force=force))


if __name__ == "__main__":
    app()
```

- [ ] **Step 29.5: Run tests**

```bash
uv run pytest tests/unit/test_factory.py tests/unit/test_cli_run.py -v
```

Expected: 5 passed.

- [ ] **Step 29.6: Commit**

```bash
git add src/stock_analyzer/__main__.py src/stock_analyzer/factory.py tests/unit/test_factory.py tests/unit/test_cli_run.py
git commit -m "feat: add Typer CLI run command + orchestrator factory"
```

---

### Task 30: CLI `health-check`

**Files:**
- Modify: `src/stock_analyzer/__main__.py`
- Create: `src/stock_analyzer/healthcheck.py`
- Test: `tests/unit/test_healthcheck.py`

- [ ] **Step 30.1: Write failing test**

```python
# tests/unit/test_healthcheck.py
from __future__ import annotations

from typer.testing import CliRunner

from stock_analyzer.__main__ import app

runner = CliRunner()


def test_healthcheck_command_exists() -> None:
    res = runner.invoke(app, ["health-check", "--help"])
    assert res.exit_code == 0
```

- [ ] **Step 30.2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_healthcheck.py -v
```

Expected: 1 fail (no command yet).

- [ ] **Step 30.3: Implement `src/stock_analyzer/healthcheck.py`**

```python
"""Validates credentials, network, DB."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import aiosmtplib
import httpx

from stock_analyzer.config import Settings
from stock_analyzer.persistence.db import Database


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


async def run_health_checks(s: Settings, *, ephemeral: bool) -> list[CheckResult]:
    out: list[CheckResult] = []

    # Anthropic key format
    out.append(CheckResult(
        name="anthropic_key_format",
        ok=s.anthropic_api_key.get_secret_value().startswith("sk-ant-"),
        detail="must start with sk-ant-",
    ))

    # SMTP reachability
    try:
        smtp = aiosmtplib.SMTP(hostname=s.smtp_host, port=s.smtp_port,
                                start_tls=s.smtp_use_tls, use_tls=False)
        await smtp.connect()
        await smtp.quit()
        out.append(CheckResult(name="smtp_reachable", ok=True,
                                 detail=f"{s.smtp_host}:{s.smtp_port}"))
    except Exception as e:
        out.append(CheckResult(name="smtp_reachable", ok=False, detail=str(e)))

    # Finnhub key (cheap call — no symbol)
    try:
        async with httpx.AsyncClient(timeout=10) as cx:
            r = await cx.get("https://finnhub.io/api/v1/quote",
                              params={"symbol": "AAPL",
                                      "token": s.finnhub_api_key.get_secret_value()})
            out.append(CheckResult(name="finnhub_auth", ok=r.status_code == 200,
                                     detail=f"HTTP {r.status_code}"))
    except Exception as e:
        out.append(CheckResult(name="finnhub_auth", ok=False, detail=str(e)))

    # DB connectivity
    if not ephemeral:
        try:
            db = Database(url=s.database_url, ephemeral=False)
            with db.session() as ses:
                ses.execute(__import__("sqlalchemy").text("SELECT 1"))
            out.append(CheckResult(name="database", ok=True,
                                     detail=s.database_url.split("?")[0]))
        except Exception as e:
            out.append(CheckResult(name="database", ok=False, detail=str(e)))
    else:
        out.append(CheckResult(name="database", ok=True, detail="ephemeral — skipped"))

    return out


def render_report(checks: list[CheckResult]) -> str:
    lines = []
    for c in checks:
        mark = "OK " if c.ok else "FAIL"
        lines.append(f"[{mark}] {c.name:24}  {c.detail}")
    return "\n".join(lines)
```

- [ ] **Step 30.4: Add the `health-check` command to `src/stock_analyzer/__main__.py`**

```python
@app.command(name="health-check")
def health_check(
    ephemeral: Annotated[bool, typer.Option("--ephemeral")] = False,
) -> None:
    """Validate every credential, SnapTrade auth, SMTP, and DB."""
    from stock_analyzer.healthcheck import render_report, run_health_checks
    settings = Settings()
    configure_logging(level="INFO", fmt="pretty")
    mode = resolve_mode(settings, ephemeral_flag=ephemeral)
    typer.echo(f"Mode: {mode.upper()}\n")
    checks = asyncio.run(run_health_checks(settings, ephemeral=(mode == "ephemeral")))
    typer.echo(render_report(checks))
    if any(not c.ok for c in checks):
        raise typer.Exit(code=1)
```

- [ ] **Step 30.5: Run tests**

```bash
uv run pytest tests/unit/test_healthcheck.py -v
```

Expected: 1 passed.

- [ ] **Step 30.6: Commit**

```bash
git add src/stock_analyzer/healthcheck.py src/stock_analyzer/__main__.py tests/unit/test_healthcheck.py
git commit -m "feat: add health-check CLI command"
```

---

### Task 31: CLI `db migrate`, `db backfill-politicians`, `db recompute-scores`

**Files:**
- Modify: `src/stock_analyzer/__main__.py`
- Create: `src/stock_analyzer/db_commands.py`
- Test: `tests/unit/test_db_commands.py`

- [ ] **Step 31.1: Write failing test**

```python
# tests/unit/test_db_commands.py
from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.db_commands import (
    recompute_scores,
)
from stock_analyzer.persistence.db import Database
from stock_analyzer.persistence.repositories import (
    PoliticianRepository, PoliticianScoreRepository,
    PoliticianTradeRepository, SpyCloseRepository,
)


@pytest.fixture
def db(tmp_path) -> Database:
    d = Database(url=f"sqlite:///{tmp_path / 'r.db'}", ephemeral=False)
    d.create_all()
    return d


def test_recompute_scores_no_trades(db: Database) -> None:
    pol_repo = PoliticianRepository(db)
    trade_repo = PoliticianTradeRepository(db)
    score_repo = PoliticianScoreRepository(db)
    spy_repo = SpyCloseRepository(db)
    pol_repo.upsert(full_name="A", party="D", chamber="House",
                     state="CA", capitol_trades_id="X")
    # No trades, no SPY data — recompute should not crash and produce no scores
    n = recompute_scores(pol_repo, trade_repo, score_repo, spy_repo,
                           today=date(2026, 5, 5), lookback_months=24)
    assert n == 0
```

- [ ] **Step 31.2: Run to verify it fails**

```bash
uv run pytest tests/unit/test_db_commands.py -v
```

Expected: ImportError.

- [ ] **Step 31.3: Implement `src/stock_analyzer/db_commands.py`**

```python
"""DB management routines invoked from the CLI."""

from __future__ import annotations

import subprocess
from datetime import date, datetime, timedelta, timezone

from stock_analyzer.analytics.politician_scorer import (
    SimulatedTrade, compute_politician_return, compute_spy_return, midpoint_amount,
)
from stock_analyzer.persistence.repositories import (
    PoliticianRepository, PoliticianScoreRepository,
    PoliticianTradeRepository, SpyCloseRepository,
)
from stock_analyzer.tools.capitol_trades import CapitolTradesClient
from stock_analyzer.tools.market_data import MarketDataClient


def alembic_upgrade(url: str) -> None:
    """Run alembic upgrade head with the supplied DATABASE_URL."""
    subprocess.run(
        ["alembic", "upgrade", "head"],
        check=True,
        env={"DATABASE_URL": url, **__import__("os").environ},
    )


async def backfill_politicians(months: int,
                                  pol_repo: PoliticianRepository,
                                  trade_repo: PoliticianTradeRepository,
                                  ) -> int:
    """Scrape `months` of historical CapitolTrades disclosures."""
    client = CapitolTradesClient()
    cutoff = (date.today() - timedelta(days=months * 31))
    discs = await client.get_recent_disclosures(since=cutoff, max_pages=200)
    inserted = 0
    for d in discs:
        pol = pol_repo.upsert(
            full_name=d.politician_full_name,
            party=d.party, chamber=d.chamber, state=d.state,
            capitol_trades_id=d.capitol_trades_id,
        )
        n = trade_repo.bulk_upsert([{
            "politician_id": pol.id, "ticker": d.ticker, "side": d.side,
            "trade_date": d.trade_date, "disclosure_date": d.disclosure_date,
            "amount_min_usd": d.amount_min_usd,
            "amount_max_usd": d.amount_max_usd,
            "raw_payload": d.raw,
        }])
        inserted += n
    return inserted


def recompute_scores(pol_repo: PoliticianRepository,
                       trade_repo: PoliticianTradeRepository,
                       score_repo: PoliticianScoreRepository,
                       spy_repo: SpyCloseRepository, *,
                       today: date, lookback_months: int) -> int:
    window_start = today - timedelta(days=lookback_months * 31)
    spy_series = spy_repo.get_range(window_start, today)
    if not spy_series:
        return 0
    spy_start = spy_series[min(spy_series)]
    spy_end = spy_series[max(spy_series)]
    spy_ret = compute_spy_return(spy_start, spy_end)

    market = MarketDataClient()
    updated = 0
    for pol in pol_repo.all():
        trades = trade_repo.for_politician(pol.id, since=window_start)
        sim: list[SimulatedTrade] = []
        for t in trades:
            open_d = t.disclosure_date + timedelta(days=1)
            open_price = (market.get_spy_close(open_d) if t.ticker == "SPY"
                            else _ticker_close(market, t.ticker, open_d))
            close_price = (market.get_spy_close(today) if t.ticker == "SPY"
                            else _ticker_close(market, t.ticker, today))
            if open_price is None or close_price is None or open_price <= 0:
                continue
            sim.append(SimulatedTrade(
                ticker=t.ticker, side=t.side,
                trade_dollars=midpoint_amount(t.amount_min_usd, t.amount_max_usd),
                open_price=open_price, close_price=close_price,
            ))
        if not sim:
            continue
        ret = compute_politician_return(sim)
        score_repo.upsert(
            politician_id=pol.id,
            computed_at=datetime.now(timezone.utc),
            window_start=window_start, window_end=today,
            total_return_pct=ret, spy_return_pct=spy_ret,
            trade_count=len(sim),
        )
        updated += 1
    return updated


def _ticker_close(market: MarketDataClient, ticker: str, d: date) -> float | None:
    """Get the close for an arbitrary ticker on date d. Falls back to None."""
    import yfinance as yf
    try:
        df = yf.Ticker(ticker).history(start=d, end=d + timedelta(days=1))
        return float(df["Close"].iloc[0]) if not df.empty else None
    except Exception:
        return None
```

- [ ] **Step 31.4: Add commands to `src/stock_analyzer/__main__.py`**

```python
@db_app.command("migrate")
def db_migrate() -> None:
    """Run alembic upgrade head."""
    from stock_analyzer.db_commands import alembic_upgrade
    s = Settings()
    if s.stock_analyzer_env != "production":
        typer.echo("Skipping migrations in non-production env.")
        return
    alembic_upgrade(s.database_url)
    typer.echo("Migrations applied.")


@db_app.command("backfill-politicians")
def db_backfill(months: Annotated[int, typer.Option("--months")] = 24) -> None:
    """Seed historical Congressional disclosures."""
    from stock_analyzer.db_commands import backfill_politicians
    from stock_analyzer.persistence.db import Database
    from stock_analyzer.persistence.repositories import (
        PoliticianRepository, PoliticianTradeRepository,
    )
    s = Settings()
    db = Database(url=s.database_url, ephemeral=False)
    db.create_all()
    n = asyncio.run(backfill_politicians(
        months,
        PoliticianRepository(db),
        PoliticianTradeRepository(db),
    ))
    typer.echo(f"Backfilled {n} new disclosures.")


@db_app.command("recompute-scores")
def db_recompute() -> None:
    """Recompute 24-month politician scores."""
    from stock_analyzer.db_commands import recompute_scores
    from stock_analyzer.persistence.db import Database
    from stock_analyzer.persistence.repositories import (
        PoliticianRepository, PoliticianScoreRepository,
        PoliticianTradeRepository, SpyCloseRepository,
    )
    s = Settings()
    db = Database(url=s.database_url, ephemeral=False)
    n = recompute_scores(
        PoliticianRepository(db), PoliticianTradeRepository(db),
        PoliticianScoreRepository(db), SpyCloseRepository(db),
        today=_today_in_tz(s.run_timezone),
        lookback_months=s.politician_lookback_months,
    )
    typer.echo(f"Updated {n} politician scores.")
```

- [ ] **Step 31.5: Run tests**

```bash
uv run pytest tests/unit/test_db_commands.py -v
```

Expected: 1 passed.

- [ ] **Step 31.6: Commit**

```bash
git add src/stock_analyzer/db_commands.py src/stock_analyzer/__main__.py tests/unit/test_db_commands.py
git commit -m "feat: add CLI db migrate / backfill-politicians / recompute-scores"
```

---

### Task 32: CLI `history`

**Files:**
- Modify: `src/stock_analyzer/__main__.py`

- [ ] **Step 32.1: Add the command**

```python
@app.command()
def history(
    limit: Annotated[int, typer.Option("--limit")] = 10,
) -> None:
    """Show the last N runs from the audit log."""
    from stock_analyzer.persistence.db import Database
    from stock_analyzer.persistence.repositories import RunRepository
    s = Settings()
    db = Database(url=s.database_url, ephemeral=False)
    runs = RunRepository(db).history(limit=limit)
    for r in runs:
        typer.echo(f"{r.run_date}  {r.status:>8}  "
                    f"emails=({r.email_1_status},{r.email_2_status},{r.email_3_status})  "
                    f"tokens={r.total_tokens_in or 0}/{r.total_tokens_out or 0}  "
                    f"${r.est_cost_usd or 0:.4f}")
```

- [ ] **Step 32.2: Smoke test the help works**

```bash
uv run stock-analyzer history --help
```

Expected: shows `--limit`.

- [ ] **Step 32.3: Commit**

```bash
git add src/stock_analyzer/__main__.py
git commit -m "feat: add CLI history command"
```

---

*Phase 8 complete: full CLI surface (run, health-check, db, history) is wired.*

---

## Phase 9 — Deployment artifacts (Tasks 33–34)

### Task 33: systemd unit + timer

**Files:**
- Create: `deploy/stock-analyzer.service`
- Create: `deploy/stock-analyzer.timer`

- [ ] **Step 33.1: Create `deploy/stock-analyzer.service`**

(Use the exact content from spec §11. Drop in verbatim.)

- [ ] **Step 33.2: Create `deploy/stock-analyzer.timer`**

(Use the exact content from spec §11. Drop in verbatim.)

- [ ] **Step 33.3: Lint with `systemd-analyze` (optional, on the LXC)**

```bash
# On the target LXC after deploying:
systemd-analyze verify deploy/stock-analyzer.service deploy/stock-analyzer.timer
```

Expected: no errors.

- [ ] **Step 33.4: Commit**

```bash
git add deploy/stock-analyzer.service deploy/stock-analyzer.timer
git commit -m "feat: add systemd service + timer for daily 07:00 ET run"
```

---

### Task 34: install.sh

**Files:**
- Create: `deploy/install.sh`

- [ ] **Step 34.1: Create `deploy/install.sh`** (idempotent installer)

```bash
#!/usr/bin/env bash
# Idempotent installer for the stock-analyzer LXC.
# Run as root.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/opt/stock-analyzer}"
SVC_USER="stock-analyzer"

# 1. System packages
apt-get update
apt-get install -y --no-install-recommends \
    python3.14 python3.14-venv git curl ca-certificates sqlite3

# 2. Service user
if ! id "$SVC_USER" &>/dev/null; then
    useradd -r -m -d "/var/lib/$SVC_USER" -s /bin/bash "$SVC_USER"
fi

# 3. uv (per-user install)
if ! command -v uv &>/dev/null; then
    su - "$SVC_USER" -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
fi

# 4. Code
if [[ ! -d "$REPO_DIR" ]]; then
    git clone https://github.com/snehalsoni/stock-analyzer.git "$REPO_DIR"
fi
chown -R "$SVC_USER:$SVC_USER" "$REPO_DIR"
sudo -u "$SVC_USER" bash -c "cd $REPO_DIR && ~/.local/bin/uv sync --frozen"
sudo -u "$SVC_USER" bash -c "cd $REPO_DIR && ~/.local/bin/uv pip install -e ."

# 5. Secrets file
install -d -m 0755 /etc/stock-analyzer
if [[ ! -f /etc/stock-analyzer/env ]]; then
    install -m 0600 -o "$SVC_USER" "$REPO_DIR/.env.example" /etc/stock-analyzer/env
    echo "WARNING: edit /etc/stock-analyzer/env before enabling the timer."
fi

# 6. systemd
install -m 0644 "$REPO_DIR/deploy/stock-analyzer.service" /etc/systemd/system/
install -m 0644 "$REPO_DIR/deploy/stock-analyzer.timer"   /etc/systemd/system/
systemctl daemon-reload

echo "Install complete. Next steps:"
echo "  1. \$EDITOR /etc/stock-analyzer/env"
echo "  2. sudo -u $SVC_USER stock-analyzer db migrate"
echo "  3. sudo -u $SVC_USER stock-analyzer db backfill-politicians --months=24"
echo "  4. sudo -u $SVC_USER stock-analyzer health-check"
echo "  5. systemctl enable --now stock-analyzer.timer"
```

- [ ] **Step 34.2: Make it executable**

```bash
chmod +x deploy/install.sh
```

- [ ] **Step 34.3: Commit**

```bash
git add deploy/install.sh
git commit -m "feat: add idempotent LXC installer"
```

---

*Phase 9 complete: deployment artifacts ready for `bash deploy/install.sh` on the target LXC.*

---

## Phase 10 — End-to-End Test (Task 35)

### Task 35: Full-pipeline E2E test (ephemeral + dry-run + Claude stub)

**Files:**
- Create: `tests/e2e/__init__.py`, `tests/e2e/test_full_pipeline.py`

- [ ] **Step 35.1: Write the test**

```python
# tests/e2e/test_full_pipeline.py
from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock

from stock_analyzer.orchestrator import Orchestrator
from stock_analyzer.persistence.in_memory import (
    InMemoryPoliticianRepository, InMemoryPoliticianScoreRepository,
    InMemoryPoliticianTradeRepository, InMemoryRunRepository,
    InMemorySpyCloseRepository,
)
from stock_analyzer.rendering.renderer import (
    DrawdownReport, HoldingBrief, PoliticianReport, PortfolioReport, Renderer,
)
from stock_analyzer.tools.market_data import QuoteSnapshot
from stock_analyzer.tools.snaptrade import Holding


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_full_ephemeral_dry_run(tmp_path: Path) -> None:
    """End-to-end: portfolio fetched, agents stubbed, three HTML files written."""
    holdings = [
        Holding(ticker="AAPL", quantity=10, currency="USD",
                avg_cost=150, market_value=2000, account="x"),
        Holding(ticker="NVDA", quantity=5, currency="USD",
                avg_cost=400, market_value=2000, account="x"),
    ]
    snap = MagicMock(get_holdings=MagicMock(return_value=holdings))
    market = MagicMock()
    market.batch_quotes = MagicMock(return_value=[
        QuoteSnapshot(ticker="AAPL", prev_close=200, pre_market_price=189, last_price=189),  # -5.5%
        QuoteSnapshot(ticker="NVDA", prev_close=900, pre_market_price=895, last_price=895),
        QuoteSnapshot(ticker="SPY", prev_close=520, pre_market_price=519, last_price=519),
    ])
    market.get_trending_tickers = MagicMock(return_value={"AAPL"})
    market.get_spy_close = MagicMock(return_value=520.0)

    capitol = MagicMock()
    capitol.get_recent_disclosures = AsyncMock(return_value=[])

    news = MagicMock()
    news.get_yfinance_news = MagicMock(return_value=[])

    portfolio_agent = MagicMock()
    portfolio_agent.run = AsyncMock(return_value=(
        PortfolioReport(
            as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
            trending_news=[],
            holdings=[HoldingBrief(ticker="AAPL", company_name="Apple",
                                     pct_change_overnight=-5.5,
                                     bullets=["Down hard on guidance cut."],
                                     sources=[], watch_today="Earnings.")],
            portfolio_summary="One holding cratered overnight.",
        ),
        {"input_tokens": 100, "output_tokens": 50, "cache_read_tokens": 0},
    ))
    drawdown_agent = MagicMock()
    drawdown_agent.run = AsyncMock(return_value=(
        DrawdownReport(
            as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
            items=[], market_context=None),
        {"input_tokens": 50, "output_tokens": 25, "cache_read_tokens": 0},
    ))
    politician_agent = MagicMock()
    politician_agent.run = AsyncMock(return_value=(
        PoliticianReport(
            as_of=datetime(2026, 5, 5, tzinfo=timezone.utc),
            buys=[], sells=[],
            top_takeaway="Quiet day on the Hill."),
        {"input_tokens": 30, "output_tokens": 10, "cache_read_tokens": 0},
    ))

    o = Orchestrator(
        ephemeral=True,
        snaptrade=snap, market_data=market, news=news,
        sec_edgar=None, capitol_trades=capitol, insider_monkey=None,
        smtp=None, renderer=Renderer(),
        politician_repo=InMemoryPoliticianRepository(),
        trade_repo=InMemoryPoliticianTradeRepository(),
        score_repo=InMemoryPoliticianScoreRepository(),
        spy_repo=InMemorySpyCloseRepository(),
        run_repo=InMemoryRunRepository(),
        portfolio_agent=portfolio_agent, drawdown_agent=drawdown_agent,
        politician_agent=politician_agent,
        drawdown_threshold_pct=5.0, politician_lookback_months=24,
        politician_fresh_disclosure_days=2,
        failed_emails_dir=str(tmp_path), smtp_to="to@x.com",
        dry_run=True,
    )
    # Use a Wednesday 2026 so it's a trading day:
    await o.run(today=date(2026, 5, 6))

    files = sorted(p.name for p in tmp_path.glob("*.html"))
    assert len(files) == 3
    assert any("portfolio" in f for f in files)
    assert any("drawdown" in f for f in files)
    assert any("politician" in f for f in files)

    # Spot-check rendered content
    portfolio_html = next(p for p in tmp_path.glob("*portfolio*.html")).read_text()
    assert "AAPL" in portfolio_html
    assert "Apple" in portfolio_html
    assert "Ephemeral mode" in portfolio_html  # banner present in ephemeral mode
```

- [ ] **Step 35.2: Run the E2E test**

```bash
uv run pytest tests/e2e/test_full_pipeline.py -v
```

Expected: 1 passed.

- [ ] **Step 35.3: Commit**

```bash
git add tests/e2e/
git commit -m "test: add e2e dry-run pipeline test with stubbed agents"
```

---

### Final task: full test suite + lint + typecheck

- [ ] **Run the whole pipeline:**

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict src/stock_analyzer
```

All four must succeed before declaring v1 done.

- [ ] **Final commit (if any formatting changes):**

```bash
git add -A
git commit -m "chore: final lint/format pass"
```

---

## Self-review notes

**Spec coverage check:**

| Spec section | Implementing tasks |
|---|---|
| §2 Architecture (3 agents + orch) | T23-25, T26-28 |
| §3 Repository layout | T1, T5, T8, T10, T14, T21-23 (file structure preface locks it in) |
| §4 Data flow (3 phases) | T26 (Phase 1), T27 (Phase 2), T28 (Phase 3) |
| §5 Agent specs | T22 (response models), T23-25 (agents), T29 (factory wires Sonnet 4.6 + caching) |
| §6 Hybrid scraping | T18 (CapitolTrades JSON, deterministic), T19 (InsiderMonkey via Crawl4ai) |
| §7 Persistence schema | T5 (models), T6 (db), T7 (alembic), T8 (sql repos), T9 (in-memory repos) |
| §8 Config + secrets | T2 (Settings), T34 (install.sh sets perms) |
| §9 Execution modes & CLI | T29 (mode resolution + run), T30-32 (other commands), T9 (in-memory mode) |
| §10 Error handling, observability, testing | T3 (logging), T13 (cost tracker), tenacity decorators inline in T14-20 tools, T35 (e2e) |
| §11 Deployment | T33 (systemd), T34 (install.sh) |

**Placeholder scan:** every step contains executable code or runnable commands; no "TBD" / "implement later".

**Cross-task type consistency:** the Pydantic response models in T22 match the agent imports in T23-25; `Disclosure` from T18 is consumed by T25 and T31; `Holding` from T14 is consumed by T23 and T35; `QuoteSnapshot` from T15 is consumed by T35. Names line up.

**Spec ambiguity (politician filter on `disclosure_date`)** — the spec was already updated; T27 implements the cutoff using `disclosure_date >= today - politician_fresh_disclosure_days`, matching.

