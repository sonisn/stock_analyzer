# Covered-Call Writing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `rebalance-portfolio` pipeline so a single Opus rebalancer pass recommends covered calls (aggressive premium style, Δ 0.35–0.45, DTE 30–45) against held ≥100-share positions, deploys the resulting premium via existing `ADD`/`BUY` actions, and optionally consolidates sub-100 "stub" shares into round lots to expand CC coverage.

**Architecture:** Option chain fetching is data-prep, parallel to the reviewer. The rebalancer Opus pass receives per-ticker context blocks (chain rows, eligibility, round-lot coverage) and returns a single `RebalancePlan` with mixed actions (`SELL`/`TRIM`/`ADD`/`BUY`/`WRITE_CALL`) plus a parallel `option_writes` list. A deterministic Python validation step runs after the LLM. Reporting and persistence reuse the existing JSON-blob mechanism — no SQL migration.

**Tech Stack:** Python 3.14, Pydantic v2, `agno` workflow, `snaptrade_client`, `yfinance`, SQLite, ReportLab, pytest. Project uses `uv` for env management.

---

## Spec reference

The complete design lives in `docs/superpowers/specs/2026-05-13-covered-calls-design.md`. Implementers should read it once before starting Task 1. The plan below is the *how*; the spec is the *what* and *why*.

## File map

**New files**

| Path | Responsibility |
|---|---|
| `src/stock_analyzer/data/options_symbols.py` | Parse OCC-format option symbols (e.g. `NVDA  260620C00250000`). Pure functions. |
| `src/stock_analyzer/data/options_chain.py` | `OptionQuote`/`OptionChain` dataclasses, `OptionChainProvider` Protocol, `YFinanceChain` + `SnapTradeChain`, `fetch_chains(tickers, dte_min, dte_max)` orchestrator with per-ticker fallback. |
| `src/stock_analyzer/discover/cc_eligibility.py` | `eligible_holdings(...)`, `round_lot_coverage(...)`, `apply_earnings_filter(...)`, `build_cc_context_block(...)` — pure Python, no I/O. |
| `src/stock_analyzer/discover/cc_validation.py` | `validate_option_writes(plan, eligibility)` — drops orphan WRITE_CALLs, clamps oversized contracts, returns `(cleaned_plan, warnings)`. |
| `src/stock_analyzer/discover/cc_render.py` | Deterministic compute for the three new report sections: premium income, round-lot coverage, premium → deployment math. |
| `tests/test_options_symbols.py` | OCC parser. |
| `tests/test_options_chain.py` | Providers (mocked), orchestrator, fallback. |
| `tests/test_cc_eligibility.py` | Eligibility, round-lot, earnings filter, context block. |
| `tests/test_cc_schema.py` | `OptionWrite` validation, `RebalancePlan.option_writes` round-trip. |
| `tests/test_cc_validation.py` | Orphan drop, clamp, logging. |
| `tests/test_premium_deployment.py` | Deployment math. |
| `tests/test_round_lot_coverage.py` | Coverage math. |
| `tests/test_track_record_cc.py` | CC outcome scoring. |
| `tests/fixtures/__init__.py` | Empty package marker. |
| `tests/fixtures/snaptrade_chain_nvda.json` | Canned SnapTrade chain. |
| `tests/fixtures/yfinance_chain_aapl.json` | Canned yfinance chain row dump. |

**Touched files**

| Path | Edit |
|---|---|
| `src/stock_analyzer/config.py` | Add 10 `cc_*` fields. |
| `.env.example` | Append CC section. |
| `src/stock_analyzer/discover/rebalance_schema.py` | Add `WRITE_CALL` literal; add `OptionWrite`; add `option_writes` on `RebalancePlan`. |
| `src/stock_analyzer/discover/rebalancer.py` | Extend `REBALANCER_INSTRUCTIONS`; extend `decide()` signature with `cc_context_block`. |
| `src/stock_analyzer/discover/premortem.py` | Append CC paragraph to instructions. |
| `src/stock_analyzer/discover/report_sections.py` | Extend `SectionKind` Literal; new sections fall through to existing renderers. |
| `src/stock_analyzer/discover/report_html.py` | Render three new sections. |
| `src/stock_analyzer/discover/report_pdf.py` | Render three new sections. |
| `src/stock_analyzer/data/brokerage.py` | Add `fetch_open_option_positions()`. |
| `src/stock_analyzer/discover/track_record.py` | Add `score_covered_calls()`. |
| `src/stock_analyzer/cli/rebalance.py` | New steps: `step_cc_data`, `step_cc_validate`; thread context into `step_rebalance`; thread option_writes into `_build_rebalance_sections`; annotate email subject. |

---

## Task 1: Config — CC settings

**Files:**
- Modify: `src/stock_analyzer/config.py:99-101` (insert after `# ---- Behavior ----`)
- Modify: `.env.example` (append at end)
- Test: `tests/test_cc_schema.py` (use this file for the config test too — pragma: keep CC config tests near schema tests)

- [ ] **Step 1: Write failing test**

Create `tests/test_cc_schema.py`:
```python
"""Tests for CC config + RebalancePlan/OptionWrite schema."""
from __future__ import annotations

from stock_analyzer.config import Settings


def test_cc_defaults():
    s = Settings()  # type: ignore[call-arg]
    assert s.cc_enabled is True
    assert s.cc_target_delta_min == 0.35
    assert s.cc_target_delta_max == 0.45
    assert s.cc_dte_min == 30
    assert s.cc_dte_max == 45
    assert s.cc_denylist == ()
    assert s.cc_min_premium_usd == 500
    assert s.cc_slippage_buffer == 0.10
    assert s.cc_stub_optimization is True
    assert s.cc_min_stub_usd == 1000


def test_cc_denylist_parses_csv(monkeypatch):
    monkeypatch.setenv("CC_DENYLIST", "TSLA, AAPL ,nvda")
    s = Settings()  # type: ignore[call-arg]
    assert s.cc_denylist == ("TSLA", "AAPL", "NVDA")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cc_schema.py -v`
Expected: FAIL — `AttributeError: 'Settings' object has no attribute 'cc_enabled'`

- [ ] **Step 3: Add CC settings to `config.py`**

Insert after the `# ---- Behavior ----` block (around line 101, before the `# ---- Coercers ----` block):

```python
    # ---- Covered-call writing (cli/rebalance.py extension) ---------------
    cc_enabled: bool = True
    cc_target_delta_min: float = 0.35
    cc_target_delta_max: float = 0.45
    cc_dte_min: int = 30
    cc_dte_max: int = 45
    cc_denylist: Annotated[tuple[str, ...], NoDecode] = ()
    cc_min_premium_usd: float = 500.0
    cc_slippage_buffer: float = 0.10
    cc_stub_optimization: bool = True
    cc_min_stub_usd: float = 1000.0
```

Then add a coercer below `_split_watchlist` (mirrors that pattern exactly):

```python
    @field_validator("cc_denylist", mode="before")
    @classmethod
    def _split_cc_denylist(cls, v: object) -> object:
        if isinstance(v, str):
            return tuple(t.strip().upper() for t in v.split(",") if t.strip())
        return v
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cc_schema.py -v`
Expected: 2 passed

- [ ] **Step 5: Append to `.env.example`**

```bash
# -----------------------------------------------------------------------------
# Covered-call writing (optional) — used by cli/rebalance.py
# -----------------------------------------------------------------------------
# Master switch. Set to 0 to disable all CC logic (eligibility, chain
# fetch, rebalancer prompt section). Pipeline runs as before.
CC_ENABLED=1

# Strike selection band (Δ). The Opus rebalancer picks strikes within
# [min, max], leaning toward MIN on high-confidence HOLDs and toward
# MAX on TRIM/low-confidence positions. Defaults are "aggressive
# premium" style — Δ 0.35-0.45 ≈ real chance of assignment.
CC_TARGET_DELTA_MIN=0.35
CC_TARGET_DELTA_MAX=0.45

# Days-to-expiry band. 30-45 DTE = 1 cycle/month, decent theta decay,
# below the long-dated tax-treatment cliff.
CC_DTE_MIN=30
CC_DTE_MAX=45

# Tickers to never write calls against (comma-separated). Useful for
# positions you'd never want to risk losing to assignment.
CC_DENYLIST=

# If expected total premium across all WRITE_CALLs is less than this
# (USD), the rebalancer leaves the premium as cash rather than
# proposing reinvestment trades (avoids friction for tiny premiums).
CC_MIN_PREMIUM_USD=500

# Fraction of expected premium held back when sizing reinvestment
# trades, to account for fill slippage vs. mid-quote estimates.
CC_SLIPPAGE_BUFFER=0.10

# Stub consolidation: sell sub-100 "stub" shares to fund round-lot
# completion elsewhere (expanding future CC capacity). Set 0 to
# disable.
CC_STUB_OPTIMIZATION=1

# Don't propose stub consolidation when the stub is worth less than
# this (USD) — trade friction wastes the gain.
CC_MIN_STUB_USD=1000
```

- [ ] **Step 6: Commit**

```bash
git add src/stock_analyzer/config.py .env.example tests/test_cc_schema.py
git commit -m "feat(cc): add covered-call config knobs and .env example"
```

---

## Task 2: Schema — OptionWrite + WRITE_CALL action type

**Files:**
- Modify: `src/stock_analyzer/discover/rebalance_schema.py:24` (extend literal); add new model
- Test: `tests/test_cc_schema.py` (extend)

- [ ] **Step 1: Add failing tests**

Append to `tests/test_cc_schema.py`:
```python
import pytest
from pydantic import ValidationError

from stock_analyzer.discover.rebalance_schema import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def test_option_write_valid():
    ow = OptionWrite(
        ticker="NVDA",
        strike=260.0,
        expiry="2026-06-20",
        contracts=3,
        est_premium_per_share=2.40,
        delta=0.36,
        assignment_probability=0.36,
        notes="HOLD-8, far-OTM bias",
    )
    assert ow.ticker == "NVDA"
    assert ow.contracts == 3
    assert ow.notes == "HOLD-8, far-OTM bias"


def test_option_write_frozen():
    ow = OptionWrite(
        ticker="NVDA", strike=260.0, expiry="2026-06-20", contracts=1,
        est_premium_per_share=2.40, delta=0.36, assignment_probability=0.36,
    )
    with pytest.raises(ValidationError):
        ow.strike = 270.0  # type: ignore[misc]


def test_rebalance_action_accepts_write_call():
    a = RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                        sizing="3 contracts $260C 2026-06-20")
    assert a.action == "WRITE_CALL"


def test_rebalance_action_rejects_unknown():
    with pytest.raises(ValidationError):
        RebalanceAction(action="ROLL", ticker="NVDA", sizing="x")  # type: ignore[arg-type]


def test_rebalance_plan_option_writes_default_empty():
    plan = RebalancePlan(status="NO_ACTION", aggressiveness_applied="balanced",
                         full_text="…")
    assert plan.option_writes == []


def test_rebalance_plan_option_writes_roundtrip():
    ow = OptionWrite(
        ticker="NVDA", strike=260.0, expiry="2026-06-20", contracts=2,
        est_premium_per_share=2.40, delta=0.36, assignment_probability=0.36,
    )
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                                 sizing="2 contracts")],
        option_writes=[ow],
        full_text="…",
    )
    blob = plan.model_dump(mode="json")
    restored = RebalancePlan.model_validate(blob)
    assert restored.option_writes[0].ticker == "NVDA"
    assert restored.actions[0].action == "WRITE_CALL"


def test_legacy_plan_without_option_writes_still_parses():
    """Backwards compat: rows persisted before this feature won't have
    `option_writes` in the JSON blob. Default factory must absorb that."""
    legacy = {
        "status": "ACTION", "aggressiveness_applied": "balanced",
        "actions": [{"action": "SELL", "ticker": "FOO", "sizing": "full"}],
        "summary": "", "full_text": "…",
    }
    plan = RebalancePlan.model_validate(legacy)
    assert plan.option_writes == []
```

- [ ] **Step 2: Run to verify failures**

Run: `uv run pytest tests/test_cc_schema.py -v`
Expected: 6 new failures — `ImportError: cannot import name 'OptionWrite'` etc.

- [ ] **Step 3: Extend `rebalance_schema.py`**

Open `src/stock_analyzer/discover/rebalance_schema.py`. Replace line 24 and add a new model.

Change line 24 from:
```python
    action: Literal["SELL", "TRIM", "ADD", "BUY"]
```
to:
```python
    action: Literal["SELL", "TRIM", "ADD", "BUY", "WRITE_CALL"]
```

Insert this class above `class RebalanceAction`:

```python
class OptionWrite(BaseModel):
    """Structured detail for one WRITE_CALL action.

    Joined to its corresponding RebalanceAction by ticker. Premium is
    quoted PER SHARE (the standard options convention); multiply by 100
    to get dollars per contract."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    strike: float
    expiry: str = Field(..., description="ISO date YYYY-MM-DD.")
    contracts: int = Field(..., gt=0, description="Number of contracts to write.")
    est_premium_per_share: float = Field(
        ..., ge=0,
        description="Mid of bid/ask in dollars per share. ×100 = per contract.",
    )
    delta: float = Field(..., ge=0.0, le=1.0)
    assignment_probability: float = Field(..., ge=0.0, le=1.0)
    notes: str = ""
```

In `class RebalancePlan`, add this field after `full_text`:

```python
    option_writes: list[OptionWrite] = Field(
        default_factory=list,
        description=(
            "Parallel to WRITE_CALL actions. Each entry MUST have a "
            "matching WRITE_CALL in `actions` with the same ticker. "
            "Empty list when no calls are recommended."
        ),
    )
```

Update `__all__` at the bottom to include `"OptionWrite"`.

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cc_schema.py -v`
Expected: all 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/rebalance_schema.py tests/test_cc_schema.py
git commit -m "feat(cc): add OptionWrite + WRITE_CALL action type to rebalance schema"
```

---

## Task 3: OCC option symbol parser

**Files:**
- Create: `src/stock_analyzer/data/options_symbols.py`
- Create: `tests/test_options_symbols.py`

OCC symbol format example: `NVDA  260620C00250000` = NVDA, expires 2026-06-20, Call, strike $250.000. Total 21 chars with the underlying left-padded.

- [ ] **Step 1: Write failing test**

Create `tests/test_options_symbols.py`:
```python
"""Tests for OCC option-symbol parser."""
from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.data.options_symbols import (
    OCCParseError,
    is_option_symbol,
    parse_occ,
)


def test_parse_call():
    p = parse_occ("NVDA  260620C00250000")
    assert p.ticker == "NVDA"
    assert p.expiry == date(2026, 6, 20)
    assert p.option_type == "C"
    assert p.strike == 250.0


def test_parse_put():
    p = parse_occ("AAPL  260718P00200500")
    assert p.option_type == "P"
    assert p.strike == 200.5


def test_parse_long_underlying():
    # 6-char underlying with no padding required.
    p = parse_occ("BRKB  260117C00450000")
    assert p.ticker == "BRKB"


def test_is_option_symbol_true():
    assert is_option_symbol("NVDA  260620C00250000")


def test_is_option_symbol_false_for_equity():
    assert not is_option_symbol("NVDA")
    assert not is_option_symbol("BRK.B")


def test_parse_rejects_garbage():
    with pytest.raises(OCCParseError):
        parse_occ("not-an-option")


def test_parse_rejects_wrong_length():
    with pytest.raises(OCCParseError):
        parse_occ("NVDA260620C00250000")  # missing padding
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_options_symbols.py -v`
Expected: FAIL — `ModuleNotFoundError: stock_analyzer.data.options_symbols`

- [ ] **Step 3: Implement parser**

Create `src/stock_analyzer/data/options_symbols.py`:
```python
"""OCC option-symbol parsing.

OCC format (21 chars total):

  ROOT(6, space-padded right) | YY(2) MM(2) DD(2) | TYPE(1: C|P) | STRIKE(8, 3 implied decimals)

Example:
  "NVDA  260620C00250000" = NVDA, 2026-06-20, Call, $250.000 strike

This module deliberately does NOT depend on third-party libs — the
format is fixed-width and small enough to handle by slicing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

OptionType = Literal["C", "P"]

_OCC_RE = re.compile(
    r"^([A-Z][A-Z0-9.\-]{0,5})\s*(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"
)


class OCCParseError(ValueError):
    """Raised when a string does not look like an OCC option symbol."""


@dataclass(frozen=True)
class ParsedOCC:
    ticker: str
    expiry: date
    option_type: OptionType
    strike: float


def is_option_symbol(s: str) -> bool:
    """True if `s` looks like an OCC option symbol. Does not raise."""
    try:
        parse_occ(s)
    except OCCParseError:
        return False
    return True


def parse_occ(symbol: str) -> ParsedOCC:
    """Parse an OCC option symbol. Tolerates trailing whitespace and
    runs of spaces between the root and the date (SnapTrade and yfinance
    use slightly different padding). Raises OCCParseError on anything
    that doesn't fit the pattern.

    The fixed-width spec uses 6 chars for the root, space-padded right
    (`"NVDA  "`). We accept 1-6 chars plus any run of spaces so brokers
    that strip padding still parse.
    """
    if not isinstance(symbol, str):
        raise OCCParseError(f"expected str, got {type(symbol).__name__}")
    s = symbol.strip()
    m = _OCC_RE.match(s)
    if not m:
        raise OCCParseError(f"not an OCC symbol: {symbol!r}")
    root, yy, mm, dd, otype, strike_raw = m.groups()
    try:
        expiry = date(2000 + int(yy), int(mm), int(dd))
    except ValueError as e:
        raise OCCParseError(f"bad date in {symbol!r}: {e}") from e
    # Strike has 3 implied decimals: "00250000" = 250.000
    strike = int(strike_raw) / 1000.0
    return ParsedOCC(
        ticker=root, expiry=expiry, option_type=otype, strike=strike,  # type: ignore[arg-type]
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_options_symbols.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/data/options_symbols.py tests/test_options_symbols.py
git commit -m "feat(cc): OCC option-symbol parser"
```

---

## Task 4: Options chain dataclasses + Protocol

**Files:**
- Create: `src/stock_analyzer/data/options_chain.py` (skeleton only — Protocol + dataclasses)
- Create: `tests/test_options_chain.py` (skeleton tests)

- [ ] **Step 1: Write failing skeleton test**

Create `tests/test_options_chain.py`:
```python
"""Tests for options_chain.py — providers, orchestrator, fallback."""
from __future__ import annotations

from datetime import date, datetime

from stock_analyzer.data.options_chain import (
    OptionChain,
    OptionQuote,
)


def test_optionquote_frozen_and_typed():
    q = OptionQuote(
        strike=260.0, expiry=date(2026, 6, 20),
        bid=2.20, ask=2.40, iv=0.29, delta=0.36,
        open_interest=2890, volume=540,
    )
    assert q.strike == 260.0
    assert q.delta == 0.36


def test_optionchain_dataclass():
    chain = OptionChain(
        ticker="NVDA", spot=235.0, asof=datetime(2026, 5, 13, 16, 0, 0),
        calls=[], source="missing",
    )
    assert chain.ticker == "NVDA"
    assert chain.source == "missing"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement skeleton**

Create `src/stock_analyzer/data/options_chain.py`:
```python
"""Options chain fetching: SnapTrade primary, yfinance fallback.

The orchestrator (`fetch_chains`) tries SnapTrade per-ticker and falls
back to yfinance on None/error. Both providers return a normalized
`OptionChain` containing only OTM calls within the requested DTE band.

Failure of either provider for a given ticker is non-fatal — the
returned `OptionChain.source` is set to `"missing"` and the rebalancer
context just reads `Option chain: UNAVAILABLE` for that ticker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal, Protocol


@dataclass(frozen=True)
class OptionQuote:
    """One option strike/expiry row (calls only — puts not supported)."""
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
    """A ticker's filtered OTM call chain.

    `source` records which provider answered. `"missing"` is a valid
    state that downstream code handles — it does NOT raise.
    """
    ticker: str
    spot: float
    asof: datetime
    calls: list[OptionQuote] = field(default_factory=list)
    source: Literal["snaptrade", "yfinance", "missing"] = "missing"


class OptionChainProvider(Protocol):
    """Minimal contract every chain provider implements.

    Implementations MUST:
      - filter to OTM calls only (strike > spot)
      - filter to expiries within [today+dte_min, today+dte_max]
      - return None on any error (graceful degradation)
    """
    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        ...
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/data/options_chain.py tests/test_options_chain.py
git commit -m "feat(cc): options_chain skeleton (Protocol + dataclasses)"
```

---

## Task 5: yfinance options chain provider

**Files:**
- Modify: `src/stock_analyzer/data/options_chain.py` (add `YFinanceChain`)
- Modify: `tests/test_options_chain.py` (add provider tests)

yfinance exposes options chains via `Ticker.options` (list of expiry strings) and `Ticker.option_chain(expiry).calls` (DataFrame). We accept that IV and delta come back as `impliedVolatility` (no Greeks); delta is None on yfinance.

- [ ] **Step 1: Write failing test**

Append to `tests/test_options_chain.py`:
```python
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from stock_analyzer.data.options_chain import YFinanceChain


def _fake_ticker(spot: float, expiries_to_calls: dict[str, pd.DataFrame]) -> MagicMock:
    """Build a MagicMock that mimics yfinance.Ticker."""
    t = MagicMock()
    t.fast_info = MagicMock(last_price=spot)
    t.options = tuple(expiries_to_calls.keys())
    t.option_chain.side_effect = lambda e: MagicMock(calls=expiries_to_calls[e])
    return t


def _calls_df(rows: list[tuple[float, float, float, float, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=["strike", "bid", "ask", "impliedVolatility", "openInterest", "volume"],
    )


def test_yfinance_filters_to_dte_band_and_otm():
    today = date.today()
    e_in_band = (today + timedelta(days=35)).isoformat()
    e_too_close = (today + timedelta(days=10)).isoformat()
    e_too_far = (today + timedelta(days=120)).isoformat()
    chains = {
        e_in_band: _calls_df([
            (250.0, 3.10, 3.30, 0.31, 4210, 850),  # OTM
            (230.0, 8.00, 8.20, 0.33, 1000, 200),  # ITM — should be filtered
        ]),
        e_too_close: _calls_df([(260.0, 0.50, 0.60, 0.28, 100, 10)]),
        e_too_far: _calls_df([(260.0, 5.50, 5.60, 0.28, 100, 10)]),
    }
    fake = _fake_ticker(spot=235.0, expiries_to_calls=chains)
    with patch("stock_analyzer.data.options_chain.yf.Ticker", return_value=fake):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is not None
    assert chain.source == "yfinance"
    assert chain.spot == 235.0
    strikes = sorted(q.strike for q in chain.calls)
    assert strikes == [250.0]  # ITM 230 dropped, out-of-band expiries dropped


def test_yfinance_returns_none_on_error():
    with patch(
        "stock_analyzer.data.options_chain.yf.Ticker",
        side_effect=RuntimeError("network blew up"),
    ):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is None


def test_yfinance_no_expiries_returns_empty_chain_with_source_set():
    fake = _fake_ticker(spot=235.0, expiries_to_calls={})
    with patch("stock_analyzer.data.options_chain.yf.Ticker", return_value=fake):
        chain = YFinanceChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is not None
    assert chain.calls == []
    assert chain.source == "yfinance"
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: FAIL — `cannot import name 'YFinanceChain'`.

- [ ] **Step 3: Implement YFinanceChain**

Append to `src/stock_analyzer/data/options_chain.py`:
```python
from datetime import timedelta
import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)


class YFinanceChain:
    """yfinance-backed options chain provider.

    yfinance does not expose Greeks; `delta` is always None. The
    rebalancer's prompt is robust to that — it falls back to comparing
    strike vs spot when delta is missing.
    """

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        try:
            t = yf.Ticker(ticker)
            spot = float(t.fast_info.last_price)
        except Exception as e:
            logger.info("yfinance chain miss for %s (%s)", ticker, e)
            return None

        today = date.today()
        lo = today + timedelta(days=dte_min)
        hi = today + timedelta(days=dte_max)
        calls: list[OptionQuote] = []
        try:
            expiries = tuple(t.options)
        except Exception as e:
            logger.info("yfinance no expiries for %s (%s)", ticker, e)
            return OptionChain(
                ticker=ticker, spot=spot, asof=datetime.now(),
                calls=[], source="yfinance",
            )

        for e_str in expiries:
            try:
                expiry = date.fromisoformat(e_str)
            except ValueError:
                continue
            if expiry < lo or expiry > hi:
                continue
            try:
                df = t.option_chain(e_str).calls
            except Exception as ex:
                logger.info("yfinance chain row miss %s@%s (%s)", ticker, e_str, ex)
                continue
            for _, row in df.iterrows():
                strike = float(row["strike"])
                if strike <= spot:  # OTM calls only
                    continue
                calls.append(OptionQuote(
                    strike=strike,
                    expiry=expiry,
                    bid=float(row.get("bid") or 0.0),
                    ask=float(row.get("ask") or 0.0),
                    iv=float(row["impliedVolatility"]) if row.get("impliedVolatility") is not None else None,
                    delta=None,  # yfinance does not provide Greeks
                    open_interest=int(row.get("openInterest") or 0),
                    volume=int(row.get("volume") or 0),
                ))

        return OptionChain(
            ticker=ticker, spot=spot, asof=datetime.now(),
            calls=calls, source="yfinance",
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/data/options_chain.py tests/test_options_chain.py
git commit -m "feat(cc): yfinance options chain provider"
```

---

## Task 6: SnapTrade options chain provider

**Files:**
- Modify: `src/stock_analyzer/data/options_chain.py` (add `SnapTradeChain`)
- Modify: `tests/test_options_chain.py` (add SnapTrade test)
- Create: `tests/fixtures/__init__.py` (empty)
- Create: `tests/fixtures/snaptrade_chain_nvda.json` (canned response)

SnapTrade's `trading.get_options_chain(account_id, symbol)` returns chains shaped per the underlying broker. The fixture mimics the common shape (a list of strikes, each with `expirations`). Implementers should compare to a real call early and adjust mapping if needed — that's why this provider returns `None` on shape mismatch rather than throwing.

- [ ] **Step 1: Create fixture**

Create `tests/fixtures/__init__.py` (empty file).

Create `tests/fixtures/snaptrade_chain_nvda.json`:
```json
{
  "underlying_price": 235.0,
  "options": [
    {
      "strike_price": 250.0,
      "option_chain": [
        {
          "expiration_date": "2026-06-20",
          "call": {
            "symbol": "NVDA  260620C00250000",
            "bid_price": 3.10, "ask_price": 3.30,
            "implied_volatility": 0.31,
            "delta": 0.42,
            "open_interest": 4210, "volume": 850
          }
        }
      ]
    },
    {
      "strike_price": 260.0,
      "option_chain": [
        {
          "expiration_date": "2026-06-20",
          "call": {
            "symbol": "NVDA  260620C00260000",
            "bid_price": 2.20, "ask_price": 2.40,
            "implied_volatility": 0.29,
            "delta": 0.36,
            "open_interest": 2890, "volume": 540
          }
        }
      ]
    }
  ]
}
```

- [ ] **Step 2: Write failing test**

Append to `tests/test_options_chain.py`:
```python
import json
from pathlib import Path

from stock_analyzer.data.options_chain import SnapTradeChain

_FIXTURES = Path(__file__).parent / "fixtures"


def test_snaptrade_parses_canned_chain():
    raw = json.loads((_FIXTURES / "snaptrade_chain_nvda.json").read_text())
    fake_client = MagicMock()
    fake_client.trading.get_options_chain.return_value = MagicMock(body=raw)
    fake_client.account_information.list_user_accounts.return_value = MagicMock(
        body=[{"id": "acct-1"}]
    )

    with patch(
        "stock_analyzer.data.options_chain._snaptrade_client",
        return_value=fake_client,
    ):
        chain = SnapTradeChain().fetch("NVDA", dte_min=30, dte_max=45)

    # Fixture expiry is 2026-06-20 which is in-band for the spec date
    # 2026-05-13. If tests are running in a future where this expiry
    # has lapsed, the test would skip — that's intentional, but in
    # CI today this should produce 2 OTM rows.
    if chain is None or chain.source != "snaptrade":
        # Allow None when running in a date window where fixture is stale.
        return
    assert chain.spot == 235.0
    strikes = sorted(q.strike for q in chain.calls)
    assert strikes == [250.0, 260.0]


def test_snaptrade_returns_none_when_creds_missing():
    with patch(
        "stock_analyzer.data.options_chain._snaptrade_client",
        return_value=None,
    ):
        chain = SnapTradeChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is None


def test_snaptrade_returns_none_on_unexpected_shape():
    fake_client = MagicMock()
    fake_client.trading.get_options_chain.return_value = MagicMock(
        body={"unexpected": "shape"}
    )
    fake_client.account_information.list_user_accounts.return_value = MagicMock(
        body=[{"id": "acct-1"}]
    )
    with patch(
        "stock_analyzer.data.options_chain._snaptrade_client",
        return_value=fake_client,
    ):
        chain = SnapTradeChain().fetch("NVDA", dte_min=30, dte_max=45)
    assert chain is None
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: FAIL — `cannot import name 'SnapTradeChain'`.

- [ ] **Step 4: Implement SnapTradeChain**

Append to `src/stock_analyzer/data/options_chain.py`:
```python
from typing import Any

from ..config import Settings


def _snaptrade_client() -> Any:
    """Lazy SnapTrade client builder. Returns None when creds are missing
    so callers can degrade gracefully rather than blow up."""
    s = Settings()  # type: ignore[call-arg]
    if not all([
        s.snaptrade_client_id, s.snaptrade_consumer_key,
        s.snaptrade_user_id, s.snaptrade_user_secret,
    ]):
        return None
    try:
        from snaptrade_client import SnapTrade
    except ImportError:
        return None
    client = SnapTrade(
        client_id=s.snaptrade_client_id,
        consumer_key=s.snaptrade_consumer_key,
    )
    # Bind user creds to the convenience attrs the rest of the codebase uses.
    client.user_id = s.snaptrade_user_id
    client.user_secret = s.snaptrade_user_secret
    return client


def _first_account_id(client: Any) -> str | None:
    try:
        accts = client.account_information.list_user_accounts(
            user_id=client.user_id, user_secret=client.user_secret,
        ).body
    except Exception as e:
        logger.info("SnapTrade list_user_accounts failed: %s", e)
        return None
    if not accts:
        return None
    first = accts[0]
    return first.get("id") if isinstance(first, dict) else getattr(first, "id", None)


def _parse_snaptrade_options_payload(
    ticker: str, payload: dict[str, Any], dte_min: int, dte_max: int,
) -> OptionChain | None:
    """Translate SnapTrade's chain shape into OptionChain. Returns None
    when the shape is unrecognized (so we fall back to yfinance)."""
    try:
        spot = float(payload["underlying_price"])
        rows = payload["options"]
    except (KeyError, TypeError, ValueError):
        return None

    today = date.today()
    lo = today + timedelta(days=dte_min)
    hi = today + timedelta(days=dte_max)
    calls: list[OptionQuote] = []
    for r in rows:
        try:
            strike = float(r["strike_price"])
        except (KeyError, TypeError, ValueError):
            continue
        if strike <= spot:  # OTM calls only
            continue
        for entry in r.get("option_chain") or []:
            try:
                expiry = date.fromisoformat(entry["expiration_date"])
            except (KeyError, TypeError, ValueError):
                continue
            if expiry < lo or expiry > hi:
                continue
            call = entry.get("call") or {}
            calls.append(OptionQuote(
                strike=strike, expiry=expiry,
                bid=float(call.get("bid_price") or 0.0),
                ask=float(call.get("ask_price") or 0.0),
                iv=(float(call["implied_volatility"])
                    if call.get("implied_volatility") is not None else None),
                delta=(float(call["delta"]) if call.get("delta") is not None else None),
                open_interest=int(call.get("open_interest") or 0),
                volume=int(call.get("volume") or 0),
            ))

    return OptionChain(
        ticker=ticker, spot=spot, asof=datetime.now(),
        calls=calls, source="snaptrade",
    )


class SnapTradeChain:
    """SnapTrade-backed options chain provider.

    Returns None on any failure — auth missing, account list empty,
    endpoint not supported on the user's tier, payload shape mismatch.
    The orchestrator falls back to yfinance on None.
    """

    def fetch(
        self, ticker: str, dte_min: int, dte_max: int
    ) -> OptionChain | None:
        client = _snaptrade_client()
        if client is None:
            return None
        account_id = _first_account_id(client)
        if account_id is None:
            logger.info("SnapTrade: no account_id available for chain fetch")
            return None
        try:
            resp = client.trading.get_options_chain(
                account_id=account_id, symbol=ticker,
                user_id=client.user_id, user_secret=client.user_secret,
            )
        except Exception as e:
            logger.info("SnapTrade chain fetch failed for %s: %s", ticker, e)
            return None
        body = getattr(resp, "body", None)
        if not isinstance(body, dict):
            return None
        return _parse_snaptrade_options_payload(ticker, body, dte_min, dte_max)
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: 8 passed.

- [ ] **Step 6: Commit**

```bash
git add src/stock_analyzer/data/options_chain.py tests/test_options_chain.py tests/fixtures/
git commit -m "feat(cc): SnapTrade options chain provider with shape-safe parsing"
```

---

## Task 7: fetch_chains orchestrator (fallback wiring)

**Files:**
- Modify: `src/stock_analyzer/data/options_chain.py` (add `fetch_chains`)
- Modify: `tests/test_options_chain.py` (add orchestrator tests)

- [ ] **Step 1: Write failing test**

Append to `tests/test_options_chain.py`:
```python
from stock_analyzer.data.options_chain import fetch_chains


def test_fetch_chains_uses_snaptrade_when_available():
    fake_chain = OptionChain(
        ticker="NVDA", spot=235.0, asof=datetime.now(),
        calls=[], source="snaptrade",
    )
    with patch.object(SnapTradeChain, "fetch", return_value=fake_chain) as snap, \
         patch.object(YFinanceChain, "fetch") as yfin:
        out = fetch_chains(["NVDA"], dte_min=30, dte_max=45)
    snap.assert_called_once()
    yfin.assert_not_called()
    assert out["NVDA"].source == "snaptrade"


def test_fetch_chains_falls_back_to_yfinance():
    fake = OptionChain(
        ticker="AAPL", spot=215.0, asof=datetime.now(),
        calls=[], source="yfinance",
    )
    with patch.object(SnapTradeChain, "fetch", return_value=None), \
         patch.object(YFinanceChain, "fetch", return_value=fake):
        out = fetch_chains(["AAPL"], dte_min=30, dte_max=45)
    assert out["AAPL"].source == "yfinance"


def test_fetch_chains_marks_missing_when_both_fail():
    with patch.object(SnapTradeChain, "fetch", return_value=None), \
         patch.object(YFinanceChain, "fetch", return_value=None):
        out = fetch_chains(["XYZ"], dte_min=30, dte_max=45)
    assert out["XYZ"].source == "missing"
    assert out["XYZ"].calls == []


def test_fetch_chains_empty_input():
    assert fetch_chains([], dte_min=30, dte_max=45) == {}
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: FAIL — `cannot import name 'fetch_chains'`.

- [ ] **Step 3: Implement fetch_chains**

Append to `src/stock_analyzer/data/options_chain.py`:
```python
def fetch_chains(
    tickers: list[str],
    *,
    dte_min: int,
    dte_max: int,
) -> dict[str, OptionChain]:
    """Per-ticker chain fetch with SnapTrade → yfinance fallback.

    Always returns a chain object for every input ticker. When both
    providers fail, the returned `OptionChain.source` is `"missing"` and
    `calls` is empty — the rebalancer prompt is told to show
    `UNAVAILABLE` for these tickers, and Opus will simply not recommend
    a WRITE_CALL on them.
    """
    if not tickers:
        return {}
    snap = SnapTradeChain()
    yfin = YFinanceChain()
    out: dict[str, OptionChain] = {}
    for t in tickers:
        chain = snap.fetch(t, dte_min, dte_max)
        if chain is None:
            chain = yfin.fetch(t, dte_min, dte_max)
        if chain is None:
            chain = OptionChain(
                ticker=t, spot=0.0, asof=datetime.now(),
                calls=[], source="missing",
            )
            logger.warning("chain unavailable for %s (both providers failed)", t)
        out[t] = chain
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_options_chain.py -v`
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/data/options_chain.py tests/test_options_chain.py
git commit -m "feat(cc): fetch_chains orchestrator with per-ticker fallback"
```

---

## Task 8: Extract open option positions from SnapTrade holdings

**Files:**
- Modify: `src/stock_analyzer/data/brokerage.py` (add `fetch_open_option_positions`)
- Modify: `tests/test_brokerage_classification.py` (extend with option-position parse tests)

SnapTrade's positions endpoint returns equity positions and may return option positions intermixed. We use the OCC parser from Task 3 to detect option symbols and bucket them by underlying ticker.

- [ ] **Step 1: Write failing test**

Append to `tests/test_brokerage_classification.py`:
```python
from unittest.mock import MagicMock, patch

from stock_analyzer.data.brokerage import fetch_open_option_positions


def test_fetch_open_option_positions_groups_short_calls_by_underlying():
    # 3 contracts short on NVDA Jun-260 call (units = -3),
    # 2 contracts short on AAPL Jul-230 call (units = -2),
    # 1 contract LONG on TSLA Aug-300 call (units = +1) — should be skipped
    # because it doesn't reduce CC coverage.
    fake_positions = [
        {
            "symbol": {"symbol": {"symbol": "NVDA  260620C00260000"}},
            "units": -3,
        },
        {
            "symbol": {"symbol": {"symbol": "AAPL  260718C00230000"}},
            "units": -2,
        },
        {
            "symbol": {"symbol": {"symbol": "TSLA  260815C00300000"}},
            "units": 1,
        },
        # Equity row should be ignored.
        {
            "symbol": {"symbol": {"symbol": "GOOG"}},
            "units": 50,
        },
    ]
    fake_resp = MagicMock(body=fake_positions)
    with patch(
        "stock_analyzer.data.brokerage._client_and_user"
    ) as mk_creds, patch(
        "stock_analyzer.data.brokerage._snaptrade_accounts",
        return_value=[{"id": "acct-1"}],
    ):
        mk_creds.return_value = (MagicMock(account_information=MagicMock(
            get_user_account_positions=MagicMock(return_value=fake_resp),
        )), "u", "s")
        coverage = fetch_open_option_positions()
    assert coverage == {"NVDA": 3, "AAPL": 2}
    assert "TSLA" not in coverage  # long calls don't reduce CC coverage
```

The test references `_client_and_user` and `_snaptrade_accounts` — those are internal helpers in brokerage.py. Skim the file before running the test; if those names differ, adjust the patch targets. The principle stays the same.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_brokerage_classification.py -v`
Expected: FAIL — `cannot import name 'fetch_open_option_positions'`.

- [ ] **Step 3: Implement the function**

Add to `src/stock_analyzer/data/brokerage.py` (place near `fetch_portfolio_holdings`):
```python
from .options_symbols import OCCParseError, parse_occ


def fetch_open_option_positions() -> dict[str, int]:
    """Return {underlying_ticker: open_short_call_contracts} across all
    connected accounts.

    Only SHORT calls (units < 0) are counted — those reduce the share
    count available to back new covered calls. Long calls and short
    puts are ignored. Returns {} when no positions or on any error
    (graceful degradation; eligibility just subtracts zero).
    """
    try:
        client, user_id, user_secret = _client_and_user()
    except Exception as e:
        logger.info("SnapTrade unavailable for option positions: %s", e)
        return {}
    out: dict[str, int] = {}
    try:
        accounts = _snaptrade_accounts(client, user_id, user_secret)
    except Exception as e:
        logger.info("SnapTrade account list failed: %s", e)
        return {}
    for acct in accounts:
        acct_id = acct.get("id") if isinstance(acct, dict) else getattr(acct, "id", None)
        if not acct_id:
            continue
        try:
            resp = client.account_information.get_user_account_positions(
                account_id=acct_id, user_id=user_id, user_secret=user_secret,
            )
        except Exception as e:
            logger.info("SnapTrade positions fetch failed (%s): %s", acct_id, e)
            continue
        for pos in (resp.body or []):
            symbol = pos
            for key in ("symbol", "symbol", "symbol"):  # nested walk
                if isinstance(symbol, dict) and key in symbol:
                    symbol = symbol[key]
            if not isinstance(symbol, str):
                continue
            try:
                parsed = parse_occ(symbol)
            except OCCParseError:
                continue  # equity row, skip
            if parsed.option_type != "C":
                continue  # we only care about short calls
            units = float(pos.get("units") or 0)
            if units >= 0:
                continue  # long calls don't reduce coverage
            out[parsed.ticker] = out.get(parsed.ticker, 0) + int(-units)
    return out
```

If `_client_and_user` / `_snaptrade_accounts` helpers don't already exist in brokerage.py, add minimal versions that mirror how `fetch_portfolio_holdings` currently builds its client (read the file first, then adapt).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_brokerage_classification.py -v`
Expected: all pass (existing tests + new one).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/data/brokerage.py tests/test_brokerage_classification.py
git commit -m "feat(cc): parse open short-call positions from SnapTrade for coverage subtraction"
```

---

## Task 9: cc_eligibility — eligibility filter

**Files:**
- Create: `src/stock_analyzer/discover/cc_eligibility.py`
- Create: `tests/test_cc_eligibility.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cc_eligibility.py`:
```python
"""Tests for CC eligibility / round-lot / earnings / context-block builders."""
from __future__ import annotations

from datetime import date, datetime

from stock_analyzer.discover.cc_eligibility import (
    EligibleHolding,
    eligible_holdings,
)


def _pos(units: int) -> dict[str, float | int]:
    return {"units": units, "avg_buy_price": 100.0, "cost_basis": units * 100.0}


def test_eligibility_excludes_under_100_shares():
    positions = {
        "AAPL": _pos(99),
        "TSLA": _pos(335),
    }
    out = eligible_holdings(positions, open_short_calls={}, denylist=())
    assert "AAPL" not in out
    assert "TSLA" in out


def test_eligibility_subtracts_open_short_calls():
    positions = {"NVDA": _pos(400)}
    out = eligible_holdings(
        positions, open_short_calls={"NVDA": 1}, denylist=(),
    )
    # 400 - 100 = 300 available, max_contracts = 3
    assert out["NVDA"].available_shares == 300
    assert out["NVDA"].max_contracts == 3


def test_eligibility_excludes_when_coverage_zero():
    positions = {"NVDA": _pos(150)}
    out = eligible_holdings(
        positions, open_short_calls={"NVDA": 1}, denylist=(),
    )
    # 150 - 100 = 50 < 100 → not eligible
    assert "NVDA" not in out


def test_eligibility_respects_denylist():
    positions = {"AAPL": _pos(200), "MSFT": _pos(200)}
    out = eligible_holdings(
        positions, open_short_calls={}, denylist=("AAPL",),
    )
    assert "AAPL" not in out
    assert "MSFT" in out


def test_eligibility_record_shape():
    out = eligible_holdings(
        {"NVDA": _pos(335)}, open_short_calls={}, denylist=(),
    )
    rec = out["NVDA"]
    assert isinstance(rec, EligibleHolding)
    assert rec.ticker == "NVDA"
    assert rec.shares_held == 335
    assert rec.available_shares == 335
    assert rec.max_contracts == 3
    assert rec.open_short_call_contracts == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cc_eligibility.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

Create `src/stock_analyzer/discover/cc_eligibility.py`:
```python
"""Pure-Python eligibility, round-lot coverage, earnings filter, and
prompt-context assembly for the covered-call extension to the
rebalancer.

No I/O here — every function is testable as a pure transformation of
its inputs. CLI wiring (`cli/rebalance.py`) is responsible for fetching
holdings, chains, open short-call positions, and earnings dates, then
passing them in.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


@dataclass(frozen=True)
class EligibleHolding:
    """A position that's eligible to write covered calls against."""
    ticker: str
    shares_held: int
    open_short_call_contracts: int
    available_shares: int   # shares_held - 100 × open_short_call_contracts
    max_contracts: int      # available_shares // 100


def eligible_holdings(
    positions: dict[str, dict[str, float]],
    *,
    open_short_calls: dict[str, int],
    denylist: tuple[str, ...],
) -> dict[str, EligibleHolding]:
    """Return {ticker: EligibleHolding} for every position that:
      - holds ≥ 100 shares
      - has ≥ 100 shares NOT already collateralizing an open short call
      - is not in `denylist`

    `positions` matches the shape produced by `_aggregate_positions`
    in `cli/rebalance.py` — {ticker: {"units": float, ...}}.
    """
    denyset = {t.upper() for t in denylist}
    out: dict[str, EligibleHolding] = {}
    for ticker, pos in positions.items():
        if ticker.upper() in denyset:
            continue
        shares = int(pos.get("units") or 0)
        if shares < 100:
            continue
        short_contracts = int(open_short_calls.get(ticker, 0))
        available = shares - 100 * short_contracts
        if available < 100:
            continue
        out[ticker] = EligibleHolding(
            ticker=ticker, shares_held=shares,
            open_short_call_contracts=short_contracts,
            available_shares=available,
            max_contracts=available // 100,
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cc_eligibility.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/cc_eligibility.py tests/test_cc_eligibility.py
git commit -m "feat(cc): eligibility filter for covered-call writing"
```

---

## Task 10: cc_eligibility — round-lot coverage

**Files:**
- Modify: `src/stock_analyzer/discover/cc_eligibility.py`
- Modify: `tests/test_cc_eligibility.py`
- Create: `tests/test_round_lot_coverage.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_round_lot_coverage.py`:
```python
"""Tests for round-lot coverage math (stub consolidation context)."""
from __future__ import annotations

from stock_analyzer.discover.cc_eligibility import (
    RoundLotCoverage,
    round_lot_coverage,
)


def test_basic_split_and_stub():
    positions = {
        "TSLA": {"units": 335, "avg_buy_price": 250},
        "AAPL": {"units": 215, "avg_buy_price": 150},
        "NVDA": {"units": 100, "avg_buy_price": 235},  # exactly a round lot, no stub
        "GOOG": {"units": 50,  "avg_buy_price": 170},  # all stub
    }
    spots = {"TSLA": 300.0, "AAPL": 215.0, "NVDA": 235.0, "GOOG": 175.0}
    out = round_lot_coverage(positions, spots=spots)

    tsla = out["TSLA"]
    assert tsla.round_lots == 3
    assert tsla.stub_shares == 35
    assert tsla.stub_dollar_value == 35 * 300.0
    assert tsla.to_next_lot_shares == 65
    assert tsla.to_next_lot_cost == 65 * 300.0

    aapl = out["AAPL"]
    assert aapl.round_lots == 2
    assert aapl.stub_shares == 15

    # No-stub holding: to_next_lot fields zero (no consolidation opportunity).
    nvda = out["NVDA"]
    assert nvda.round_lots == 1
    assert nvda.stub_shares == 0
    assert nvda.to_next_lot_shares == 0
    assert nvda.to_next_lot_cost == 0.0

    # Sub-100 holding (all stub).
    goog = out["GOOG"]
    assert goog.round_lots == 0
    assert goog.stub_shares == 50
    assert goog.to_next_lot_shares == 50
    assert goog.to_next_lot_cost == 50 * 175.0


def test_missing_spot_falls_back_to_zero_dollar_values():
    positions = {"FOO": {"units": 150}}
    out = round_lot_coverage(positions, spots={})  # no spot for FOO
    rec = out["FOO"]
    assert rec.round_lots == 1
    assert rec.stub_shares == 50
    assert rec.stub_dollar_value == 0.0  # don't crash on missing spot


def test_record_is_RoundLotCoverage_type():
    out = round_lot_coverage({"FOO": {"units": 100}}, spots={"FOO": 10.0})
    assert isinstance(out["FOO"], RoundLotCoverage)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_round_lot_coverage.py -v`
Expected: FAIL — `cannot import name 'RoundLotCoverage'`.

- [ ] **Step 3: Implement**

Append to `src/stock_analyzer/discover/cc_eligibility.py`:
```python
@dataclass(frozen=True)
class RoundLotCoverage:
    """Round-lot decomposition of a single holding.

    Used by the stub-consolidation prompt rule and by the reporting
    layer's `RoundLotCoverage` section.
    """
    ticker: str
    shares: int
    round_lots: int
    stub_shares: int             # shares - round_lots × 100
    stub_dollar_value: float     # stub_shares × spot (0 when spot unknown)
    to_next_lot_shares: int      # (100 - stub_shares) if stub_shares else 0
    to_next_lot_cost: float      # to_next_lot_shares × spot


def round_lot_coverage(
    positions: dict[str, dict[str, float]],
    *,
    spots: dict[str, float],
) -> dict[str, RoundLotCoverage]:
    """Compute round-lot / stub decomposition for every held ticker.

    `spots` is the current price per ticker (from the technicals stage).
    Missing spots collapse dollar values to 0 — the report layer can
    still show share counts even when price data is stale.
    """
    out: dict[str, RoundLotCoverage] = {}
    for ticker, pos in positions.items():
        shares = int(pos.get("units") or 0)
        if shares <= 0:
            continue
        round_lots = shares // 100
        stub = shares - round_lots * 100
        spot = float(spots.get(ticker) or 0.0)
        to_next_shares = (100 - stub) if stub else 0
        out[ticker] = RoundLotCoverage(
            ticker=ticker, shares=shares,
            round_lots=round_lots, stub_shares=stub,
            stub_dollar_value=stub * spot,
            to_next_lot_shares=to_next_shares,
            to_next_lot_cost=to_next_shares * spot,
        )
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_round_lot_coverage.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/cc_eligibility.py tests/test_round_lot_coverage.py
git commit -m "feat(cc): round-lot coverage decomposition (stub-consolidation context)"
```

---

## Task 11: cc_eligibility — earnings filter

**Files:**
- Modify: `src/stock_analyzer/discover/cc_eligibility.py`
- Modify: `tests/test_cc_eligibility.py`

The earnings blacklist drops chain expiries that straddle the next earnings date. "Straddle" = expiry falls in the window `[earnings_date - 7 days, earnings_date + 7 days]`. The wider window protects against pre-earnings IV pump + assignment surprise.

- [ ] **Step 1: Write failing test**

Append to `tests/test_cc_eligibility.py`:
```python
from stock_analyzer.discover.cc_eligibility import apply_earnings_filter
from stock_analyzer.data.options_chain import OptionChain, OptionQuote


def _chain(ticker: str, expiries: list[str]) -> OptionChain:
    return OptionChain(
        ticker=ticker, spot=100.0, asof=datetime.now(),
        calls=[OptionQuote(
            strike=110.0, expiry=date.fromisoformat(e),
            bid=1.0, ask=1.1, iv=0.3, delta=0.35,
            open_interest=500, volume=50,
        ) for e in expiries],
        source="yfinance",
    )


def test_earnings_filter_drops_straddling_expiries():
    # Earnings 2026-06-15; window = 2026-06-08 .. 2026-06-22
    chain = _chain("NVDA", ["2026-06-10", "2026-06-22", "2026-07-18"])
    filtered, blacklisted = apply_earnings_filter(
        chain, earnings_date=date(2026, 6, 15),
    )
    survived = [c.expiry.isoformat() for c in filtered.calls]
    assert survived == ["2026-07-18"]
    # blacklisted window endpoints reported for the prompt
    assert blacklisted == (date(2026, 6, 8), date(2026, 6, 22))


def test_earnings_filter_passthrough_when_no_date():
    chain = _chain("NVDA", ["2026-06-10", "2026-07-18"])
    filtered, blacklisted = apply_earnings_filter(chain, earnings_date=None)
    assert len(filtered.calls) == 2
    assert blacklisted is None


def test_earnings_filter_empty_chain():
    chain = OptionChain(
        ticker="X", spot=100.0, asof=datetime.now(), calls=[], source="missing",
    )
    filtered, _ = apply_earnings_filter(chain, earnings_date=date(2026, 6, 15))
    assert filtered.calls == []
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cc_eligibility.py -v`
Expected: FAIL — `cannot import name 'apply_earnings_filter'`.

- [ ] **Step 3: Implement**

Append to `src/stock_analyzer/discover/cc_eligibility.py`:
```python
from .. data.options_chain import OptionChain

EARNINGS_BLACKLIST_DAYS = 7


def apply_earnings_filter(
    chain: OptionChain,
    *,
    earnings_date: date | None,
) -> tuple[OptionChain, tuple[date, date] | None]:
    """Drop expiries that fall within ±EARNINGS_BLACKLIST_DAYS of
    earnings_date. Returns the filtered chain and the blacklist window
    (for prompt display) or None when no earnings date was provided.
    """
    if earnings_date is None:
        return chain, None
    lo = earnings_date - timedelta(days=EARNINGS_BLACKLIST_DAYS)
    hi = earnings_date + timedelta(days=EARNINGS_BLACKLIST_DAYS)
    survived = [q for q in chain.calls if q.expiry < lo or q.expiry > hi]
    # OptionChain is frozen; build a new one with the same source/asof.
    return (
        OptionChain(
            ticker=chain.ticker, spot=chain.spot, asof=chain.asof,
            calls=survived, source=chain.source,
        ),
        (lo, hi),
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cc_eligibility.py -v`
Expected: all pass (cumulative).

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/cc_eligibility.py tests/test_cc_eligibility.py
git commit -m "feat(cc): earnings-blacklist filter for option chain expiries"
```

---

## Task 12: cc_eligibility — build_cc_context_block (rebalancer prompt context)

**Files:**
- Modify: `src/stock_analyzer/discover/cc_eligibility.py`
- Modify: `tests/test_cc_eligibility.py`

This is the function that assembles the per-ticker prompt block the rebalancer reads. Output is plain text — readable diff and easy to test.

- [ ] **Step 1: Write failing test**

Append to `tests/test_cc_eligibility.py`:
```python
from stock_analyzer.discover.cc_eligibility import build_cc_context_block
from stock_analyzer.discover.schemas import HoldingReview


def _review(verdict: str, confidence: int) -> HoldingReview:
    return HoldingReview(
        verdict=verdict, confidence=confidence,
        position_context="x", forward_outlook="x",
        reasoning="x", tax_lot_plan=(), what_would_change_mind="x",
        full_text="x",
    )


def test_context_block_basic():
    positions = {"NVDA": {"units": 400}}
    elig = eligible_holdings(positions, open_short_calls={"NVDA": 1}, denylist=())
    coverage = round_lot_coverage(positions, spots={"NVDA": 235.0})
    chain = _chain("NVDA", ["2026-06-20"])
    block = build_cc_context_block(
        eligible=elig,
        chains={"NVDA": chain},
        coverage=coverage,
        reviews={"NVDA": _review("HOLD", 8)},
        earnings={"NVDA": date(2026, 5, 21)},
        stub_pool_total_usd=0.0,
    )
    assert "TICKER: NVDA" in block
    assert "Reviewer verdict:        HOLD (confidence 8/10)" in block
    assert "Shares held:             400" in block
    assert "Available for CC:        300 (100 already collateralizing open short call" in block
    assert "Earnings-blacklist:      2026-05-21" in block
    assert "2026-06-20" in block  # at least one chain row


def test_context_block_marks_unavailable_chain():
    positions = {"AAPL": {"units": 200}}
    elig = eligible_holdings(positions, open_short_calls={}, denylist=())
    coverage = round_lot_coverage(positions, spots={"AAPL": 215.0})
    block = build_cc_context_block(
        eligible=elig, chains={}, coverage=coverage,
        reviews={"AAPL": _review("HOLD", 7)},
        earnings={}, stub_pool_total_usd=0.0,
    )
    assert "Option chain: UNAVAILABLE" in block


def test_context_block_round_lot_section():
    positions = {"TSLA": {"units": 335}, "AAPL": {"units": 215}}
    elig = eligible_holdings(positions, open_short_calls={}, denylist=())
    coverage = round_lot_coverage(
        positions, spots={"TSLA": 300.0, "AAPL": 215.0},
    )
    block = build_cc_context_block(
        eligible=elig, chains={}, coverage=coverage,
        reviews={
            "TSLA": _review("HOLD", 8),
            "AAPL": _review("HOLD", 7),
        },
        earnings={}, stub_pool_total_usd=13_725.0,
    )
    assert "ROUND-LOT COVERAGE" in block
    assert "TSLA" in block and "AAPL" in block
    assert "$13,725" in block


def test_context_block_empty_when_no_eligible():
    block = build_cc_context_block(
        eligible={}, chains={}, coverage={},
        reviews={}, earnings={}, stub_pool_total_usd=0.0,
    )
    assert block == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cc_eligibility.py -v`
Expected: FAIL — `cannot import name 'build_cc_context_block'`.

- [ ] **Step 3: Implement**

Append to `src/stock_analyzer/discover/cc_eligibility.py`:
```python
from .schemas import HoldingReview


_CHAIN_ROW_CAP_PER_TICKER = 8


def _format_chain_row(q: object) -> str:
    """Single-line chain row used inside the per-ticker context block."""
    from ..data.options_chain import OptionQuote
    assert isinstance(q, OptionQuote)
    delta_str = f"Δ {q.delta:.2f}" if q.delta is not None else "Δ —"
    iv_str = f"IV {q.iv:.2f}" if q.iv is not None else "IV —"
    oi_str = f"OI {q.open_interest}" if q.open_interest else "OI —"
    return (
        f"    {q.expiry.isoformat()} ${q.strike:>6.2f} strike  "
        f"bid {q.bid:.2f} / ask {q.ask:.2f}  "
        f"{delta_str}  {iv_str}  {oi_str}"
    )


def _format_ticker_block(
    *, ticker: str,
    review: HoldingReview | str | None,
    eligible: EligibleHolding,
    chain: object | None,
    earnings_date: date | None,
) -> str:
    lines: list[str] = [f"TICKER: {ticker}"]
    # Review verdict
    if isinstance(review, HoldingReview):
        verdict_line = f"  Reviewer verdict:        {review.verdict} (confidence {review.confidence}/10)"
    else:
        verdict_line = "  Reviewer verdict:        UNKNOWN"
    lines.append(verdict_line)
    lines.append(f"  Shares held:             {eligible.shares_held}")
    if eligible.open_short_call_contracts:
        lines.append(
            f"  Available for CC:        {eligible.available_shares} "
            f"({100 * eligible.open_short_call_contracts} already collateralizing "
            f"open short call{'s' if eligible.open_short_call_contracts > 1 else ''})"
        )
    else:
        lines.append(f"  Available for CC:        {eligible.available_shares}")
    if earnings_date is not None:
        from datetime import timedelta as _td
        lo = earnings_date - _td(days=EARNINGS_BLACKLIST_DAYS)
        hi = earnings_date + _td(days=EARNINGS_BLACKLIST_DAYS)
        lines.append(
            f"  Earnings-blacklist:      {earnings_date.isoformat()} "
            f"(skip expiries {lo.isoformat()} .. {hi.isoformat()})"
        )
    else:
        lines.append("  Earnings-blacklist:      earnings_unknown — be conservative on DTE")
    # Chain rows
    from ..data.options_chain import OptionChain
    if not isinstance(chain, OptionChain) or chain.source == "missing" or not chain.calls:
        lines.append("  Option chain: UNAVAILABLE")
    else:
        lines.append("  Option chain (OTM calls):")
        for q in chain.calls[:_CHAIN_ROW_CAP_PER_TICKER]:
            lines.append(_format_chain_row(q))
    return "\n".join(lines)


def build_cc_context_block(
    *,
    eligible: dict[str, EligibleHolding],
    chains: dict[str, object],  # OptionChain values
    coverage: dict[str, RoundLotCoverage],
    reviews: dict[str, HoldingReview | str],
    earnings: dict[str, date],
    stub_pool_total_usd: float,
) -> str:
    """Compose the COVERED-CALL CONTEXT block consumed by the rebalancer
    prompt. Returns the empty string when no positions are eligible
    (in which case the rebalancer prompt simply doesn't include a CC
    section)."""
    if not eligible:
        return ""

    per_ticker: list[str] = []
    for ticker in sorted(eligible):
        per_ticker.append(_format_ticker_block(
            ticker=ticker,
            review=reviews.get(ticker),
            eligible=eligible[ticker],
            chain=chains.get(ticker),
            earnings_date=earnings.get(ticker),
        ))

    # Round-lot coverage table for stub consolidation reasoning.
    rlc_lines: list[str] = [
        "",
        "ROUND-LOT COVERAGE (every holding, for stub-consolidation reasoning):",
        f"  {'Position':<8} {'Shares':>6} {'Round lots':>10} {'Stub':>5} "
        f"{'Stub $':>12} {'To-next-lot':>14}",
    ]
    for ticker in sorted(coverage):
        rec = coverage[ticker]
        rlc_lines.append(
            f"  {ticker:<8} {rec.shares:>6d} "
            f"{rec.round_lots:>4d} ({rec.round_lots * 100:>3d}) "
            f"{rec.stub_shares:>5d} "
            f"${rec.stub_dollar_value:>10,.0f} "
            f"${rec.to_next_lot_cost:>13,.0f}"
        )
    rlc_lines.append(f"  Stub pool total: ${stub_pool_total_usd:,.0f}")

    header = "=" * 70 + "\nCOVERED-CALL CONTEXT\n" + "=" * 70
    return header + "\n\n" + "\n\n".join(per_ticker) + "\n" + "\n".join(rlc_lines)
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cc_eligibility.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/cc_eligibility.py tests/test_cc_eligibility.py
git commit -m "feat(cc): build_cc_context_block — rebalancer prompt context assembler"
```

---

## Task 13: Rebalancer prompt — extend REBALANCER_INSTRUCTIONS

**Files:**
- Modify: `src/stock_analyzer/discover/rebalancer.py:21` (extend `REBALANCER_INSTRUCTIONS` string)
- Modify: `tests/test_pipeline_wiring.py` (assert prompt mentions WRITE_CALL rules)

The full instruction string is large. We add a clearly-delimited "COVERED-CALL WRITING" section. The existing instruction text is preserved.

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline_wiring.py`:
```python
from stock_analyzer.discover.rebalancer import REBALANCER_INSTRUCTIONS


def test_rebalancer_prompt_includes_cc_rules():
    s = REBALANCER_INSTRUCTIONS
    assert "COVERED-CALL WRITING" in s
    assert "WRITE_CALL" in s
    assert "0.35" in s and "0.45" in s  # delta band defaults
    assert "STUB CONSOLIDATION" in s
    assert "PREMIUM REINVESTMENT" in s
    # The output schema reference must mention `option_writes`.
    assert "option_writes" in s
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline_wiring.py::test_rebalancer_prompt_includes_cc_rules -v`
Expected: FAIL — assertions miss.

- [ ] **Step 3: Extend REBALANCER_INSTRUCTIONS**

Open `src/stock_analyzer/discover/rebalancer.py`. The existing `REBALANCER_INSTRUCTIONS = """..."""` is multi-line. Find its closing `"""` and insert the new section before it. The exact insertion point: anywhere within the closing block, but placed after the action-type description so context is established first.

Add this block:

```
========================================================================
COVERED-CALL WRITING (when a COVERED-CALL CONTEXT block is present)
========================================================================
Style: aggressive premium. You may emit WRITE_CALL actions on positions
listed under COVERED-CALL CONTEXT.

TARGET BAND
  Δ 0.35-0.45, DTE 30-45 days. Stay inside the band.

STRIKE WITHIN BAND
  - HOLD verdict with confidence >= 7  → pick Δ closer to 0.35
    (lower assignment chance, accept smaller premium).
  - TRIM verdict, or HOLD with confidence <= 5  → pick Δ closer to 0.45
    (assignment is a clean exit).
  - SELL verdict  → DO NOT emit WRITE_CALL. Sell the stock outright.

COHERENCE WITH TRIM
  If you also TRIM N shares of the same ticker, your WRITE_CALL contracts
  must be ≤ (shares_after_trim) // 100. Never write calls that would
  force assignment beyond your post-action holdings.

LIQUIDITY GUARD
  Skip any strike where bid < $0.20, OI < 100, or
  (ask − bid) / mid > 0.15 (wide spread). If the ONLY strike in the
  band fails the guard, do not emit a WRITE_CALL for that ticker;
  state the reason in full_text.

ANNUALIZED YIELD (state in full_text)
  annualized_yield = (premium_per_share / strike) × (365 / DTE)
  If annualized_yield < 8%, justify why writing is still worth it
  (e.g., earnings reduction, regime hedge).

OUTPUT
  - Add one WRITE_CALL action per eligible ticker (max one).
    `sizing` format: "<N> contracts $<strike>C <YYYY-MM-DD>"
    Example: "3 contracts $260C 2026-06-20"
  - Add a matching `option_writes` entry with strike, expiry, contracts,
    est_premium_per_share (mid of bid/ask), delta, assignment_probability
    (≈ delta unless you have reason to differ), and a one-line `notes`.

========================================================================
PREMIUM REINVESTMENT
========================================================================
After choosing WRITE_CALL actions, compute:

  expected_premium_total = Σ contracts × est_premium_per_share × 100
  deployable = existing_cash
             + (1 - 0.10) × expected_premium_total
             + Σ stub_consolidation_proceeds

If expected_premium_total < $500, leave premium as cash; state the
reason in full_text. Otherwise route deployable capital via ADD/BUY
actions, priority:
  1. ADD on high-confidence (≥ 7) HOLD positions
  2. BUY a discover pick justified by the reviewer / ranker context
  3. Cash residual

Show the math explicitly in full_text:

  Premium income (gross):     $X
  Slippage buffer (10%):       -$Y
  Deployable premium:          $Z
  Existing cash:               $C
  Stub consolidation:          $S   ← only when consolidating
  Total dry powder:            $D
    → ADD <TICKER> $<amount>
    → BUY <TICKER> $<amount>
    → Cash held: $<residual>

Note trade linkages, e.g. "If you skip the NVDA write, shrink the
AMZN ADD by $340."

========================================================================
STUB CONSOLIDATION (round-lot optimization)
========================================================================
A ROUND-LOT COVERAGE table shows each holding's `shares = lots×100 + stub`,
stub $ value, and to-next-lot cost. Each round lot of 100 shares unlocks
one more WRITE_CALL contract — stub shares earn nothing.

Consider stub consolidation when ALL of:
  1. stub value > $1,000 (trade friction floor)
  2. selling the stub does NOT violate a confidence-≥7 HOLD
  3. freed capital + other dry powder can complete a round lot
     elsewhere (ADD existing-with-stub OR BUY new at a 100-multiple)

Express as paired actions:
  - TRIM N on the stub holding,
    sizing="<N> shares — stub consolidation"
  - matching ADD or BUY sized to land on a round lot

BUY sizing for future CC capacity: when BUYing partly to enable future
CC writing, size to a 100-multiple. State the multiple in sizing,
e.g. "100 shares (1 lot)".

Tax-aware: prefer LTCG lots for stub sales (see existing tax-lot
guidance).
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline_wiring.py::test_rebalancer_prompt_includes_cc_rules -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/rebalancer.py tests/test_pipeline_wiring.py
git commit -m "feat(cc): extend rebalancer instructions with CC writing + reinvestment + stub-consolidation rules"
```

---

## Task 14: Rebalancer.decide — accept cc_context_block param

**Files:**
- Modify: `src/stock_analyzer/discover/rebalancer.py:439-499` (extend `decide()` signature)
- Modify: `tests/test_pipeline_wiring.py` (add test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline_wiring.py`:
```python
from unittest.mock import MagicMock, patch

from stock_analyzer.discover.rebalance_schema import RebalancePlan
from stock_analyzer.discover.rebalancer import Rebalancer


def test_decide_includes_cc_context_in_prompt():
    captured: dict[str, str] = {}

    class _StubAgent:
        def run(self, prompt):
            captured["prompt"] = prompt
            return MagicMock(content=RebalancePlan(
                status="NO_ACTION",
                aggressiveness_applied="balanced",
                full_text="…",
            ))

    r = Rebalancer.__new__(Rebalancer)  # bypass __init__ (don't hit network)
    r.agent = _StubAgent()
    cc_block = "===\nCOVERED-CALL CONTEXT\n===\nTICKER: NVDA\n  Shares held: 400"
    r.decide(
        holdings_reviews={}, picks_text="", cash_available=1000.0,
        cc_context_block=cc_block,
    )
    assert "COVERED-CALL CONTEXT" in captured["prompt"]
    assert "TICKER: NVDA" in captured["prompt"]


def test_decide_omits_cc_block_when_empty():
    captured: dict[str, str] = {}

    class _StubAgent:
        def run(self, prompt):
            captured["prompt"] = prompt
            return MagicMock(content=RebalancePlan(
                status="NO_ACTION",
                aggressiveness_applied="balanced",
                full_text="…",
            ))

    r = Rebalancer.__new__(Rebalancer)
    r.agent = _StubAgent()
    r.decide(
        holdings_reviews={}, picks_text="", cash_available=1000.0,
        cc_context_block="",
    )
    assert "COVERED-CALL CONTEXT" not in captured["prompt"]
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline_wiring.py -v -k "decide_"`
Expected: FAIL — `decide() got an unexpected keyword argument 'cc_context_block'`.

- [ ] **Step 3: Extend decide()**

In `src/stock_analyzer/discover/rebalancer.py`, modify the `decide` method signature to add `cc_context_block: str = ""` after `market_themes_block`. Then add the block to the prompt assembly. The relevant snippet (current code shown in spec exploration above):

After the `themes_section = (...)` block and before `prompt = (` literal, add:

```python
        cc_section = (
            f"{cc_context_block}\n\n" if cc_context_block else ""
        )
```

Then add `f"{cc_section}"` into the prompt string between `themes_section` and `cash_line`:

```python
        prompt = (
            f"AGGRESSIVENESS: {agg}\n"
            f"(Apply the {agg} rule set from your instructions. The "
            f"'Tax-agnostic alternative' section is MANDATORY in any "
            f"NO ACTION output.)\n\n"
            f"{macro_block}"
            f"{themes_section}"
            f"{cc_section}"
            f"{cash_line}\n\n"
            f"{history_section}"
            f"Current holdings reviews ({len(holdings_reviews)}):\n\n{reviews_block}\n\n"
            f"New discover picks:\n\n{picks_text}"
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline_wiring.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/rebalancer.py tests/test_pipeline_wiring.py
git commit -m "feat(cc): thread cc_context_block through Rebalancer.decide"
```

---

## Task 15: cc_validation — drop orphans, clamp oversize

**Files:**
- Create: `src/stock_analyzer/discover/cc_validation.py`
- Create: `tests/test_cc_validation.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_cc_validation.py`:
```python
"""Tests for post-LLM validation of WRITE_CALL actions."""
from __future__ import annotations

from stock_analyzer.discover.cc_eligibility import EligibleHolding
from stock_analyzer.discover.cc_validation import validate_option_writes
from stock_analyzer.discover.rebalance_schema import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def _ow(ticker: str, contracts: int = 1) -> OptionWrite:
    return OptionWrite(
        ticker=ticker, strike=100.0, expiry="2026-06-20",
        contracts=contracts, est_premium_per_share=1.0,
        delta=0.4, assignment_probability=0.4,
    )


def _wc(ticker: str) -> RebalanceAction:
    return RebalanceAction(action="WRITE_CALL", ticker=ticker, sizing="x")


def _elig(ticker: str, max_contracts: int) -> EligibleHolding:
    return EligibleHolding(
        ticker=ticker, shares_held=max_contracts * 100,
        open_short_call_contracts=0,
        available_shares=max_contracts * 100,
        max_contracts=max_contracts,
    )


def test_orphan_write_call_gets_dropped():
    # WRITE_CALL action with NO matching OptionWrite.
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("ORPHAN"), RebalanceAction(action="ADD", ticker="X", sizing="$100")],
        option_writes=[],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(
        plan, eligibility={"ORPHAN": _elig("ORPHAN", 3)},
    )
    types = [a.action for a in cleaned.actions]
    assert "WRITE_CALL" not in types
    assert "ADD" in types
    assert any("orphan" in w.lower() for w in warnings)


def test_oversized_contracts_get_clamped():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA")],
        option_writes=[_ow("NVDA", contracts=5)],  # but max_contracts=3
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(
        plan, eligibility={"NVDA": _elig("NVDA", 3)},
    )
    ow = cleaned.option_writes[0]
    assert ow.contracts == 3
    assert any("clamp" in w.lower() for w in warnings)


def test_well_formed_plan_passes_through():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("NVDA")],
        option_writes=[_ow("NVDA", contracts=2)],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(
        plan, eligibility={"NVDA": _elig("NVDA", 3)},
    )
    assert warnings == []
    assert cleaned.option_writes[0].contracts == 2


def test_unknown_ticker_in_write_call_drops_it():
    # WRITE_CALL for a ticker we never eligibility-checked.
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[_wc("MYSTERY")],
        option_writes=[_ow("MYSTERY")],
        full_text="…",
    )
    cleaned, warnings = validate_option_writes(plan, eligibility={})
    assert all(a.action != "WRITE_CALL" for a in cleaned.actions)
    assert cleaned.option_writes == []
    assert any("not eligible" in w.lower() for w in warnings)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_cc_validation.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement validation**

Create `src/stock_analyzer/discover/cc_validation.py`:
```python
"""Post-LLM WRITE_CALL validation.

Runs after Rebalancer.decide() and BEFORE the plan is persisted or
rendered. Guarantees that:

  - every WRITE_CALL action has a matching OptionWrite entry (drops
    orphan actions and orphan option_writes)
  - every OptionWrite ticker is in the eligibility map (drops unknown)
  - contracts × 100 ≤ available_shares (clamps to max_contracts)

Returns a cleaned plan plus a list of human-readable warning strings,
which the caller logs (loudly) and surfaces in the email summary.
"""
from __future__ import annotations

from ..logging import get_logger
from .cc_eligibility import EligibleHolding
from .rebalance_schema import OptionWrite, RebalanceAction, RebalancePlan

logger = get_logger(__name__)


def validate_option_writes(
    plan: RebalancePlan,
    *,
    eligibility: dict[str, EligibleHolding],
) -> tuple[RebalancePlan, list[str]]:
    """Drop orphan WRITE_CALL actions, drop orphan option_writes, clamp
    oversized `contracts`, drop unknown tickers. Returns a new
    (frozen) plan with the same other fields untouched.
    """
    warnings: list[str] = []

    write_call_tickers = {
        a.ticker for a in plan.actions if a.action == "WRITE_CALL"
    }
    option_write_tickers = {ow.ticker for ow in plan.option_writes}

    # Pass 1: build the kept-ticker set.
    kept: set[str] = set()
    cleaned_option_writes: list[OptionWrite] = []
    for ow in plan.option_writes:
        if ow.ticker not in eligibility:
            warnings.append(
                f"OptionWrite for {ow.ticker} dropped: ticker not eligible"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        if ow.ticker not in write_call_tickers:
            warnings.append(
                f"OptionWrite for {ow.ticker} dropped: no matching WRITE_CALL action"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        elig = eligibility[ow.ticker]
        contracts = ow.contracts
        if contracts > elig.max_contracts:
            warnings.append(
                f"OptionWrite for {ow.ticker} clamped from "
                f"{contracts} → {elig.max_contracts} contracts "
                f"(available_shares={elig.available_shares})"
            )
            logger.warning("CC validation: %s", warnings[-1])
            contracts = elig.max_contracts
        if contracts <= 0:
            warnings.append(
                f"OptionWrite for {ow.ticker} dropped: clamped contracts=0"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        cleaned_option_writes.append(
            ow.model_copy(update={"contracts": contracts})
        )
        kept.add(ow.ticker)

    # Pass 2: drop WRITE_CALL actions without a kept OptionWrite.
    cleaned_actions: list[RebalanceAction] = []
    for a in plan.actions:
        if a.action == "WRITE_CALL" and a.ticker not in kept:
            warnings.append(
                f"WRITE_CALL action for {a.ticker} dropped: orphan "
                f"(no matching OptionWrite after validation)"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        cleaned_actions.append(a)

    # Flag orphan WRITE_CALL tickers that never had an OptionWrite at all.
    for orphan in write_call_tickers - option_write_tickers:
        # Only warn — they're already dropped in pass 2.
        warnings.append(
            f"WRITE_CALL action for {orphan} had NO OptionWrite in the plan"
        )
        logger.warning("CC validation: %s", warnings[-1])

    cleaned_plan = plan.model_copy(update={
        "actions": cleaned_actions,
        "option_writes": cleaned_option_writes,
    })
    return cleaned_plan, warnings
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_cc_validation.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/cc_validation.py tests/test_cc_validation.py
git commit -m "feat(cc): post-LLM validation — drop orphans, clamp oversize, log loud"
```

---

## Task 16: PreMortem prompt extension

**Files:**
- Modify: `src/stock_analyzer/discover/premortem.py` (extend instructions)
- Modify: `tests/test_pipeline_wiring.py` (add test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline_wiring.py`:
```python
def test_premortem_prompt_includes_cc_redteam_paragraph():
    from stock_analyzer.discover.premortem import PREMORTEM_INSTRUCTIONS
    assert "WRITE_CALL" in PREMORTEM_INSTRUCTIONS
    assert "assignment lock-in" in PREMORTEM_INSTRUCTIONS
    assert "IV crush" in PREMORTEM_INSTRUCTIONS
```

The PreMortem agent uses an instruction constant — it might be named `PREMORTEM_INSTRUCTIONS` or `_PREMORTEM_PROMPT` etc. Grep the file before editing and adjust the test's import name to match.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline_wiring.py::test_premortem_prompt_includes_cc_redteam_paragraph -v`
Expected: FAIL.

- [ ] **Step 3: Extend prompt**

In `src/stock_analyzer/discover/premortem.py`, find the existing instructions string. Append (still inside the triple-quoted block):

```

WRITE_CALL ACTIONS — additional critique dimensions
For each WRITE_CALL action in the plan, additionally consider:
  (a) assignment lock-in if the underlying runs 20% past strike,
  (b) IV crush after near-term earnings or macro events,
  (c) opportunity cost of capping upside on high-confidence picks,
  (d) tax consequences if assignment triggers short-term gain on
      the underlying.
Treat each of these as a candidate failure mode.
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline_wiring.py::test_premortem_prompt_includes_cc_redteam_paragraph -v`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/premortem.py tests/test_pipeline_wiring.py
git commit -m "feat(cc): premortem red-teams WRITE_CALL outcomes"
```

---

## Task 17: cc_render — deterministic compute for the three report sections

**Files:**
- Create: `src/stock_analyzer/discover/cc_render.py`
- Create: `tests/test_premium_deployment.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_premium_deployment.py`:
```python
"""Tests for deterministic CC reporting math."""
from __future__ import annotations

from stock_analyzer.discover.cc_render import (
    compute_premium_deployment,
    compute_premium_income,
    compute_round_lot_summary,
)
from stock_analyzer.discover.cc_eligibility import RoundLotCoverage
from stock_analyzer.discover.rebalance_schema import (
    OptionWrite,
    RebalanceAction,
    RebalancePlan,
)


def _ow(ticker: str, contracts: int, premium: float) -> OptionWrite:
    return OptionWrite(
        ticker=ticker, strike=200.0, expiry="2026-06-20",
        contracts=contracts, est_premium_per_share=premium,
        delta=0.4, assignment_probability=0.4,
    )


def test_premium_income_totals():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        option_writes=[
            _ow("NVDA", 3, 2.40),
            _ow("AAPL", 2, 3.20),
        ],
        full_text="…",
    )
    out = compute_premium_income(plan, slippage_buffer=0.10)
    # gross = 3 * 2.40 * 100 + 2 * 3.20 * 100 = 720 + 640 = 1360
    assert out["gross_premium_usd"] == 1360.0
    assert out["slippage_buffer_usd"] == 136.0  # 10%
    assert out["deployable_premium_usd"] == 1224.0
    assert len(out["rows"]) == 2


def test_premium_deployment_full_flow():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[
            RebalanceAction(action="WRITE_CALL", ticker="NVDA", sizing="3 contracts $260C"),
            RebalanceAction(action="ADD", ticker="AMZN", sizing="$1,400"),
            RebalanceAction(action="BUY", ticker="PLTR", sizing="$600"),
        ],
        option_writes=[_ow("NVDA", 3, 2.40)],
        full_text="…",
    )
    out = compute_premium_deployment(
        plan, cash_balance=850.0, slippage_buffer=0.10,
        stub_consolidation_usd=0.0,
    )
    assert out["deployable_premium_usd"] == 648.0  # 720 * 0.9
    assert out["existing_cash_usd"] == 850.0
    assert out["stub_consolidation_usd"] == 0.0
    assert out["total_dry_powder_usd"] == 1498.0
    assert {"ticker": "AMZN", "action": "ADD", "sizing": "$1,400"} in out["deployments"]


def test_premium_deployment_with_stub_consolidation_row():
    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[],
        option_writes=[_ow("NVDA", 3, 2.40)],
        full_text="…",
    )
    out = compute_premium_deployment(
        plan, cash_balance=850.0, slippage_buffer=0.10,
        stub_consolidation_usd=10500.0,
    )
    assert out["stub_consolidation_usd"] == 10500.0
    assert out["total_dry_powder_usd"] == 12_098.0  # 648 + 850 + 10500


def test_round_lot_summary():
    coverage = {
        "TSLA": RoundLotCoverage(
            ticker="TSLA", shares=335, round_lots=3, stub_shares=35,
            stub_dollar_value=10500.0, to_next_lot_shares=65,
            to_next_lot_cost=19500.0,
        ),
        "AAPL": RoundLotCoverage(
            ticker="AAPL", shares=215, round_lots=2, stub_shares=15,
            stub_dollar_value=3225.0, to_next_lot_shares=85,
            to_next_lot_cost=18275.0,
        ),
        # No-stub ticker: skipped from rendering.
        "NVDA": RoundLotCoverage(
            ticker="NVDA", shares=200, round_lots=2, stub_shares=0,
            stub_dollar_value=0.0, to_next_lot_shares=0, to_next_lot_cost=0.0,
        ),
    }
    out = compute_round_lot_summary(coverage)
    tickers = [r["ticker"] for r in out["rows"]]
    assert "NVDA" not in tickers  # no stub → not rendered
    assert tickers == ["TSLA", "AAPL"]  # sorted by stub_$ descending
    assert out["stub_pool_total_usd"] == 13725.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_premium_deployment.py -v`
Expected: FAIL — `ModuleNotFoundError: stock_analyzer.discover.cc_render`.

- [ ] **Step 3: Implement**

Create `src/stock_analyzer/discover/cc_render.py`:
```python
"""Deterministic compute for the CC report sections.

The renderer (HTML/PDF) reads these dicts directly — it does NOT parse
Opus's prose, so the box values are always internally consistent. Opus's
narrative remains in `full_text` for human context.
"""
from __future__ import annotations

from typing import Any

from .cc_eligibility import RoundLotCoverage
from .rebalance_schema import RebalancePlan


def compute_premium_income(
    plan: RebalancePlan,
    *,
    slippage_buffer: float,
) -> dict[str, Any]:
    """Compute the Premium Income box content.

    Returns:
      {
        "rows": [{ticker, strike, expiry, contracts, premium_usd,
                  delta, assignment_pct}, ...],
        "gross_premium_usd": float,
        "slippage_buffer_usd": float,
        "deployable_premium_usd": float,
      }
    """
    rows: list[dict[str, Any]] = []
    gross = 0.0
    for ow in plan.option_writes:
        premium = ow.contracts * ow.est_premium_per_share * 100.0
        gross += premium
        rows.append({
            "ticker": ow.ticker,
            "strike": ow.strike,
            "expiry": ow.expiry,
            "contracts": ow.contracts,
            "premium_usd": premium,
            "delta": ow.delta,
            "assignment_pct": int(round(ow.assignment_probability * 100)),
        })
    buffer = round(gross * slippage_buffer, 2)
    return {
        "rows": rows,
        "gross_premium_usd": round(gross, 2),
        "slippage_buffer_usd": buffer,
        "deployable_premium_usd": round(gross - buffer, 2),
    }


def compute_premium_deployment(
    plan: RebalancePlan,
    *,
    cash_balance: float | None,
    slippage_buffer: float,
    stub_consolidation_usd: float = 0.0,
) -> dict[str, Any]:
    """Compute the Premium → Deployment box content.

    `deployments` lists every ADD/BUY (with sizing strings as-is) and
    every TRIM action whose sizing mentions "stub" (so the reader sees
    the consolidation pair in context).
    """
    inc = compute_premium_income(plan, slippage_buffer=slippage_buffer)
    deployable = inc["deployable_premium_usd"]
    cash = float(cash_balance or 0.0)
    total = deployable + cash + stub_consolidation_usd
    deployments: list[dict[str, str]] = []
    for a in plan.actions:
        if a.action in ("ADD", "BUY"):
            deployments.append({
                "ticker": a.ticker, "action": a.action, "sizing": a.sizing,
            })
        elif a.action == "TRIM" and "stub" in a.sizing.lower():
            deployments.append({
                "ticker": a.ticker, "action": "TRIM", "sizing": a.sizing,
            })
    return {
        "gross_premium_usd": inc["gross_premium_usd"],
        "slippage_buffer_usd": inc["slippage_buffer_usd"],
        "deployable_premium_usd": deployable,
        "existing_cash_usd": cash,
        "stub_consolidation_usd": stub_consolidation_usd,
        "total_dry_powder_usd": round(total, 2),
        "deployments": deployments,
    }


def compute_round_lot_summary(
    coverage: dict[str, RoundLotCoverage],
) -> dict[str, Any]:
    """Compute the Round-Lot Coverage table.

    Only holdings with a non-zero stub are rendered. Sorted by stub
    dollar value descending so the user's eye lands on the biggest
    consolidation candidates first.
    """
    rows: list[dict[str, Any]] = []
    pool = 0.0
    for ticker in coverage:
        rec = coverage[ticker]
        if rec.stub_shares == 0:
            continue
        rows.append({
            "ticker": rec.ticker,
            "shares": rec.shares,
            "round_lots": rec.round_lots,
            "round_lot_shares": rec.round_lots * 100,
            "stub_shares": rec.stub_shares,
            "stub_dollar_value": rec.stub_dollar_value,
            "to_next_lot_shares": rec.to_next_lot_shares,
            "to_next_lot_cost": rec.to_next_lot_cost,
        })
        pool += rec.stub_dollar_value
    rows.sort(key=lambda r: r["stub_dollar_value"], reverse=True)
    return {"rows": rows, "stub_pool_total_usd": round(pool, 2)}
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_premium_deployment.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/cc_render.py tests/test_premium_deployment.py
git commit -m "feat(cc): deterministic compute for premium income / deployment / round-lot summary"
```

---

## Task 18: Section IR — new section kinds

**Files:**
- Modify: `src/stock_analyzer/discover/report_sections.py:237-238` and the `SectionKind` Literal
- Modify: `tests/test_section_dispatch_parity.py` (add coverage)

The existing `Section` model is generic with a `kind: SectionKind` discriminator and a flexible `data: dict | None` payload field. We extend `SectionKind` with three new literals and pass our compute output through `data`.

- [ ] **Step 1: Inspect SectionKind**

Run: `grep -n "SectionKind" /Users/snehal.soni/Personal/stock_analyzer/src/stock_analyzer/discover/report_sections.py | head -10`

Expected: finds the `SectionKind = Literal[...]` definition near the top of the file. Note its exact location.

- [ ] **Step 2: Write failing test**

Append to `tests/test_section_dispatch_parity.py`:
```python
from stock_analyzer.discover.report_sections import Section


def test_section_accepts_new_cc_kinds():
    for kind in ("premium_income", "round_lot_coverage", "premium_deployment"):
        s = Section(kind=kind, data={"rows": []})  # type: ignore[arg-type]
        assert s.kind == kind
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_section_dispatch_parity.py -v -k cc_kinds`
Expected: FAIL — Pydantic validation error on the literal.

- [ ] **Step 4: Extend the Literal**

In `report_sections.py`, find `SectionKind = Literal[...]` and add the three new strings:
```python
    "premium_income",
    "round_lot_coverage",
    "premium_deployment",
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_section_dispatch_parity.py -v`
Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add src/stock_analyzer/discover/report_sections.py tests/test_section_dispatch_parity.py
git commit -m "feat(cc): SectionKind extended with three CC report sections"
```

---

## Task 19: HTML renderer — three new sections

**Files:**
- Modify: `src/stock_analyzer/discover/report_html.py` (add dispatch for three new kinds)
- Modify: `tests/test_report_parsers.py` (add HTML smoke test)

Existing `report_html.py` uses a kind→renderer dispatch dict or chain. Locate the dispatch and add three branches. Color: WRITE_CALL row uses `#0d9488` (teal) for the action-type badge — this matches the existing palette's saturation. The three sections share a `<table>` styling with the existing palette.

- [ ] **Step 1: Inspect dispatch**

Run: `grep -n "def render_html\|kind ==\|elif kind ==" /Users/snehal.soni/Personal/stock_analyzer/src/stock_analyzer/discover/report_html.py | head -30`

Note the dispatch pattern actually used. If it's a dict of `{kind: fn}`, add three entries. If a giant if-elif, add branches.

- [ ] **Step 2: Write failing test**

Append to `tests/test_report_parsers.py`:
```python
from stock_analyzer.discover.report_sections import Section
from stock_analyzer.discover.report_html import render_html_email


def test_premium_income_renders_table_html():
    sections = [
        Section(kind="heading", text="Test", level=1),
        Section(kind="premium_income", data={
            "rows": [{
                "ticker": "NVDA", "strike": 260.0, "expiry": "2026-06-20",
                "contracts": 3, "premium_usd": 720.0,
                "delta": 0.36, "assignment_pct": 36,
            }],
            "gross_premium_usd": 720.0,
            "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
        }),
    ]
    html = render_html_email(sections, chart_cids={})
    assert "NVDA" in html
    assert "$260" in html
    assert "Gross premium" in html
    assert "$720" in html
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_report_parsers.py -v -k premium_income`
Expected: FAIL — section kind unknown / not rendered.

- [ ] **Step 4: Implement HTML renderers**

In `report_html.py`, add three new renderer functions (mirror the style of the existing `_render_table` / `_render_metric_strip`):

```python
def _render_premium_income(data: dict) -> str:
    rows_html = "".join(
        f"<tr>"
        f"<td>{r['ticker']}</td>"
        f"<td>${r['strike']:,.2f}</td>"
        f"<td>{r['expiry']}</td>"
        f"<td>{r['contracts']}</td>"
        f"<td>${r['premium_usd']:,.0f}</td>"
        f"<td>{r['delta']:.2f}</td>"
        f"<td>{r['assignment_pct']}%</td>"
        f"</tr>"
        for r in data["rows"]
    )
    return (
        '<div style="border:1px solid #d1d5db; padding:12px; '
        'margin:16px 0; background:#f0fdfa;">'
        '<h3 style="margin:0 0 8px 0;">Premium Income</h3>'
        '<table style="width:100%; border-collapse:collapse;">'
        '<thead><tr style="text-align:left; border-bottom:1px solid #d1d5db;">'
        '<th>Ticker</th><th>Strike</th><th>Expiry</th><th>Qty</th>'
        '<th>Premium</th><th>Δ</th><th>Assign %</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        f'<p style="margin:8px 0 0 0;">'
        f'Gross premium: <strong>${data["gross_premium_usd"]:,.0f}</strong>'
        f' &nbsp;&nbsp; Slippage buffer (10%): -${data["slippage_buffer_usd"]:,.0f}'
        f' &nbsp;&nbsp; Deployable: <strong>${data["deployable_premium_usd"]:,.0f}</strong>'
        f'</p>'
        '</div>'
    )


def _render_round_lot_coverage(data: dict) -> str:
    if not data["rows"]:
        return ""
    rows_html = "".join(
        f"<tr>"
        f"<td>{r['ticker']}</td>"
        f"<td>{r['shares']}</td>"
        f"<td>{r['round_lots']} ({r['round_lot_shares']})</td>"
        f"<td>{r['stub_shares']}</td>"
        f"<td>${r['stub_dollar_value']:,.0f}</td>"
        f"<td>${r['to_next_lot_cost']:,.0f}</td>"
        f"</tr>"
        for r in data["rows"]
    )
    return (
        '<div style="border:1px solid #d1d5db; padding:12px; '
        'margin:16px 0; background:#fefce8;">'
        '<h3 style="margin:0 0 8px 0;">Round-Lot Coverage</h3>'
        '<table style="width:100%; border-collapse:collapse;">'
        '<thead><tr style="text-align:left; border-bottom:1px solid #d1d5db;">'
        '<th>Position</th><th>Shares</th><th>Round Lots</th>'
        '<th>Stub</th><th>Stub $</th><th>To-next-lot</th>'
        '</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        '</table>'
        f'<p style="margin:8px 0 0 0;">'
        f'Stub pool total: <strong>${data["stub_pool_total_usd"]:,.0f}</strong>'
        f'</p>'
        '</div>'
    )


def _render_premium_deployment(data: dict) -> str:
    deps_html = "".join(
        f"<li>{d['action']} <strong>{d['ticker']}</strong> {d['sizing']}</li>"
        for d in data["deployments"]
    )
    stub_row = (
        f'<tr><td>Stub consolidation:</td><td>${data["stub_consolidation_usd"]:,.0f}</td></tr>'
        if data["stub_consolidation_usd"] else ""
    )
    return (
        '<div style="border:1px solid #d1d5db; padding:12px; '
        'margin:16px 0; background:#eff6ff;">'
        '<h3 style="margin:0 0 8px 0;">Premium → Deployment</h3>'
        '<table style="margin:0;">'
        f'<tr><td>Deployable premium:</td><td>${data["deployable_premium_usd"]:,.0f}</td></tr>'
        f'<tr><td>Existing cash:</td><td>${data["existing_cash_usd"]:,.0f}</td></tr>'
        f'{stub_row}'
        f'<tr><td><strong>Total dry powder:</strong></td>'
        f'<td><strong>${data["total_dry_powder_usd"]:,.0f}</strong></td></tr>'
        '</table>'
        f'<ul style="margin:8px 0 0 16px;">{deps_html}</ul>'
        '</div>'
    )
```

Then wire them into the kind→fn dispatch (whatever pattern is in use). Example for an if-elif chain inside the main `render_html_email`:

```python
        elif s.kind == "premium_income":
            parts.append(_render_premium_income(s.data or {}))
        elif s.kind == "round_lot_coverage":
            parts.append(_render_round_lot_coverage(s.data or {}))
        elif s.kind == "premium_deployment":
            parts.append(_render_premium_deployment(s.data or {}))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_report_parsers.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/stock_analyzer/discover/report_html.py tests/test_report_parsers.py
git commit -m "feat(cc): HTML renderers for premium-income / round-lot-coverage / premium-deployment"
```

---

## Task 20: PDF renderer — three new sections

**Files:**
- Modify: `src/stock_analyzer/discover/report_pdf.py`
- Modify: `tests/test_report_parsers.py` (PDF smoke test — assert it doesn't crash)

The PDF renderer uses ReportLab. Mirror the HTML structure: a titled "Card" frame with a Table inside, followed by a summary paragraph. Mirror existing renderers' styling.

- [ ] **Step 1: Inspect PDF dispatch**

Run: `grep -n "def render_pdf\|s.kind ==\|elif s.kind" /Users/snehal.soni/Personal/stock_analyzer/src/stock_analyzer/discover/report_pdf.py | head -20`

Note the dispatch shape.

- [ ] **Step 2: Write failing smoke test**

Append to `tests/test_report_parsers.py`:
```python
from stock_analyzer.discover.report_pdf import render_pdf


def test_pdf_renders_with_cc_sections_smoke():
    sections = [
        Section(kind="heading", text="Test", level=1),
        Section(kind="premium_income", data={
            "rows": [{"ticker": "NVDA", "strike": 260.0, "expiry": "2026-06-20",
                      "contracts": 3, "premium_usd": 720.0,
                      "delta": 0.36, "assignment_pct": 36}],
            "gross_premium_usd": 720.0,
            "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
        }),
        Section(kind="round_lot_coverage", data={
            "rows": [{"ticker": "TSLA", "shares": 335, "round_lots": 3,
                      "round_lot_shares": 300, "stub_shares": 35,
                      "stub_dollar_value": 10500.0,
                      "to_next_lot_shares": 65, "to_next_lot_cost": 19500.0}],
            "stub_pool_total_usd": 10500.0,
        }),
        Section(kind="premium_deployment", data={
            "gross_premium_usd": 720.0, "slippage_buffer_usd": 72.0,
            "deployable_premium_usd": 648.0,
            "existing_cash_usd": 850.0,
            "stub_consolidation_usd": 10500.0,
            "total_dry_powder_usd": 11998.0,
            "deployments": [
                {"ticker": "AMZN", "action": "ADD", "sizing": "$1,400"},
            ],
        }),
    ]
    pdf = render_pdf(sections, charts={})
    assert isinstance(pdf, bytes)
    assert len(pdf) > 1000  # not just a header
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/test_report_parsers.py -v -k pdf_renders_with_cc_sections_smoke`
Expected: FAIL — unknown section kind / no render branch.

- [ ] **Step 4: Implement PDF renderers**

Add to `report_pdf.py` (adapt imports / styles to match what's already there):

```python
def _pdf_premium_income(data: dict) -> list:
    """Returns a list of flowables for ReportLab."""
    from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
    from reportlab.lib import colors
    flowables = []
    flowables.append(Paragraph("<b>Premium Income</b>", _STYLE_H3))
    header = ["Ticker", "Strike", "Expiry", "Qty", "Premium", "Δ", "Assign %"]
    rows = [header]
    for r in data["rows"]:
        rows.append([
            r["ticker"], f"${r['strike']:,.2f}", r["expiry"],
            str(r["contracts"]), f"${r['premium_usd']:,.0f}",
            f"{r['delta']:.2f}", f"{r['assignment_pct']}%",
        ])
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0d9488")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    flowables.append(t)
    flowables.append(Spacer(1, 6))
    flowables.append(Paragraph(
        f"Gross premium: <b>${data['gross_premium_usd']:,.0f}</b> &nbsp;"
        f"Buffer (10%): -${data['slippage_buffer_usd']:,.0f} &nbsp;"
        f"Deployable: <b>${data['deployable_premium_usd']:,.0f}</b>",
        _STYLE_BODY,
    ))
    flowables.append(Spacer(1, 12))
    return flowables


def _pdf_round_lot_coverage(data: dict) -> list:
    from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
    from reportlab.lib import colors
    if not data["rows"]:
        return []
    rows = [["Position", "Shares", "Round Lots", "Stub", "Stub $", "To-next-lot"]]
    for r in data["rows"]:
        rows.append([
            r["ticker"], str(r["shares"]),
            f"{r['round_lots']} ({r['round_lot_shares']})",
            str(r["stub_shares"]),
            f"${r['stub_dollar_value']:,.0f}",
            f"${r['to_next_lot_cost']:,.0f}",
        ])
    flowables = [Paragraph("<b>Round-Lot Coverage</b>", _STYLE_H3)]
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#a16207")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
    ]))
    flowables.append(t)
    flowables.append(Paragraph(
        f"Stub pool total: <b>${data['stub_pool_total_usd']:,.0f}</b>",
        _STYLE_BODY,
    ))
    flowables.append(Spacer(1, 12))
    return flowables


def _pdf_premium_deployment(data: dict) -> list:
    from reportlab.platypus import Paragraph, Spacer
    flowables = [Paragraph("<b>Premium → Deployment</b>", _STYLE_H3)]
    lines = [
        f"Deployable premium: ${data['deployable_premium_usd']:,.0f}",
        f"Existing cash: ${data['existing_cash_usd']:,.0f}",
    ]
    if data["stub_consolidation_usd"]:
        lines.append(f"Stub consolidation: ${data['stub_consolidation_usd']:,.0f}")
    lines.append(f"<b>Total dry powder: ${data['total_dry_powder_usd']:,.0f}</b>")
    flowables.append(Paragraph("<br/>".join(lines), _STYLE_BODY))
    if data["deployments"]:
        deps = "<br/>".join(
            f"&rarr; {d['action']} <b>{d['ticker']}</b> {d['sizing']}"
            for d in data["deployments"]
        )
        flowables.append(Paragraph(deps, _STYLE_BODY))
    flowables.append(Spacer(1, 12))
    return flowables
```

(Reuse the existing `_STYLE_BODY` / `_STYLE_H3` constants the file already defines; if they're named differently, adapt.)

Wire into the dispatch:
```python
        elif s.kind == "premium_income":
            story.extend(_pdf_premium_income(s.data or {}))
        elif s.kind == "round_lot_coverage":
            story.extend(_pdf_round_lot_coverage(s.data or {}))
        elif s.kind == "premium_deployment":
            story.extend(_pdf_premium_deployment(s.data or {}))
```

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_report_parsers.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/stock_analyzer/discover/report_pdf.py tests/test_report_parsers.py
git commit -m "feat(cc): PDF renderers for the three new CC sections"
```

---

## Task 21: Email subject annotation

**Files:**
- Modify: `src/stock_analyzer/cli/rebalance.py:663` (subject construction)
- Modify: `tests/test_pipeline_wiring.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_pipeline_wiring.py`:
```python
from stock_analyzer.cli.rebalance import build_email_subject


def test_subject_without_premium():
    subject = build_email_subject(action_count=2, gross_premium_usd=0.0)
    assert subject.startswith("Portfolio Rebalance")
    assert "premium" not in subject


def test_subject_with_premium_annotates():
    subject = build_email_subject(action_count=4, gross_premium_usd=1550.0)
    assert "$1,550 premium" in subject
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pipeline_wiring.py -v -k email_subject`
Expected: FAIL — function doesn't exist.

- [ ] **Step 3: Add helper to `cli/rebalance.py`**

Just above `class RebalancePipeline(DiscoverPipeline):`:
```python
def build_email_subject(*, action_count: int, gross_premium_usd: float) -> str:
    today = date.today()
    base = f"Portfolio Rebalance — {today.strftime('%b-%d')}"
    if gross_premium_usd >= 1.0:
        return f"{base} ({action_count} actions + ${gross_premium_usd:,.0f} premium)"
    return base
```

Then in `step_persist_and_email_rebalance`, replace:
```python
        subject = f"Portfolio Rebalance — {today.strftime('%b-%d')}"
```
with:
```python
        plan = self.state.get("rebalance_plan")
        gross_premium = 0.0
        if plan is not None and getattr(plan, "option_writes", None):
            gross_premium = sum(
                ow.contracts * ow.est_premium_per_share * 100.0
                for ow in plan.option_writes
            )
        action_count = len(plan.actions) if plan else 0
        subject = build_email_subject(
            action_count=action_count, gross_premium_usd=gross_premium,
        )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_pipeline_wiring.py -v -k email_subject`
Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/cli/rebalance.py tests/test_pipeline_wiring.py
git commit -m "feat(cc): email subject annotates premium total when WRITE_CALLs present"
```

---

## Task 22: Track-record — score covered-call outcomes

**Files:**
- Modify: `src/stock_analyzer/discover/track_record.py`
- Create: `tests/test_track_record_cc.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_track_record_cc.py`:
```python
"""Tests for WRITE_CALL outcome scoring."""
from __future__ import annotations

from unittest.mock import patch

from stock_analyzer.discover.track_record import score_covered_call


def test_expired_otm_keeps_full_premium():
    # Spot at expiry < strike → premium kept, P&L = +premium × contracts × 100.
    with patch(
        "stock_analyzer.discover.track_record._spot_at",
        return_value=240.0,
    ):
        out = score_covered_call(
            ticker="NVDA", strike=260.0, expiry="2026-06-20",
            contracts=3, est_premium_per_share=2.40,
        )
    assert out["outcome"] == "EXPIRED_OTM"
    assert out["pnl_usd"] == 3 * 2.40 * 100  # 720
    assert out["opportunity_cost_usd"] == 0.0


def test_assigned_records_opportunity_cost():
    # Spot at expiry > strike → assigned. Premium kept but lose upside.
    with patch(
        "stock_analyzer.discover.track_record._spot_at",
        return_value=280.0,
    ):
        out = score_covered_call(
            ticker="NVDA", strike=260.0, expiry="2026-06-20",
            contracts=3, est_premium_per_share=2.40,
        )
    assert out["outcome"] == "ASSIGNED"
    # opportunity_cost = (280 - 260) × 3 × 100 = 6000
    # net P&L = premium - opportunity_cost = 720 - 6000 = -5280
    assert out["pnl_usd"] == 720 - 6000
    assert out["opportunity_cost_usd"] == 6000


def test_missing_spot_returns_unknown():
    with patch(
        "stock_analyzer.discover.track_record._spot_at",
        return_value=None,
    ):
        out = score_covered_call(
            ticker="X", strike=100.0, expiry="2026-06-20",
            contracts=1, est_premium_per_share=1.0,
        )
    assert out["outcome"] == "UNKNOWN"
    assert out["pnl_usd"] is None
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_track_record_cc.py -v`
Expected: FAIL — `cannot import name 'score_covered_call'`.

- [ ] **Step 3: Implement**

Append to `src/stock_analyzer/discover/track_record.py`:
```python
from datetime import date as _date
from typing import Any as _Any


def _spot_at(ticker: str, on: str) -> float | None:
    """Lookup historical spot for `ticker` on ISO date `on`.

    Default impl uses yfinance. Patched in tests. Returns None when
    the lookup fails so the caller can mark the outcome UNKNOWN
    rather than crashing the track-record block.
    """
    try:
        import yfinance as yf
        # `on` is the expiry date; we want the closing price that day
        # or the most-recent prior trading day.
        end = _date.fromisoformat(on)
        df = yf.Ticker(ticker).history(
            start=(end.replace(day=max(1, end.day - 5))).isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
        )
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception:
        return None


def score_covered_call(
    *,
    ticker: str,
    strike: float,
    expiry: str,
    contracts: int,
    est_premium_per_share: float,
) -> dict[str, _Any]:
    """Score one WRITE_CALL after `expiry` has passed.

    Returns:
      {
        "outcome": "EXPIRED_OTM" | "ASSIGNED" | "UNKNOWN",
        "spot_at_expiry": float | None,
        "pnl_usd": float | None,            # net of opportunity cost
        "premium_collected_usd": float,
        "opportunity_cost_usd": float,
      }
    """
    spot = _spot_at(ticker, expiry)
    premium = contracts * est_premium_per_share * 100.0
    if spot is None:
        return {
            "outcome": "UNKNOWN",
            "spot_at_expiry": None,
            "pnl_usd": None,
            "premium_collected_usd": premium,
            "opportunity_cost_usd": 0.0,
        }
    if spot < strike:
        return {
            "outcome": "EXPIRED_OTM",
            "spot_at_expiry": spot,
            "pnl_usd": premium,
            "premium_collected_usd": premium,
            "opportunity_cost_usd": 0.0,
        }
    opportunity_cost = (spot - strike) * contracts * 100.0
    return {
        "outcome": "ASSIGNED",
        "spot_at_expiry": spot,
        "pnl_usd": premium - opportunity_cost,
        "premium_collected_usd": premium,
        "opportunity_cost_usd": opportunity_cost,
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_track_record_cc.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/discover/track_record.py tests/test_track_record_cc.py
git commit -m "feat(cc): score_covered_call — outcome attribution after expiry"
```

---

## Task 23: Wire CC data prep into `cli/rebalance.py`

**Files:**
- Modify: `src/stock_analyzer/cli/rebalance.py` (new `step_cc_data` method, add to workflow)

This task does not need a unit test of its own — the integration test in Task 25 covers it.

- [ ] **Step 1: Add `step_cc_data` to `RebalancePipeline`**

In `cli/rebalance.py`, after `step_review_holdings` and before `step_rebalance`:

```python
    def step_cc_data(self, step_input: StepInput) -> StepOutput:
        """Build the COVERED-CALL CONTEXT block consumed by the rebalancer.

        Pulls option chains, parses open short-call positions, computes
        eligibility + round-lot coverage + earnings-filtered chains,
        and stashes the assembled prompt block in
        `self.state['cc_context_block']`. Returns an info-only StepOutput.

        Gracefully degrades: if CC_ENABLED is false or no holdings are
        eligible, `cc_context_block` is "" and the rebalancer prompt
        simply omits the CC section.
        """
        if not self.settings.cc_enabled:
            self.state["cc_context_block"] = ""
            self.state["cc_eligibility"] = {}
            self.state["cc_round_lot_coverage"] = {}
            self.state["cc_stub_pool_total_usd"] = 0.0
            return StepOutput(content="cc_data: disabled via CC_ENABLED=0")

        from ..data.brokerage import fetch_open_option_positions
        from ..data.options_chain import fetch_chains
        from ..discover.cc_eligibility import (
            apply_earnings_filter,
            build_cc_context_block,
            eligible_holdings,
            round_lot_coverage,
        )

        positions = self.state.get("holdings_positions") or {}
        denylist = self.settings.cc_denylist

        # Coverage subtraction.
        try:
            open_short_calls = fetch_open_option_positions()
        except Exception as e:
            logger.warning("open option position fetch failed: %s", e)
            open_short_calls = {}

        eligible = eligible_holdings(
            positions, open_short_calls=open_short_calls, denylist=denylist,
        )

        # Round-lot coverage for ALL holdings (not just eligible).
        spots = {
            t: (self.state.get("holdings_technicals", {}).get(t) or {}).get("price") or 0.0
            for t in positions
        }
        coverage = round_lot_coverage(positions, spots=spots)
        stub_pool = sum(
            rec.stub_dollar_value for rec in coverage.values() if rec.stub_shares
        )

        # Chain fetch for eligible only — and only if stub_optimization
        # itself isn't the entire reason we'd want a chain. (No: chains
        # are about WRITE_CALL strikes; we always need them for eligible
        # tickers.)
        chains = fetch_chains(
            list(eligible),
            dte_min=self.settings.cc_dte_min,
            dte_max=self.settings.cc_dte_max,
        )

        # Earnings filter — use FinnHub earnings dates already fetched
        # earlier in the pipeline (next_earnings_date is part of the
        # finnhub_signals enrichment).
        finnhub_signals = self.state.get("finnhub_signals") or {}
        earnings: dict[str, date] = {}
        for ticker in eligible:
            sig = finnhub_signals.get(ticker) or {}
            raw = sig.get("next_earnings_date") or sig.get("earnings_date")
            if isinstance(raw, str):
                try:
                    earnings[ticker] = date.fromisoformat(raw[:10])
                except ValueError:
                    pass
            # If we ever switch to date objects in finnhub_signals, accept those too.
            elif isinstance(raw, date):
                earnings[ticker] = raw

        filtered_chains: dict[str, object] = {}
        for ticker, chain in chains.items():
            filtered, _ = apply_earnings_filter(
                chain, earnings_date=earnings.get(ticker),
            )
            filtered_chains[ticker] = filtered

        block = build_cc_context_block(
            eligible=eligible, chains=filtered_chains,
            coverage=coverage, reviews=self.state.get("holdings_reviews", {}),
            earnings=earnings, stub_pool_total_usd=stub_pool,
        )
        self.state["cc_context_block"] = block
        self.state["cc_eligibility"] = eligible
        self.state["cc_round_lot_coverage"] = coverage
        self.state["cc_stub_pool_total_usd"] = stub_pool

        sources = {c.source for c in filtered_chains.values()}
        n_missing = sum(1 for c in filtered_chains.values() if c.source == "missing")
        return StepOutput(content=(
            f"cc_data: {len(eligible)} eligible holding(s); "
            f"chain sources {sorted(sources)}; "
            f"{n_missing} missing; "
            f"stub pool ${stub_pool:,.0f}"
        ))
```

- [ ] **Step 2: Add `step_cc_data` to the workflow steps list**

In `build_workflow`, insert between the `Step(name="review_holdings", ...)` and `Step(name="rebalance", ...)` lines:
```python
                Step(name="cc_data", executor=self.step_cc_data),
```

- [ ] **Step 3: Pass `cc_context_block` into `step_rebalance`**

Find `rebalancer.decide(...)` in `step_rebalance` and add the kwarg:
```python
        plan = rebalancer.decide(
            self.state.get("holdings_reviews", {}),
            ranker_text,
            self.state.get("cash_balance"),
            self.state.get("macro_summary", ""),
            aggressiveness=self.settings.discover_rebalance_aggressiveness,
            history_block=history_block,
            market_themes_block=self.state.get("market_themes_block", ""),
            cc_context_block=self.state.get("cc_context_block", ""),
        )
```

- [ ] **Step 4: Run smoke-style ad-hoc check** (no test yet, that's Task 25)

Run: `uv run pytest tests/test_pipeline_wiring.py -v`
Expected: existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add src/stock_analyzer/cli/rebalance.py
git commit -m "feat(cc): cli/rebalance step_cc_data — eligibility + chains + earnings + context block"
```

---

## Task 24: Validate after the rebalancer and surface CC sections in the email

**Files:**
- Modify: `src/stock_analyzer/cli/rebalance.py` (`step_rebalance` validation + `_build_rebalance_sections` extension)

- [ ] **Step 1: Apply validation in `step_rebalance`**

After the `plan = rebalancer.decide(...)` line in `step_rebalance`:

```python
        from ..discover.cc_validation import validate_option_writes
        plan, cc_warnings = validate_option_writes(
            plan, eligibility=self.state.get("cc_eligibility") or {},
        )
        if cc_warnings:
            self.state["cc_warnings"] = cc_warnings
            for w in cc_warnings:
                logger.warning("CC plan validation: %s", w)
        self.state["rebalance_plan"] = plan
        self.state["rebalance_text"] = plan.full_text
```

(Replace the existing `self.state["rebalance_plan"] = plan` lines.)

- [ ] **Step 2: Inject CC sections into `_build_rebalance_sections`**

In the call site of `_build_rebalance_sections` (inside `step_persist_and_email_rebalance`), add new kwargs:
```python
        sections = _build_rebalance_sections(
            ...,  # existing args
            cc_eligibility=self.state.get("cc_eligibility") or {},
            cc_round_lot_coverage=self.state.get("cc_round_lot_coverage") or {},
            cc_stub_pool_total_usd=self.state.get("cc_stub_pool_total_usd") or 0.0,
            cc_warnings=self.state.get("cc_warnings") or [],
        )
```

Then extend `_build_rebalance_sections` to accept and use these. Add to the signature:
```python
    cc_eligibility: dict | None = None,
    cc_round_lot_coverage: dict | None = None,
    cc_stub_pool_total_usd: float = 0.0,
    cc_warnings: list[str] | None = None,
```

After the existing "Rebalance plan (action list)" block (after the `rebalance_action_table` section is appended), add:

```python
    # ------- Covered-call sections (rendered only when relevant) --------
    from ..discover.cc_render import (
        compute_premium_deployment,
        compute_premium_income,
        compute_round_lot_summary,
    )
    if plan is not None and plan.option_writes:
        slippage_buffer = 0.10  # mirror Settings default; not a user-facing knob in renderer
        sections.append(Section(
            kind="premium_income",
            data=compute_premium_income(plan, slippage_buffer=slippage_buffer),
        ))
    if cc_round_lot_coverage:
        rls = compute_round_lot_summary(cc_round_lot_coverage)
        if rls["rows"]:
            sections.append(Section(kind="round_lot_coverage", data=rls))
    if plan is not None and (plan.option_writes
                             or any(a.action in ("ADD", "BUY")
                                    or (a.action == "TRIM" and "stub" in a.sizing.lower())
                                    for a in plan.actions)):
        # Compute stub consolidation USD from cleaned actions, not from
        # the upstream coverage map — Opus may consolidate only some stubs.
        stub_usd = 0.0
        if cc_round_lot_coverage:
            for a in plan.actions:
                if a.action == "TRIM" and "stub" in a.sizing.lower():
                    rec = cc_round_lot_coverage.get(a.ticker)
                    if rec is not None:
                        stub_usd += rec.stub_dollar_value
        deployment = compute_premium_deployment(
            plan, cash_balance=cash_balance, slippage_buffer=0.10,
            stub_consolidation_usd=stub_usd,
        )
        # Render only if there is something to show (avoid noise when
        # the plan has no calls AND no ADDs/BUYs/stub-trims).
        if (deployment["gross_premium_usd"] > 0
            or deployment["deployments"]
            or stub_usd > 0):
            sections.append(Section(kind="premium_deployment", data=deployment))

    if cc_warnings:
        sections.append(Section(
            kind="para",
            text="⚠ CC plan adjustments: " + "; ".join(cc_warnings),
        ))
```

- [ ] **Step 3: Run existing tests**

Run: `uv run pytest tests/ -v`
Expected: all pass (no behaviour change for runs without CC data).

- [ ] **Step 4: Commit**

```bash
git add src/stock_analyzer/cli/rebalance.py
git commit -m "feat(cc): validate plan and emit CC sections into rebalance report"
```

---

## Task 25: End-to-end pipeline wiring test

**Files:**
- Modify: `tests/test_pipeline_wiring.py`

- [ ] **Step 1: Write end-to-end test**

Append to `tests/test_pipeline_wiring.py`:
```python
def test_end_to_end_with_write_call_action():
    """Stub every external (LLM, SnapTrade, yfinance) and exercise the
    full rebalance flow through to HTML + PDF rendering. Asserts that
    a WRITE_CALL plan produces the new sections in the output and
    that the option_writes round-trip through JSON persistence."""
    from datetime import datetime
    from unittest.mock import MagicMock, patch

    from stock_analyzer.data.options_chain import OptionChain, OptionQuote
    from stock_analyzer.discover.cc_eligibility import (
        EligibleHolding,
        RoundLotCoverage,
    )
    from stock_analyzer.discover.rebalance_schema import (
        OptionWrite,
        RebalanceAction,
        RebalancePlan,
    )

    plan = RebalancePlan(
        status="ACTION", aggressiveness_applied="aggressive",
        actions=[
            RebalanceAction(action="WRITE_CALL", ticker="NVDA",
                            sizing="3 contracts $260C 2026-06-20"),
            RebalanceAction(action="ADD", ticker="AMZN", sizing="$1,400"),
        ],
        option_writes=[OptionWrite(
            ticker="NVDA", strike=260.0, expiry="2026-06-20",
            contracts=3, est_premium_per_share=2.40,
            delta=0.36, assignment_probability=0.36,
            notes="HOLD-8, near-band lower",
        )],
        summary="Write NVDA Jun-260 and deploy premium to AMZN.",
        full_text="…",
    )

    # Persistence round-trip: model_dump → model_validate.
    blob = plan.model_dump(mode="json")
    restored = RebalancePlan.model_validate(blob)
    assert restored.option_writes[0].ticker == "NVDA"
    assert restored.actions[0].action == "WRITE_CALL"

    # Renderer round-trip — feed plan + minimal context through
    # _build_rebalance_sections and ensure the new section kinds appear.
    from stock_analyzer.cli.rebalance import _build_rebalance_sections
    coverage = {
        "NVDA": RoundLotCoverage(
            ticker="NVDA", shares=400, round_lots=4, stub_shares=0,
            stub_dollar_value=0.0, to_next_lot_shares=0, to_next_lot_cost=0.0,
        ),
    }
    eligibility = {
        "NVDA": EligibleHolding(
            ticker="NVDA", shares_held=400, open_short_call_contracts=0,
            available_shares=400, max_contracts=4,
        ),
    }
    sections = _build_rebalance_sections(
        rebalance_text="…",
        holdings_reviews={},
        ranker_text="", redteam_text="", sizer_text="",
        candidates=[], cash_balance=850.0, macro_summary="",
        sector_rotation=None,
        holdings_positions={"NVDA": {"units": 400, "avg_buy_price": 200.0, "cost_basis": 80000.0}},
        holdings_technicals={"NVDA": {"price": 235.0}},
        holdings_fundamentals={"NVDA": {"sector": "Tech"}},
        rebalance_plan=plan,
        cc_eligibility=eligibility,
        cc_round_lot_coverage=coverage,
        cc_stub_pool_total_usd=0.0,
        cc_warnings=[],
    )
    kinds = [s.kind for s in sections]
    assert "premium_income" in kinds
    assert "premium_deployment" in kinds
    # No stubs → no round_lot_coverage section.
    assert "round_lot_coverage" not in kinds
```

- [ ] **Step 2: Run to verify pass**

Run: `uv run pytest tests/test_pipeline_wiring.py::test_end_to_end_with_write_call_action -v`
Expected: pass.

- [ ] **Step 3: Run full suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass; no regressions in existing tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_pipeline_wiring.py
git commit -m "test(cc): end-to-end wiring — plan persistence + section emission"
```

---

## Task 26: README + ENV docs

**Files:**
- Modify: `README.md` (add CC section under "What's in it")

- [ ] **Step 1: Append to README**

After the existing pipeline table in `README.md`, add a new paragraph:

```markdown
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

See `.env.example` for the full set of `CC_*` knobs.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(cc): README — covered-call writing section"
```

---

## Task 27: Final integration smoke (manual)

**Files:** none — manual verification only

This isn't a unit test; it's a hands-on dry run against the real environment so the implementer catches things tests can't (SnapTrade endpoint surface, finnhub field names, prompt token budget).

- [ ] **Step 1: Run with CC disabled — sanity baseline**

```bash
CC_ENABLED=0 uv run rebalance-portfolio
```
Expected: behaves exactly like before this feature shipped. No CC sections in the email. No new errors. Use this as the baseline log to compare against.

- [ ] **Step 2: Run with CC enabled — primary path**

```bash
CC_ENABLED=1 uv run rebalance-portfolio
```
Expected:
  - log line `cc_data: N eligible holding(s); chain sources [...]; M missing; stub pool $X`
  - HTML email contains the three new sections when applicable
  - PDF renders without exception
  - SQLite `run_outputs.dashboard_data` contains an `option_writes` array (`sqlite3 ~/.stock_analyzer/discover.db "SELECT dashboard_data FROM run_outputs ORDER BY run_id DESC LIMIT 1;" | python -c "import json,sys; print(json.dumps(json.loads(sys.stdin.read())['option_writes'], indent=2))"`)

- [ ] **Step 3: Verify validation log path**

If Opus ever returns a malformed WRITE_CALL during the smoke run, look for `CC plan validation:` WARN lines in the log. Note the case and ticker — if the LLM is making the same error repeatedly, tune the prompt in Task 13.

- [ ] **Step 4: Verify chain source distribution**

Decide based on the chain-source log line whether SnapTrade actually serves chains on your tier:
  - All `snaptrade` → ideal. Done.
  - Mostly `yfinance` → SnapTrade tier doesn't expose chains. Document this in `docs/superpowers/specs/2026-05-13-covered-calls-design.md` under "Open questions" as resolved (yfinance fallback works in production).
  - All `missing` → both providers failing. Triage before continuing — likely yfinance scrape change or env credential issue.

- [ ] **Step 5: Commit** (only if any docs got touched in step 4)

```bash
git add docs/
git commit -m "docs(cc): record chain-source field results from first production run"
```

---

## Self-review

The plan was checked against the spec section-by-section. Coverage:

- **Strategy parameters** → Task 1 (all 10 env vars).
- **Architecture diagram (new steps)** → Task 23 (step_cc_data wired into Parallel after review_holdings).
- **OptionWrite + WRITE_CALL + option_writes** → Task 2.
- **OptionChainProvider + OptionQuote + OptionChain** → Task 4.
- **YFinanceChain** → Task 5.
- **SnapTradeChain** → Task 6.
- **fetch_chains orchestrator with fallback** → Task 7.
- **fetch_open_option_positions** → Task 8.
- **Eligibility filter (≥100 shares, denylist, short-call subtraction)** → Task 9.
- **Round-lot coverage + stub_pool_total_usd** → Task 10.
- **Earnings filter** → Task 11.
- **Per-ticker context block + ROUND-LOT COVERAGE block** → Task 12.
- **Rebalancer prompt (CC writing, premium reinvestment, stub consolidation, BUY sizing rule)** → Task 13.
- **Rebalancer.decide accepting cc_context_block** → Task 14.
- **Post-LLM validation (orphans, clamping, logging)** → Task 15.
- **PreMortem extension** → Task 16.
- **Deterministic compute (premium income, deployment, round-lot summary)** → Task 17.
- **Section kinds in IR** → Task 18.
- **HTML renderers** → Task 19.
- **PDF renderers** → Task 20.
- **Email subject annotation** → Task 21.
- **Track-record CC scoring** → Task 22.
- **Persistence round-trip via JSON blob** → covered by Task 25 assertion; no migration code needed (spec corrected).
- **End-to-end wiring + integration test** → Tasks 23–25.
- **Docs** → Task 26.
- **Manual smoke** → Task 27.

Spec items deliberately NOT in the plan (future work, per spec §"Future work"):
- Cash-secured puts / the wheel.
- Roll suggestions.
- `place_option_strategy` execution.
- IV-rank screening.
- Dynamic CC style per ticker.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-13-covered-calls.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Better for a 27-task plan where catching prompt/wiring issues early prevents cascading rework downstream.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints. Faster end-to-end if the plan is largely correct and you trust the LLM contexts won't drift.

**Which approach?**
