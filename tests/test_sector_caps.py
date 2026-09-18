"""Hard sector caps applied to the Sizer's allocations."""

from __future__ import annotations

import pytest

from stock_analyzer.cli.discover import _holdings_value_by_sector
from stock_analyzer.discover.sizer import enforce_sector_caps, format_sector_exposure_block
from stock_analyzer.models.llm import Allocation, SizerOutput


def _alloc(ticker: str, *, pct: float | None = None, usd: float | None = None) -> Allocation:
    return Allocation(ticker=ticker, allocation_pct=pct, allocation_usd=usd, rationale="r")


def _output(allocations: list[Allocation]) -> SizerOutput:
    return SizerOutput(allocations=allocations, concentration_warnings=[], full_text="x")


def _by_ticker(out: SizerOutput) -> dict[str, Allocation]:
    return {a.ticker: a for a in out.allocations}


SECTORS = {"NVDA": "Technology", "AMD": "Technology", "JPM": "Financial Services"}


def test_new_capital_cap_scales_sector_proportionally():
    out = enforce_sector_caps(
        _output([_alloc("NVDA", pct=40), _alloc("AMD", pct=20), _alloc("JPM", pct=40)]),
        SECTORS,
        {},
        max_new_pct=50,
    )
    a = _by_ticker(out)
    assert a["NVDA"].allocation_pct == pytest.approx(50 * 40 / 60)
    assert a["AMD"].allocation_pct == pytest.approx(50 * 20 / 60)
    assert a["JPM"].allocation_pct == 40
    assert out.concentration_warnings[0].startswith("SECTOR CAP: Technology picks (AMD + NVDA)")


def test_book_cap_uses_existing_holdings_with_a_budget():
    # Book = 90k holdings + 10k budget = 100k; Technology already 25k, so
    # only 5k (50% of the budget) fits under a 30% book cap.
    out = enforce_sector_caps(
        _output([_alloc("NVDA", usd=6000), _alloc("JPM", usd=4000)]),
        SECTORS,
        {"Technology": 25_000, "Energy": 65_000},
        cash_budget=10_000,
        max_book_pct=30,
        max_new_pct=80,
    )
    a = _by_ticker(out)
    assert a["NVDA"].allocation_usd == pytest.approx(5000)
    assert a["JPM"].allocation_usd == 4000
    assert "already 25.0% of that book" in out.concentration_warnings[0]


def test_sector_already_over_book_cap_gets_no_new_money():
    out = enforce_sector_caps(
        _output([_alloc("NVDA", usd=5000), _alloc("JPM", usd=5000)]),
        SECTORS,
        {"Technology": 50_000, "Energy": 40_000},
        cash_budget=10_000,
    )
    assert _by_ticker(out)["NVDA"].allocation_usd == 0


def test_without_budget_an_overweight_sector_only_warns():
    out = enforce_sector_caps(
        _output([_alloc("NVDA", pct=30), _alloc("JPM", pct=70)]),
        SECTORS,
        {"Technology": 60_000, "Energy": 40_000},
        max_new_pct=100,
    )
    assert _by_ticker(out)["NVDA"].allocation_pct == 30
    assert out.concentration_warnings == [
        "SECTOR CONCENTRATION: Technology is already 60% of current holdings and the "
        "picks add 30.0% of new capital to it — set DISCOVER_CASH_BUDGET to enforce "
        "the 30% book cap"
    ]


def test_within_caps_and_unknown_sectors_are_untouched():
    original = _output([_alloc("NVDA", pct=30), _alloc("XYZ", pct=70)])
    assert enforce_sector_caps(original, SECTORS, {}) is original


def test_holdings_value_by_sector_and_prompt_block():
    holdings = {
        "Brokerage": [
            {"ticker": "aapl", "units": 10, "price": 200.0},
            {"ticker": "JPM", "units": 5, "price": 100.0},
            {"ticker": "ODD", "units": 5, "price": 10.0},  # no known sector
        ],
        "IRA": [{"ticker": "AAPL", "units": 5, "price": 200.0}],
    }
    values = _holdings_value_by_sector(
        holdings, {"AAPL": "Technology", "JPM": "Financial Services"}
    )
    assert values == {"Technology": 3000.0, "Financial Services": 500.0}
    block = format_sector_exposure_block(
        {"NVDA": "Technology"}, values, max_book_pct=30, max_new_pct=50
    )
    assert "Technology: picks NVDA; 86% of current holdings" in block
