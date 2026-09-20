"""Filed fundamentals, and what happens when they disagree with yfinance.

Measured on 2026-09-20: yfinance put NVDA's trailing free cash flow at
$41.8B against $127.0B in the filings, and read ANET's 2025-12-31
`NetIncomeLoss` as -$2,556M, which turns a 38% net margin into 5%.
"""

from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.data import fundamentals as fund
from stock_analyzer.discover.data_reconciliation import (
    flag_non_usd_fundamentals,
    reconcile_fundamentals,
)


def test_agreeing_sources_say_nothing():
    warnings, rejected = reconcile_fundamentals(
        {"gross_margin": 0.630, "free_cash_flow": 5.0e9},
        {"gross_margin": 0.630, "free_cash_flow": 5.1e9},
    )
    assert warnings == [] and rejected == set()


def test_a_wide_but_believable_gap_prefers_the_filing():
    # AVGO: GAAP 68.8% as filed vs yfinance's adjusted 75.5%.
    warnings, rejected = reconcile_fundamentals({"gross_margin": 0.755}, {"gross_margin": 0.688})
    assert len(warnings) == 1 and "using the filed figure" in warnings[0]
    assert rejected == set()  # the filed value is still the one taken


def test_an_unreconcilable_gap_keeps_yfinance():
    # ANET's mis-extracted quarter: 38.4% vs 5.1% is not GAAP vs adjusted.
    warnings, rejected = reconcile_fundamentals({"profit_margin": 0.384}, {"profit_margin": 0.051})
    assert rejected == {"profit_margin"}
    assert "cannot be reconciled" in warnings[0]


def test_a_sign_flip_is_never_silently_adopted():
    warnings, rejected = reconcile_fundamentals(
        {"free_cash_flow": 5.0e9}, {"free_cash_flow": -5.0e9}
    )
    assert rejected == {"free_cash_flow"} and "opposite signs" in warnings[0]


def test_ratios_and_amounts_are_formatted_differently():
    # "debt to equity 0 (yfinance) vs 0 (as filed)" is what a shared
    # money formatter printed for GOOGL's 0.40 vs 0.19.
    warnings, _ = reconcile_fundamentals({"debt_to_equity": 0.40}, {"debt_to_equity": 0.19})
    assert "0.40" in warnings[0] and "0.19" in warnings[0]
    warnings, _ = reconcile_fundamentals({"free_cash_flow": 2.3e10}, {"free_cash_flow": 5.3e10})
    assert "23,000,000,000" in warnings[0]


def test_a_foreign_issuers_balance_sheet_is_flagged():
    assert flag_non_usd_fundamentals("TSM", {"debt_to_equity": 42.16}) is not None
    assert "currency" in flag_non_usd_fundamentals("TSM", {"debt_to_equity": 42.16})
    assert flag_non_usd_fundamentals("NVDA", {"debt_to_equity": 0.17}) is None
    assert flag_non_usd_fundamentals("X", {}) is None


# --- the overlay ------------------------------------------------------------------


@pytest.fixture
def filed(monkeypatch):
    """Stub the API: {ticker: filed values} in, overlay applied."""

    def _install(values, *, configured=True):
        from stock_analyzer.data import wisesheets

        monkeypatch.setattr(wisesheets, "is_configured", lambda: configured)
        monkeypatch.setattr(
            wisesheets, "fetch_trailing_fundamentals", lambda tickers, as_of=None: values
        )

    return _install


def test_filed_values_replace_the_derived_ones(filed):
    filed({"NVDA": {"free_cash_flow": 127.0e9, "period_end": date.today().isoformat()}})
    rows = {"NVDA": {"free_cash_flow": 41.8e9, "market_cap": 5.0e12}}
    fund._overlay_filed_values(rows)
    assert rows["NVDA"]["free_cash_flow"] == 127.0e9
    assert rows["NVDA"]["filed_fields"] == ["free_cash_flow"]
    # the derived yield has to follow the value it was derived from
    assert rows["NVDA"]["fcf_yield"] == pytest.approx(127.0e9 / 5.0e12)


def test_a_stale_filing_is_left_alone(filed):
    # BE's newest quarter was 2026-03-31 on 2026-09-20 — 173 days.
    filed({"BE": {"free_cash_flow": 2.0e8, "period_end": "2026-03-31"}})
    rows = {"BE": {"free_cash_flow": 5.3e8}}
    fund._overlay_filed_values(rows, as_of=date(2026, 9, 20))
    assert rows["BE"]["free_cash_flow"] == 5.3e8
    assert "173d old" in rows["BE"]["filed_note"]
    assert "filed_fields" not in rows["BE"]


def test_an_uncovered_ticker_is_untouched(filed):
    filed({})  # TSM is a foreign private issuer: no SEC XBRL
    rows = {"TSM": {"free_cash_flow": 7.3e11, "debt_to_equity": 42.16}}
    fund._overlay_filed_values(rows)
    assert rows["TSM"]["free_cash_flow"] == 7.3e11
    assert any("currency" in f for f in rows["TSM"]["data_reconciliation_flags"])


def test_without_a_key_nothing_changes(filed):
    filed({"NVDA": {"free_cash_flow": 127.0e9}}, configured=False)
    rows = {"NVDA": {"free_cash_flow": 41.8e9}}
    fund._overlay_filed_values(rows)
    assert rows["NVDA"] == {"free_cash_flow": 41.8e9}


def test_an_api_failure_leaves_the_run_intact(filed, monkeypatch):
    from stock_analyzer.data import wisesheets

    monkeypatch.setattr(wisesheets, "is_configured", lambda: True)

    def boom(tickers, as_of=None):
        raise RuntimeError("wisesheets down")

    monkeypatch.setattr(wisesheets, "fetch_trailing_fundamentals", boom)
    rows = {"NVDA": {"free_cash_flow": 41.8e9}}
    fund._overlay_filed_values(rows)
    assert rows["NVDA"]["free_cash_flow"] == 41.8e9
