"""One price per ticker, and what the fix protects.

The case these come from: on 2026-09-19 the HSA reported BE at $298.61
while the live quote and the other two accounts said $265.63 — a $2,408
overstatement on 73 shares, which fed the sector weights, the add-on
sizing and the snapshot the quarterly vs-SPY return is built from.
"""

from __future__ import annotations

from stock_analyzer.data.pricing import quotes_from_ticker_data, reconcile_prices
from stock_analyzer.reporting.health import aggregate_positions, build_portfolio_health

HOLDINGS = {
    "Robinhood Individual": [
        {"ticker": "BE", "units": 10.886, "price": 265.57, "average_purchase_price": 280.09}
    ],
    "Traditional IRA": [
        {"ticker": "BE", "units": 300.471, "price": 265.63, "average_purchase_price": 237.20}
    ],
    "HSA Brokerage ...263": [
        {"ticker": "BE", "units": 73.0, "price": 298.61, "average_purchase_price": 123.25}
    ],
}
LIVE = {"BE": 265.63}


def test_stale_account_price_does_not_inflate_the_value():
    prices, notes = reconcile_prices(HOLDINGS, LIVE)
    assert prices["BE"] == 265.63
    assert any("HSA" in n and "BE" in n for n in notes)

    stale = sum(p["value"] for p in aggregate_positions(HOLDINGS).values())
    fixed = sum(p["value"] for p in aggregate_positions(HOLDINGS, prices).values())
    # 73 HSA shares overstated by $32.98 each, less a 6c correction on the
    # Robinhood slice, which was a hair low.
    assert round(stale - fixed) == 2407


def test_prices_within_tolerance_are_not_flagged():
    _, notes = reconcile_prices(
        {"A": [{"ticker": "BE", "units": 1, "price": 265.57}]}, {"BE": 265.63}
    )
    assert notes == []


def test_without_a_quote_accounts_are_averaged_not_left_to_disagree():
    """One holding must not be worth two amounts in the same report."""
    prices, notes = reconcile_prices(HOLDINGS, {})
    assert 265.57 < prices["BE"] < 298.61
    assert notes == []  # nothing to compare against, so nothing to claim


def test_quotes_come_from_what_the_run_already_fetched():
    assert quotes_from_ticker_data(
        {"BE": {"price_value": 265.63}, "SPAXX": {"price_value": None}, "X": {}}
    ) == {"BE": 265.63}


def test_a_position_with_no_cost_basis_is_not_counted_as_pure_profit():
    holdings = {
        "IRA": [
            {"ticker": "NVDA", "units": 10, "price": 200.0, "average_purchase_price": 150.0},
            # Transferred in with no cost basis: worth $20k, cost unknown.
            {"ticker": "SPAXX", "units": 20_000, "price": 1.0, "average_purchase_price": 0},
        ]
    }
    h = build_portfolio_health(holdings)
    assert h.snapshot["value"] == 22_000  # still counted as portfolio value
    assert h.snapshot["unrealized"] == 500  # only NVDA's gain, not SPAXX's $20k
    assert round(h.snapshot["unrealized_pct"], 2) == 33.33
    assert any("SPAXX" in note for note in h.data_notes)


def test_price_notes_reach_the_email():
    from stock_analyzer.reporting.health import render_health_html

    h = build_portfolio_health(HOLDINGS, prices={"BE": 265.63}, data_notes=["BE priced +12% off"])
    assert "BE priced +12% off" in render_health_html(h)
