"""Asset location and the goal projection: the arithmetic behind the plan check."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_analyzer.data.brokerage import classify_account_kind
from stock_analyzer.discover import asset_location as al
from stock_analyzer.discover import goal_projection as gp

LT, ST = 0.15, 0.32
KINDS = {"Brokerage": "taxable", "Traditional IRA": "tax_deferred", "HSA": "tax_free"}


def _row(ticker, units, price, cost):
    return {"ticker": ticker, "units": units, "price": price, "average_purchase_price": cost}


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("Robinhood Individual", "taxable"),
        ("Traditional IRA", "tax_deferred"),
        ("Broadcom U.S. 401(k) Plan", "tax_deferred"),
        ("HSA Brokerage ...263", "tax_free"),
        ("Fidelity Roth IRA", "tax_free"),
    ],
)
def test_account_kinds(name, kind):
    assert classify_account_kind(None, name) == kind


# --- asset location -----------------------------------------------------------


def test_drag_counts_dividends_at_their_rate_and_premium_as_short_term():
    qualified = al.yearly_drag(
        10_000, al.TickerTaxFacts(0.04, False), 0, long_term_rate=LT, short_term_rate=ST
    )
    reit = al.yearly_drag(
        10_000, al.TickerTaxFacts(0.04, True), 0, long_term_rate=LT, short_term_rate=ST
    )
    calls = al.yearly_drag(
        10_000, al.TickerTaxFacts(0.0, False), 1_000, long_term_rate=LT, short_term_rate=ST
    )
    assert qualified == pytest.approx(60.0)
    assert reit == pytest.approx(128.0)
    assert calls == pytest.approx(320.0)


def test_reit_detection():
    assert al.ordinary_dividends("Real Estate", "REIT - Industrial")
    assert al.ordinary_dividends(None, "REIT—Specialty")
    assert not al.ordinary_dividends("Technology", "Semiconductors")


def test_trailing_yield_from_bars():
    idx = pd.date_range(
        end=pd.Timestamp("2026-09-25"), periods=400, freq="D", tz="America/New_York"
    )
    bars = pd.DataFrame({"Close": 50.0, "Dividends": 0.0}, index=idx)
    for d in ("2026-01-15", "2026-04-15", "2026-07-15", "2025-07-15"):
        bars.loc[pd.Timestamp(d, tz="America/New_York"), "Dividends"] = 0.5
    # three payments inside the last year: 1.50 / 50
    assert al.trailing_yield(bars, today=date(2026, 9, 25)) == pytest.approx(0.03)
    assert al.trailing_yield(None) == 0.0


def _report(holdings, facts, premium=None, **kw):
    return al.analyze(
        holdings,
        KINDS,
        facts,
        premium or {},
        long_term_rate=LT,
        short_term_rate=ST,
        today=date(2026, 9, 26),
        **kw,
    )


def test_swap_moves_the_income_stock_into_the_ira_and_names_both_legs():
    holdings = {
        "Brokerage": [_row("O", 500, 60, 58)],  # $30k REIT, small gain
        "Traditional IRA": [_row("NVDA", 300, 180, 50)],  # $54k, no yield
    }
    facts = {"O": al.TickerTaxFacts(0.055, True), "NVDA": al.TickerTaxFacts(0.0003, False)}
    report = _report(holdings, facts)
    [swap] = report.swaps
    assert (swap.inefficient, swap.efficient) == ("O", "NVDA")
    assert (swap.taxable_account, swap.into_account) == ("Brokerage", "Traditional IRA")
    assert swap.amount == pytest.approx(30_000)
    assert swap.tax_on_sale == pytest.approx(1_000 * LT)
    assert swap.yearly_saving > 400
    assert swap.breakeven_years < 1
    assert swap.wash_sale_until is None


def test_no_swap_when_the_sale_tax_takes_too_long_to_earn_back():
    holdings = {
        "Brokerage": [_row("KO", 400, 70, 10)],  # $28k, huge gain
        "Traditional IRA": [_row("NVDA", 300, 180, 50)],
    }
    facts = {"KO": al.TickerTaxFacts(0.03, False), "NVDA": al.TickerTaxFacts(0.0, False)}
    report = _report(holdings, facts)
    assert report.swaps == []
    assert report.taxable_drag == pytest.approx(28_000 * 0.03 * LT)


def test_a_loss_leg_carries_the_wash_sale_date():
    holdings = {
        "Brokerage": [_row("O", 500, 60, 70)],  # at a loss
        "Traditional IRA": [_row("NVDA", 300, 180, 50)],
    }
    facts = {"O": al.TickerTaxFacts(0.055, True)}
    [swap] = _report(holdings, facts).swaps
    assert swap.realized_gain < 0 and swap.tax_on_sale == 0
    assert swap.wash_sale_until == date(2026, 9, 26) + timedelta(days=31)


def test_small_drag_is_left_alone():
    holdings = {
        "Brokerage": [_row("AAPL", 100, 250, 100)],
        "Traditional IRA": [_row("NVDA", 300, 180, 50)],
    }
    facts = {"AAPL": al.TickerTaxFacts(0.004, False)}
    assert _report(holdings, facts).swaps == []


def test_premium_written_in_taxable_on_a_stock_the_ira_holds():
    holdings = {
        "Brokerage": [_row("TSLA", 100, 400, 300)],
        "Traditional IRA": [_row("TSLA", 300, 400, 200)],
    }
    premium = {("Brokerage", "TSLA"): 6_000.0, ("Traditional IRA", "TSLA"): 9_000.0}
    report = _report(holdings, {}, premium)
    assert report.options_elsewhere == {"TSLA": (6_000.0, "Traditional IRA")}
    brokerage = next(p for p in report.placements if p.account == "Brokerage")
    assert brokerage.drag_if_taxable == pytest.approx(6_000 * ST)


def test_premium_by_account_counts_only_short_options_in_the_window():
    since = date(2025, 9, 26)
    rows = [
        SimpleNamespace(
            account="B",
            underlying="tsla",
            opened_short=True,
            day_trade=False,
            first_day=date(2026, 1, 5),
            net_premium=500.0,
        ),
        SimpleNamespace(
            account="B",
            underlying="TSLA",
            opened_short=True,
            day_trade=False,
            first_day=date(2025, 1, 5),
            net_premium=900.0,
        ),  # too old
        SimpleNamespace(
            account="B",
            underlying="NFLX",
            opened_short=False,
            day_trade=False,
            first_day=date(2026, 2, 1),
            net_premium=-3000.0,
        ),  # a long
    ]
    assert al.option_premium_by_account(rows, since=since) == {("B", "TSLA"): 500.0}


# --- goal projection ----------------------------------------------------------


def _returns(months=180, vol_a=0.03, vol_spy=0.04, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2011-01-31", periods=months, freq="ME")
    return pd.DataFrame(
        {
            "SPY": rng.normal(0.008, vol_spy, months),
            "AAA": rng.normal(0.03, vol_a * 3, months),  # a hot stock: high mean, high vol
            "YNG": [np.nan] * (months - min(24, months))
            + list(rng.normal(0.01, 0.05, min(24, months))),
        },
        index=idx,
    )


def test_projection_centres_on_the_assumed_return_not_the_history():
    r = _returns()
    p = gp.project(
        weights={"AAA": 1.0},
        returns=r,
        start_value=100_000,
        months=60,
        monthly_contribution=0,
        expected_return=0.07,
    )
    # AAA's history averages ~40%/yr; the median must reflect ~7%, less
    # volatility drag, not that.
    assert 100_000 < p.p50 < 100_000 * 1.07**5
    assert p.p10 < p.p50 < p.p90


def test_concentration_widens_the_range_and_costs_odds():
    r = _returns()
    kw = dict(
        returns=r,
        start_value=100_000,
        months=60,
        monthly_contribution=500,
        expected_return=0.07,
        target=150_000,
    )
    hot = gp.project(weights={"AAA": 1.0}, **kw)
    assert hot.annual_volatility > hot.spy_annual_volatility
    assert hot.p90 - hot.p10 > 0
    assert hot.odds < hot.spy_odds
    assert hot.deep_drawdown_odds > 0


def test_needed_contribution_really_gives_the_target_odds():
    r = _returns()
    kw = dict(
        weights={"AAA": 0.5, "SPY": 0.5},
        returns=r,
        start_value=100_000,
        months=60,
        expected_return=0.07,
        target=200_000,
    )
    first = gp.project(monthly_contribution=0, **kw)
    assert first.odds < gp.TARGET_ODDS
    again = gp.project(monthly_contribution=first.needed_contribution + 1, **kw)
    assert again.odds >= gp.TARGET_ODDS - 0.005


def test_young_holdings_are_filled_with_spy_and_named():
    p = gp.project(
        weights={"YNG": 1.0},
        returns=_returns(),
        start_value=10_000,
        months=12,
        monthly_contribution=0,
        expected_return=0.07,
    )
    assert p.filled_from_spy == ("YNG",)


def test_too_little_history_gives_no_projection():
    assert (
        gp.project(
            weights={"SPY": 1.0},
            returns=_returns(months=20),
            start_value=1,
            months=12,
            monthly_contribution=0,
            expected_return=0.07,
        )
        is None
    )


def test_monthly_returns_drops_the_partial_month():
    idx = pd.date_range(end=pd.Timestamp.today(), periods=200, freq="D", tz="America/New_York")
    bars = {"SPY": pd.DataFrame({"Close": np.linspace(100, 120, 200)}, index=idx)}
    m = gp.monthly_returns(bars)
    assert m.index[-1] < pd.Timestamp.today().normalize().replace(day=1)


# --- rendering ------------------------------------------------------------------


def test_sections_render_decision_first():
    from stock_analyzer.reporting.plan_check import (
        asset_location_headline,
        goal_headline,
        render_asset_location_html,
        render_goal_html,
    )

    holdings = {
        "Brokerage": [_row("O", 500, 60, 70)],
        "Traditional IRA": [_row("NVDA", 300, 180, 50)],
    }
    report = _report(holdings, {"O": al.TickerTaxFacts(0.055, True)})
    page = render_asset_location_html(report)
    assert "sell O, buy NVDA" in page and "Sold at a loss" in page
    assert asset_location_headline(report).startswith("1 account swap(s)")

    p = gp.project(
        weights={"AAA": 1.0},
        returns=_returns(),
        start_value=100_000,
        months=60,
        monthly_contribution=0,
        expected_return=0.07,
        target=500_000,
    )
    html_out = render_goal_html(p, goal_date=date(2031, 9, 1), contribution_note="test")
    assert "OFF TRACK" in html_out and "/month" in html_out
    assert goal_headline(p, date(2031, 9, 1)).endswith("by Sep 2031")
    assert "Not enough" in render_goal_html(None, goal_date=None, contribution_note="")
    assert render_asset_location_html(None) == ""


def test_typical_month_ignores_one_off_lumps():
    from stock_analyzer.cli.plan_check import typical_monthly

    flows = [
        (date(2026, 4, 23), 50_000.0),  # one-off
        (date(2026, 5, 1), 500.0),
        (date(2026, 6, 1), 500.0),
        (date(2026, 7, 1), 500.0),
        (date(2026, 7, 14), 36_399.0),  # a rollover
        (date(2026, 8, 1), 500.0),
        (date(2026, 9, 1), 500.0),  # current month: incomplete, not counted
    ]
    assert typical_monthly(flows, date(2026, 9, 26)) == 500.0
    assert typical_monthly([], date(2026, 9, 26)) == 0.0


def test_money_market_funds_are_cash():
    idx = pd.date_range(end=pd.Timestamp("2026-09-25"), periods=3, freq="D")
    assert gp.is_cash_like(pd.DataFrame({"Close": [1.0, 1.0, 1.0]}, index=idx))
    assert not gp.is_cash_like(pd.DataFrame({"Close": [1.0, 1.1, 0.9]}, index=idx))
    assert not gp.is_cash_like(pd.DataFrame({"Close": [50.0, 51.0, 52.0]}, index=idx))
    assert not gp.is_cash_like(None)
