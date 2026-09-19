"""December tax planner (discover/tax_planner.py + reporting/tax_plan.py)."""

from __future__ import annotations

from datetime import date

import pytest

from stock_analyzer.cli.tax_planner import first_trading_day_of_december
from stock_analyzer.discover.tax_planner import (
    gains_turning_long_term,
    last_trading_day_of_year,
    plan_summary,
    realized_this_year,
)
from stock_analyzer.reporting.tax_plan import render_tax_plan_html


def _act(kind, ticker, day, units, price, fee=0.0):
    return {
        "type": kind,
        "symbol": {"symbol": ticker},
        "trade_date": f"{day}T00:00:00Z",
        "units": units if kind != "SELL" else -units,
        "price": price,
        "fee": fee,
    }


def test_fifo_realized_split_by_holding_period():
    acts = [
        _act("BUY", "X", "2025-01-10", 10, 100),  # long-term by the sale
        _act("REI", "X", "2026-03-01", 1, 150),  # short-term
        _act("SELL", "X", "2026-06-01", 11, 200),
        _act("BUY", "Y", "2025-06-01", 5, 50),
        _act("SELL", "Y", "2025-12-01", 5, 40),  # last year: not counted
        _act("SELL", "Z", "2026-07-01", 3, 10),  # transferred in: basis unknown
    ]
    r = realized_this_year(acts, year=2026)
    assert r["long_term"] == pytest.approx(10 * 100)
    assert r["short_term"] == pytest.approx(1 * 50)
    assert r["basis_unknown_units"] == 3
    assert [row["ticker"] for row in r["rows"]] == ["X", "Z"]


def test_selling_on_the_anniversary_is_short_term():
    acts = [_act("BUY", "X", "2025-06-01", 1, 100), _act("SELL", "X", "2026-06-01", 1, 130)]
    r = realized_this_year(acts, year=2026)
    assert (r["short_term"], r["long_term"]) == (pytest.approx(30), 0)


def test_gains_turning_long_term_soon():
    lots = {
        "X": {
            "lots": [
                {
                    "account": "Taxable",
                    "treatment": "short_term",
                    "long_term_on": "2026-12-20",
                    "units": 10,
                    "price_per_share": 100,
                },
                {
                    "account": "Taxable",
                    "treatment": "short_term",
                    "long_term_on": "2027-06-01",
                    "units": 10,
                    "price_per_share": 100,
                },  # too far out
                {
                    "account": "IRA",
                    "treatment": "short_term",
                    "long_term_on": "2026-12-20",
                    "units": 10,
                    "price_per_share": 100,
                },  # not taxable
            ]
        },
        "L": {
            "lots": [
                {
                    "account": "Taxable",
                    "treatment": "short_term",
                    "long_term_on": "2026-12-20",
                    "units": 5,
                    "price_per_share": 90,
                }
            ]
        },
    }
    soon = gains_turning_long_term(
        lots, {"X": 150.0, "L": 80.0}, {"Taxable"}, today=date(2026, 12, 1)
    )
    assert [(r["ticker"], r["gain"]) for r in soon] == [("X", 500.0)]
    assert soon[0]["tax_saved_by_waiting"] > 0


def test_plan_summary_offsets():
    realized = {
        "Taxable": {"short_term": 1000.0, "long_term": 500.0, "basis_unknown_units": 0, "rows": []}
    }
    harvest = [{"loss_usd": -6000.0, "est_tax_saving_usd": 1200.0}]
    s = plan_summary(realized, harvest)
    assert s["net_gain"] == 1500 and s["harvestable_loss"] == 6000
    assert (s["offsets_gains"], s["offsets_ordinary"], s["carry_forward"]) == (1500, 3000, 1500)


def test_calendar_and_render():
    assert first_trading_day_of_december(2026) == date(2026, 12, 1)
    assert first_trading_day_of_december(2029) == date(2029, 12, 3)  # Dec 1 is a Saturday
    assert last_trading_day_of_year(2027) == date(2027, 12, 31)
    assert last_trading_day_of_year(2022) == date(2022, 12, 30)  # Dec 31 was a Saturday
    body = render_tax_plan_html(
        year=2026,
        taxable_accounts=["Robinhood Individual"],
        realized={"Robinhood Individual": {"rows": []}},
        summary=plan_summary({}, [{"loss_usd": -2714.0, "est_tax_saving_usd": 489.0}]),
        harvest=[
            {
                "ticker": "POWL",
                "account": "Robinhood Individual",
                "loss_usd": -2714.0,
                "loss_pct": -26.5,
                "swap_candidates": ["ETN"],
                "wash_sale_until": None,
                "rebuy_ok_after": "2027-01-01",
            }
        ],
        soon=[],
        last_day=date(2026, 12, 31),
    )
    assert "Tax planner — 2026" in body and "POWL" in body and "ETN" in body
    assert "$2,714 offsets ordinary income" in body


def test_options_are_not_stock_lots_and_count_separately():
    opt = {"id": "c1", "ticker": "NFLX  260501C00095000"}
    acts = [
        {
            "type": "BUY",
            "symbol": {"symbol": "NFLX"},
            "option_symbol": opt,
            "units": 20,
            "price": 1.23,
            "amount": -2460.0,
            "trade_date": "2026-04-20T00:00:00Z",
        },
        {
            "type": "SELL",
            "symbol": {"symbol": "NFLX"},
            "option_symbol": opt,
            "units": -20,
            "price": 1.25,
            "amount": 2500.0,
            "trade_date": "2026-04-21T00:00:00Z",
        },
        # a short call that expired worthless: premium kept
        {
            "type": "SELL",
            "option_symbol": {"id": "c2"},
            "units": -1,
            "price": 9.35,
            "amount": 934.31,
            "trade_date": "2026-05-01T00:00:00Z",
        },
        {
            "type": "OPTIONEXPIRATION",
            "option_symbol": {"id": "c2"},
            "units": 1,
            "amount": 0,
            "trade_date": "2026-06-18T00:00:00Z",
        },
        # still open: not realized
        {
            "type": "SELL",
            "option_symbol": {"id": "c3"},
            "units": -1,
            "price": 2,
            "amount": 200.0,
            "trade_date": "2026-09-01T00:00:00Z",
        },
    ]
    r = realized_this_year(acts, year=2026)
    assert r["rows"] == []  # no NFLX "share" sales
    assert r["options"] == pytest.approx(40.0 + 934.31)
    assert r["short_term"] == pytest.approx(r["options"])


def test_option_trades_do_not_become_share_lots(monkeypatch):
    from unittest.mock import MagicMock

    from stock_analyzer.data import transactions

    client = MagicMock()
    client.account_information.list_user_accounts.return_value = [{"id": "a", "name": "RH"}]
    client.account_information.get_account_activities.return_value = [
        {
            "type": "BUY",
            "symbol": {"symbol": "NFLX"},
            "option_symbol": {"id": "c1"},
            "units": 20,
            "price": 1.23,
            "trade_date": "2026-04-20T00:00:00Z",
        },
        {
            "type": "BUY",
            "symbol": {"symbol": "NFLX"},
            "units": 2,
            "price": 95.0,
            "trade_date": "2026-04-20T00:00:00Z",
        },
    ]
    monkeypatch.setattr(transactions, "_credentials", lambda: ("u", "s"))
    monkeypatch.setattr(transactions, "_client", lambda: client)
    lots = transactions.fetch_transaction_history(years_back=3)["NFLX"].lots
    assert [(lot.units, lot.price) for lot in lots] == [(2.0, 95.0)]
