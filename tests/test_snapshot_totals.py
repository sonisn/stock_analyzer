"""The stored portfolio total: every dollar once, obligations included."""

from __future__ import annotations

from datetime import date

from stock_analyzer.cli.portfolio import snapshot_account_values
from stock_analyzer.reporting.health import cash_net_of_sweep
from stock_analyzer.reporting.performance import account_change_flows

IRA = "Traditional IRA"
HOLDINGS = {
    IRA: [
        {"ticker": "NVDA", "units": 100, "price": 200.0},
        {"ticker": "SPAXX", "units": 20845.9, "price": 1.0},
    ],
    "Robinhood Individual": [{"ticker": "BE", "units": 10, "price": 300.0}],
}


def test_a_sweep_reported_as_cash_too_is_counted_once():
    """Fidelity lists SPAXX as a position AND as the cash balance."""
    cash = cash_net_of_sweep(HOLDINGS, {IRA: 20845.9, "Robinhood Individual": 1.0})
    assert cash == {IRA: 0.0, "Robinhood Individual": 1.0}


def test_cash_on_top_of_the_sweep_is_kept():
    cash = cash_net_of_sweep(HOLDINGS, {IRA: 21845.9})
    assert cash == {IRA: 1000.0}


def test_a_separate_smaller_cash_balance_is_left_alone():
    """A broker whose cash excludes its money-market position."""
    cash = cash_net_of_sweep(HOLDINGS, {IRA: 500.0})
    assert cash == {IRA: 500.0}


def test_open_options_are_part_of_each_account_and_zero_is_written():
    values = snapshot_account_values(
        HOLDINGS,
        {IRA: 0.0, "Robinhood Individual": 1.0},
        options={IRA: -11746.0},
    )
    assert values[IRA] == {"value": 40845.9, "cash": 0.0, "options": -11746.0}
    assert values["Robinhood Individual"]["options"] == 0.0


def test_starting_to_value_options_is_not_a_loss():
    before = {IRA: {"value": 40845.9, "cash": 0.0}}
    after = {IRA: {"value": 40845.9, "cash": 0.0, "options": -11746.0}}
    assert account_change_flows([(date(2026, 9, 25), before), (date(2026, 9, 28), after)]) == [
        (date(2026, 9, 28), -11746.0)
    ]


def test_options_opened_and_closed_later_are_performance():
    """Once measured, a change in option value is a gain or a loss."""
    a = {IRA: {"value": 40845.9, "cash": 0.0, "options": 0.0}}
    b = {IRA: {"value": 40845.9, "cash": 1500.0, "options": -1500.0}}
    c = {IRA: {"value": 40845.9, "cash": 1500.0, "options": 0.0}}
    days = [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3)]
    assert account_change_flows(list(zip(days, [a, b, c], strict=True))) == []
