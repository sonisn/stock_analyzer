"""Premium written, told apart from option trading.

Measured on the stored ledger 2026-09-20: a called-away TSLA position
(100 shares at $410) graded as a sale into strength, with the premium
that paid for it invisible. And the ledger mixes wheel income with
speculation — one NFLX long call lost $8,900 — which cannot be summed.
"""

from __future__ import annotations

from datetime import date

from stock_analyzer.data.options_income import summarize_contracts, summarize_option_income


def _act(day, kind, units, amount, symbol):
    return {
        "trade_date": day,
        "type": kind,
        "units": units,
        "amount": amount,
        "option_symbol": symbol,
    }


CALL = "NVDA  261218C00285000"


def test_a_call_sold_then_expiring_keeps_every_dollar():
    rows = {
        "IRA": [
            _act("2026-05-26", "SELL", -3, 3012.91, CALL),
            _act("2026-12-18", "OPTIONEXPIRATION", 3, 0.0, CALL),
        ]
    }
    income = summarize_option_income(rows)
    assert income.premium_collected == 3012.91
    assert income.net_premium == 3012.91
    assert income.expired == 1 and income.assigned == 0
    assert income.by_underlying["NVDA"]["contracts"] == 3


def test_selling_the_same_contract_twice_counts_the_whole_short():
    # NVDA's 2026-12-18 $285 calls were sold -3 and then -1: four short.
    rows = {
        "IRA": [
            _act("2026-05-26", "SELL", -3, 3012.91, CALL),
            _act("2026-05-26", "SELL", -1, 934.31, CALL),
        ]
    }
    [contract] = summarize_contracts(rows)
    assert contract.contracts == 4
    assert contract.outcome == "open"


def test_assignment_reports_the_shares_that_left():
    sym = "TSLA  260520C00410000"
    rows = {
        "IRA": [
            _act("2026-04-20", "SELL", -1, 1200.0, sym),
            _act("2026-05-21", "OPTIONASSIGNMENT", 1, 0.0, sym),
        ]
    }
    income = summarize_option_income(rows)
    assert income.assigned == 1
    assert income.shares_called_away == 100
    assert income.by_underlying["TSLA"]["net_premium"] == 1200.0
    assert income.assignments[0].strike == 410.0


def test_buying_a_call_is_never_premium_income():
    sym = "NFLX  260424C00092000"
    rows = {
        "RH": [
            _act("2026-04-21", "BUY", 100, -28600.0, sym),
            _act("2026-04-23", "SELL", -100, 19700.0, sym),
        ]
    }
    income = summarize_option_income(rows)
    assert income.contracts_sold == 0
    assert income.premium_collected == 0
    assert len(income.long_positions) == 1
    assert income.long_positions[0].net_premium == -8900.0


def test_the_same_contract_sold_first_is_a_short_even_when_bought_back():
    sym = "NFLX  260424C00091000"
    rows = {
        "RH": [
            _act("2026-04-21", "SELL", -100, 35600.0, sym),
            _act("2026-04-23", "BUY", 100, -28700.0, sym),
        ]
    }
    income = summarize_option_income(rows)
    assert income.net_premium == 6900.0
    assert income.bought_back == 1 and income.long_positions == []


def test_a_same_day_round_trip_is_not_income():
    # Open and close on one day: the ledger cannot say which came first,
    # and either way it is a trade, not premium earned on a holding.
    sym = "NFLX  260424P00094000"
    rows = {
        "RH": [
            _act("2026-04-21", "BUY", 8, -968.0, sym),
            _act("2026-04-21", "SELL", -8, 1456.0, sym),
        ]
    }
    income = summarize_option_income(rows)
    assert income.contracts_sold == 0 and income.premium_collected == 0
    assert len(income.day_trades) == 1
    assert income.day_trades[0].net_premium == 488.0


def test_the_window_keeps_a_contract_whole():
    # Sold in May, assigned in September: one position, not two halves.
    sym = "TSLA  260520C00410000"
    rows = {
        "IRA": [
            _act("2026-04-20", "SELL", -1, 1200.0, sym),
            _act("2026-05-21", "OPTIONASSIGNMENT", 1, 0.0, sym),
        ]
    }
    income = summarize_option_income(rows, start=date(2026, 4, 1), end=date(2026, 4, 30))
    # Only the sale falls in April, so the assignment is not yet known.
    assert income.assigned == 0 and income.net_premium == 1200.0
    whole = summarize_option_income(rows, start=date(2026, 4, 1), end=date(2026, 6, 30))
    assert whole.assigned == 1


def test_share_trades_are_ignored():
    rows = {"RH": [{"trade_date": "2026-05-01", "type": "BUY", "units": 10, "amount": -1000.0}]}
    assert summarize_option_income(rows).contracts_sold == 0
