"""The covered-call data step end to end, offline: who is eligible, who is
held back for cheap premium, and what completing a part-lot would earn."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import patch

import pytest

from stock_analyzer.config import Settings
from stock_analyzer.discover import rebalance_cc
from stock_analyzer.models.market import OptionChain, OptionQuote, RealizedVolatility

EXP = date.today() + timedelta(days=38)


def _q(strike, delta, bid=4.0):
    return OptionQuote(
        strike=strike,
        expiry=EXP,
        bid=bid,
        ask=bid + 0.2,
        iv=0.45,
        delta=delta,
        open_interest=500,
        volume=50,
    )


def _chain(ticker, spot, calls):
    return OptionChain(
        ticker=ticker, spot=spot, asof=datetime(2026, 9, 22, 10), calls=calls, source="tradier"
    )


CHAINS = {
    "NVDA": _chain("NVDA", 120.0, [_q(135.0, 0.38), _q(140.0, 0.30, 2.5)]),
    "AVGO": _chain("AVGO", 300.0, [_q(330.0, 0.40, 9.0)]),
    "TSLA": _chain("TSLA", 250.0, [_q(280.0, 0.36, 7.0)]),
}
# AVGO's options (IV 45%) are cheap next to how much it actually moves.
HV = {"NVDA": 0.30, "AVGO": 0.60, "TSLA": 0.40}


def _split(account, status, units):
    return {"splits": [{"account": account, "tax_status": status, "units": units}]}


STATE = {
    "holdings_positions": {
        "NVDA": {"units": 250.0},
        "AVGO": {"units": 150.0},
        "TSLA": {"units": 100.0},
        "SPAXX": {"units": 5000.0},
    },
    "holdings_technicals": {
        "NVDA": {"price": 120.0},
        "AVGO": {"price": 300.0},
        "TSLA": {"price": 250.0},
    },
    "position_splits": {
        "NVDA": _split("IRA", "tax_deferred", 250.0),
        "AVGO": _split("Brokerage", "taxable", 150.0),
        "TSLA": _split("Brokerage", "taxable", 100.0),
    },
    "finnhub_signals": {},
    "holdings_reviews": {},
}


def _run(open_calls):
    with (
        patch("stock_analyzer.data.brokerage.fetch_open_option_positions", return_value=open_calls),
        patch(
            "stock_analyzer.data.options_chain.fetch_chains",
            side_effect=lambda tickers, **k: {t: CHAINS[t] for t in tickers if t in CHAINS},
        ),
        patch(
            "stock_analyzer.data.historical_volatility.fetch_realized_volatility",
            side_effect=lambda tickers, **k: {
                t: RealizedVolatility(ticker=t, hv_annualized=HV[t], sample_size=250)
                for t in tickers
            },
        ),
    ):
        return rebalance_cc.run_cc_data_pipeline(STATE, Settings())


@pytest.mark.parametrize(
    ("open_calls", "eligible"),
    [({}, ["NVDA", "TSLA"]), ({"TSLA": {"Brokerage": 1}}, ["NVDA"])],
)
def test_eligibility_skips_shares_already_backing_a_call(open_calls, eligible):
    assert sorted(_run(open_calls).eligibility) == eligible


def test_cheap_premium_holds_a_writer_back_and_says_why():
    result = _run({})
    assert "AVGO" not in result.eligibility
    assert "0.75x its realized 60%" in result.cheap_premium["AVGO"]


def test_part_lots_and_cash_are_counted_apart():
    result = _run({})
    # 50-share stubs on NVDA ($6,000) and AVGO ($15,000); SPAXX is cash.
    assert result.stub_pool == 21_000.0
    assert "SPAXX" not in result.coverage
    assert "NVDA" in result.context_block
    assert result.content.startswith("cc_data: 2 eligible holding(s)")
