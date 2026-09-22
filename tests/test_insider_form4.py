"""The weekly insider email when its news source is gone.

Tavily's quota ran out in September 2026 and every search failed; the
fetchers swallow their errors, so the email kept arriving with a quiet
"no data available". Filed Form 4s on the names you hold don't depend on
that quota, and an outage has to say so.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from stock_analyzer.cli import insider as cli
from stock_analyzer.config import Settings
from stock_analyzer.data import form4

TODAY = date(2026, 9, 21)


def _activity(*txs):
    return {"recent_transactions": list(txs)}


def _tx(d, code, shares, price, name="HUANG JEN HSUN"):
    return {"date": d, "code": code, "shares": shares, "price": price, "name": name}


def test_only_open_market_buys_and_sells_inside_the_window_are_kept():
    activity = {
        "NVDA": _activity(
            _tx("2026-09-18", "S", -10000, 120.0),
            _tx("2026-09-17", "M", 50000, 0.0),  # option exercise
            _tx("2026-09-16", "F", -2000, 118.0),  # tax withholding
            _tx("2026-09-01", "S", -5000, 110.0),  # outside the window
        ),
        "AVGO": _activity(_tx("2026-09-19", "P", 1000, 300.0, name="HOCK TAN")),
    }
    with (
        patch.object(form4.finnhub_data, "_client", return_value=object()),
        patch.object(
            form4.finnhub_data,
            "fetch_insider_activity",
            side_effect=lambda _c, t, days: activity.get(t, {}),
        ),
    ):
        trades = form4.fetch_form4_trades(["NVDA", "AVGO", "912345AB1"], days=7, today=TODAY)

    assert [(t["ticker"], t["side"]) for t in trades] == [("AVGO", "BUY"), ("NVDA", "SELL")]
    assert trades[1]["shares"] == 10000
    assert trades[1]["value_usd"] == 1_200_000
    assert trades[0]["name"] == "Hock Tan"


def test_no_finnhub_key_is_not_the_same_as_no_trades():
    with patch.object(form4.finnhub_data, "_client", return_value=None):
        assert form4.fetch_form4_trades(["NVDA"], days=7) is None


def test_the_section_says_when_nothing_was_filed():
    assert "No open-market insider buys or sells filed" in form4.form4_section([], days=7)


def _settings():
    return Settings(discover_watchlist=("NVDA",))


def test_a_news_outage_is_named_and_the_form4s_still_arrive():
    trade = {
        "ticker": "NVDA",
        "date": "2026-09-18",
        "name": "Jen-Hsun Huang",
        "side": "SELL",
        "shares": 10000.0,
        "price": 120.0,
        "value_usd": 1_200_000.0,
    }
    with (
        patch.object(cli, "fetch_political_trades", return_value=[]),
        patch.object(cli, "fetch_insider_trades", return_value=[]),
        patch.object(cli, "fetch_hedge_fund_trades", return_value=[]),
        patch.object(cli, "_my_tickers", return_value={"NVDA"}),
        patch.object(cli, "fetch_form4_trades", return_value=[trade]),
        patch.object(cli, "InsiderAgent") as agent,
    ):
        report = cli.run_analysis(_settings())

    agent.assert_not_called()  # nothing for the model to summarize
    assert report.startswith("===")
    assert "Tavily" in report
    assert "NVDA: Jen-Hsun Huang SELL 10,000 sh" in report


def test_every_source_failing_sends_nothing():
    with (
        patch.object(cli, "fetch_political_trades", return_value=[]),
        patch.object(cli, "fetch_insider_trades", return_value=[]),
        patch.object(cli, "fetch_hedge_fund_trades", return_value=[]),
        patch.object(cli, "_my_tickers", return_value=set()),
        patch.object(cli, "fetch_form4_trades", return_value=None),
    ):
        assert cli.run_analysis(_settings()) is None


def test_news_coverage_still_goes_through_the_model():
    item = {"title": "t", "link": "l", "snippet": "s", "politicians": ["Nancy Pelosi"]}
    with (
        patch.object(cli, "fetch_political_trades", return_value=[item]),
        patch.object(cli, "fetch_insider_trades", return_value=[]),
        patch.object(cli, "fetch_hedge_fund_trades", return_value=[]),
        patch.object(cli, "_my_tickers", return_value=set()),
        patch.object(cli, "fetch_form4_trades", return_value=[]),
        patch.object(cli, "InsiderAgent") as agent,
    ):
        agent.return_value.run.return_value = "=== HEADER ===\n\nNotable Congressional Trades:\n- x"
        report = cli.run_analysis(_settings())

    assert report.startswith("=== HEADER ===")
    assert "Form 4 Filings" in report


def test_the_prompt_is_dated_when_the_agent_is_built():
    from stock_analyzer.agents import insider as agents_insider

    assert "{today}" in agents_insider.INSIDER_INSTRUCTIONS
    agent = agents_insider.InsiderAgent("claude", "claude-haiku-4-5")
    assert "{today}" not in agent.agent.agent.instructions
