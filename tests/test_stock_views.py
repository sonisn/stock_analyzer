"""Daily per-stock blocks with reused long-term views, and the
post-earnings check."""

from __future__ import annotations

import sqlite3
from copy import deepcopy
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from stock_analyzer.agents import portfolio as portfolio_agent
from stock_analyzer.agents.portfolio import PortfolioAgent
from stock_analyzer.agents.stock_views import format_ticker_block, refresh_reason
from stock_analyzer.db.tables import StockView
from stock_analyzer.discover.post_earnings import recent_results, result_text, with_revisions
from stock_analyzer.discover.stock_facts import company_facts
from stock_analyzer.reporting.health import build_portfolio_health, decision_items
from stock_analyzer.reporting.html import _parse

TODAY = date(2026, 9, 18)


def _data(price=100.0, earnings_day="2026-06-03"):
    return {
        "symbol": "AVGO",
        "name": "Broadcom Inc.",
        "price": f"${price:,.2f}",
        "price_value": price,
        "pct_today": "+1.20%",
        "market_cap": "$1.6T",
        "range_52w": "$138.10 - $374.23",
        "pe": "70.1",
        "analyst_target": "$400.00",
        "analysts": {"strongBuy": 10, "buy": 20, "hold": 5, "sell": 1},
        "trend_7days": "Up (+2.1%)",
        "trend_1yr": "Up (+40.0%)",
        "news": [{"title": f"News {i}", "link": f"https://x/{i}"} for i in range(8)],
        "earnings": {
            "history": [
                {
                    "Earnings Date": "2026-12-09 15:00:00-05:00",
                    "EPS Estimate": 3.83,
                    "Reported EPS": None,
                    "Surprise(%)": None,
                },
                {
                    "Earnings Date": f"{earnings_day} 16:00:00-04:00",
                    "EPS Estimate": 2.4,
                    "Reported EPS": 2.44,
                    "Surprise(%)": 1.74,
                },
            ]
        },
    }


def _stored(written="2026-09-15", price=100.0):
    return StockView(ticker="AVGO", written_on=written, price=price, view="Durable franchise.")


@pytest.mark.parametrize(
    ("stored", "price", "reported", "reason"),
    [
        (None, 100.0, None, "no view yet"),
        (_stored("2026-09-10"), 100.0, None, "8 days old"),
        (_stored(), 109.0, None, "price moved +9.0%"),
        (_stored(), 100.0, date(2026, 9, 17), "reported earnings Sep 17"),
        (_stored(), 104.0, date(2026, 9, 2), None),  # reported before the view: reuse
    ],
)
def test_refresh_reason(stored, price, reported, reason):
    got = refresh_reason(
        stored, price=price, reported_on=reported, today=TODAY, max_age_days=7, move_pct=8.0
    )
    assert got == reason


def test_block_parses_like_the_llm_block():
    block = format_ticker_block(
        _data(), view="Durable franchise.", view_note=" (view from Sep 15)", today=TODAY
    )
    _, [section] = _parse(block)
    fields = dict(section.fields)
    assert section.symbol == "AVGO" and section.name == "Broadcom Inc."
    assert fields["Long-term view"] == "Durable franchise. (view from Sep 15)"
    assert fields["Analysts"] == "Buy 30 / Hold 5 / Sell 1, mean target $400.00"
    assert fields["Earnings"].startswith("Last: EPS 2.44 vs est 2.4 (+1.7%) on 2026-06-03")
    assert "Next: 2026-12-09" in fields["Earnings"]
    assert fields["Top News"].count("(https://x/") == 5
    # Its own row: a "52W Range" label was swallowed into Price by the parser.
    assert fields["Range 52W"] == "$138.10 - $374.23"
    assert fields["Price"] == "$100.00 (+1.20% today)"


class _FakeAgent:
    def __init__(self):
        self.calls = 0

    def run(self, prompt):
        self.calls += 1
        return SimpleNamespace(content="Durable franchise; AI networking demand runs for years.")


class _FakeReranker:
    """Stands in for the batched rerank call; `top_n` items, feed order."""

    def __init__(self, top_n=5):
        self.calls = 0
        self.top_n = top_n

    def rerank_batch(self, candidates, names=None, *, top_n=5):
        self.calls += 1
        return {t: items[: self.top_n] for t, items in candidates.items()}


def _bare_agent(tmp_path: Path, *, reranker=None) -> PortfolioAgent:
    agent = PortfolioAgent.__new__(PortfolioAgent)
    agent._positions_by_ticker = {}
    agent.db_path = str(tmp_path / "v.db")
    agent.view_max_age_days, agent.view_move_pct = 7, 8.0
    agent.ticker_data, agent.stored_views = {}, {}
    agent.ranked_news, agent.facts = {}, {}
    agent.views_written = agent.views_reused = 0
    agent.ticker_agent = _FakeAgent()
    agent.news_reranker = reranker or _FakeReranker()
    return agent


def _block(agent: PortfolioAgent, ticker: str = "AVGO") -> str:
    """Phases 1, 2 and 4 of run_analysis for one ticker (facts phase needs
    the network, so tests set `agent.facts` directly instead)."""
    agent._safe_fetch(ticker)
    agent._rank_all_news([ticker])
    return agent._run_ticker(ticker)


def test_view_written_once_then_reused(tmp_path: Path, monkeypatch):
    data = {"price": 100.0}
    monkeypatch.setattr(portfolio_agent, "fetch_ticker_data", lambda t: _data(price=data["price"]))
    agent = _bare_agent(tmp_path)

    first = _block(agent)
    second = _block(agent)
    assert agent.ticker_agent.calls == 1
    assert (agent.views_written, agent.views_reused) == (1, 1)
    assert "AI networking" in first and "(view from" in second
    data["price"] = 120.0  # +20%: rewritten
    _block(agent)
    assert agent.ticker_agent.calls == 2


def test_post_earnings_check():
    ticker_data = {
        "AVGO": _data(earnings_day="2026-09-16"),
        "OLD": _data(earnings_day="2026-06-03"),
    }
    recent = recent_results(ticker_data, today=TODAY)
    assert [(r["ticker"], r["reported_on"]) for r in recent] == [("AVGO", date(2026, 9, 16))]
    cut = with_revisions(
        recent, {"AVGO": {"next_year_up_30d": 2, "next_year_down_30d": 9, "net_revisions_7d": -3}}
    )
    assert cut[0]["direction"] == "lowering"
    assert "re-check the long-term thesis" in result_text(cut[0])
    held = with_revisions(
        recent, {"AVGO": {"next_year_up_30d": 6, "next_year_down_30d": 1, "net_revisions_7d": 2}}
    )
    assert result_text(held[0]).endswith("estimates raising since — the long-term case holds.")

    h = build_portfolio_health(
        {"IRA": [{"ticker": "AVGO", "units": 1, "average_purchase_price": 1.0, "price": 2.0}]},
        earnings_results=lambda: cut,
    )
    item = next(i for i in decision_items(h) if i["ticker"] == "AVGO")
    assert (item["label"], item["priority"]) == ("EARNINGS CUT", 2)


def test_stale_views_pruned(tmp_path: Path):
    from stock_analyzer.db.retention import RetentionPolicy, prune_database
    from stock_analyzer.db.session import get_session

    db = str(tmp_path / "p.db")
    with get_session(db) as s:
        s.add(StockView(ticker="SOLD", written_on="2026-01-01", view="Gone."))
        s.add(StockView(ticker="HELD", written_on=TODAY.isoformat(), view="Here."))
    out = prune_database(db, RetentionPolicy(), today=TODAY)
    assert out["stale_stock_views"] == 1
    with get_session(db) as s:
        assert s.get(StockView, "HELD") is not None and s.get(StockView, "SOLD") is None


def test_database_trouble_costs_a_model_call_not_the_block(tmp_path: Path, monkeypatch):
    """A busy SQLite file (holdings run in parallel threads) must never lose
    a stock's block, or the view the model was already paid for."""
    from stock_analyzer.agents import stock_views

    monkeypatch.setattr(portfolio_agent, "fetch_ticker_data", lambda t: _data())

    def boom(*a, **kw):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(stock_views, "get_session", boom)
    agent = _bare_agent(tmp_path)
    agent._safe_fetch("AVGO")
    agent._rank_all_news(["AVGO"])

    assert "AI networking" in agent._safe_ticker("AVGO")
    assert agent.views_written == 1


def test_headlines_are_not_repeated_the_next_morning(tmp_path: Path, monkeypatch):
    """A story doing the rounds for a week used to fill the section every
    morning; only what hasn't been sent is a candidate."""
    data = _data()
    data["news"] = [{"title": f"Broadcom news {i}", "link": f"https://x/{i}"} for i in range(6)]
    monkeypatch.setattr(portfolio_agent, "fetch_ticker_data", lambda t: deepcopy(data))
    agent = _bare_agent(tmp_path, reranker=_FakeReranker(top_n=3))

    first = [ln for ln in _block(agent).splitlines() if ln.startswith("- ")]
    second = [ln for ln in _block(agent).splitlines() if ln.startswith("- ")]
    assert len(first) == 3
    assert not set(first) & set(second)  # nothing shown twice
    third = [ln for ln in _block(agent).splitlines() if ln.startswith("- ")]
    assert not third  # all six have been sent
    assert "Nothing company-specific in today's feed." in _block(agent)


def test_facts_fill_a_block_with_no_news(tmp_path: Path, monkeypatch):
    """What NVDA looked like on 2026-09-19: no relevant headline, but an
    8-K and a 10-Q in the same window."""
    data = _data()
    data["news"] = []
    monkeypatch.setattr(portfolio_agent, "fetch_ticker_data", lambda t: deepcopy(data))
    agent = _bare_agent(tmp_path)
    agent._safe_fetch("AVGO")
    agent._rank_all_news(["AVGO"])
    agent.facts = {
        "AVGO": company_facts(
            filings=[{"form": "8-K", "filed_on": "2026-09-02", "url": "https://sec/8k"}],
            revisions={"next_year_up_30d": 6, "next_year_down_30d": 1},
        )
    }
    block = agent._run_ticker("AVGO")
    _, [section] = _parse(block)
    fields = dict(section.fields)
    assert fields["Top News"] == "Nothing company-specific in today's feed."
    assert fields["Filings"].startswith("8-K filed Sep 02 — material event")
    assert fields["Estimates"] == "Next-year EPS: 6 up / 1 down in 30 days — analysts raising"


def test_money_market_holdings_get_no_news_section(tmp_path: Path, monkeypatch):
    """SPAXX has no company news, no filings and no estimates — an empty
    news section for it is noise, not information."""
    data = _data()
    data["symbol"], data["name"] = "SPAXX", "Fidelity Government Money Market"
    data["quote_type"], data["news"], data["earnings"] = "MUTUALFUND", [], {}
    monkeypatch.setattr(portfolio_agent, "fetch_ticker_data", lambda t: deepcopy(data))
    agent = _bare_agent(tmp_path)

    agent._safe_fetch("SPAXX")
    agent._rank_all_news(["SPAXX"])
    agent._collect_facts(["SPAXX"])  # must not reach the network
    block = agent._run_ticker("SPAXX")
    assert "Top News" not in block
    assert agent.facts == {}


def test_equities_still_say_when_there_is_nothing(tmp_path: Path, monkeypatch):
    data = _data()
    data["news"], data["quote_type"] = [], "EQUITY"
    monkeypatch.setattr(portfolio_agent, "fetch_ticker_data", lambda t: deepcopy(data))
    agent = _bare_agent(tmp_path)
    agent._safe_fetch("AVGO")
    agent._rank_all_news(["AVGO"])
    assert "Nothing company-specific" in agent._run_ticker("AVGO")
