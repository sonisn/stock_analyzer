"""Daily sentiment news falls back to Finnhub when Tavily returns nothing."""

from __future__ import annotations

import time

from stock_analyzer.data import finnhub as finnhub_data
from stock_analyzer.data import market_news


def test_finnhub_fallback_when_tavily_is_empty(monkeypatch):
    now = time.time()
    feed = [
        {"headline": "Old story", "summary": "x", "datetime": now - 72 * 3600},
        {"headline": "Fed holds rates", "summary": "Powell said...", "datetime": now - 3600},
        {"headline": "Fed holds rates", "summary": "dup", "datetime": now - 1800},
        {"headline": "Oil slips", "summary": "Crude fell...", "datetime": now - 600},
    ]
    monkeypatch.setattr(market_news, "_tavily_market_news", lambda **kw: [])
    monkeypatch.setattr(finnhub_data, "_client", lambda: object())
    monkeypatch.setattr(finnhub_data, "fetch_general_news", lambda client: feed)

    items = market_news.fetch_market_sentiment_news(max_results=5)
    assert [i["title"] for i in items] == [
        "Oil slips",
        "Fed holds rates",
    ]  # newest first, deduped, <36h


def test_tavily_results_win_when_present(monkeypatch):
    monkeypatch.setattr(
        market_news, "_tavily_market_news", lambda **kw: [{"title": "T", "snippet": ""}]
    )
    monkeypatch.setattr(
        finnhub_data, "_client", lambda: (_ for _ in ()).throw(AssertionError("unused"))
    )
    assert market_news.fetch_market_sentiment_news() == [{"title": "T", "snippet": ""}]
