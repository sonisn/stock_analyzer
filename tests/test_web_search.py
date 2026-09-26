"""Exa first, Tavily second, in Tavily's shape."""

from __future__ import annotations

import pytest

from stock_analyzer.data import web_search as ws
from stock_analyzer.http_client import ClientError


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    ws._exa_off.clear()
    monkeypatch.setenv("EXA_API_KEY", "exa")
    monkeypatch.setenv("TAVILY_API_KEY", "tav")
    yield
    ws._exa_off.clear()


def test_exa_results_come_back_in_tavily_shape(monkeypatch):
    sent = {}

    def post_json(url, json, headers):
        sent.update(json)
        return {
            "costDollars": {"total": 0.007},
            "results": [
                {
                    "title": "Arista beats",
                    "url": "https://x/a",
                    "text": "Revenue up 38%",
                    "publishedDate": "2026-09-24T00:00:00.000Z",
                },
                {"title": "no url"},
            ],
        }

    monkeypatch.setattr(ws._EXA, "post_json", post_json)
    out = ws.WebSearch().search(
        query="ANET news",
        max_results=3,
        days=30,
        topic="news",
        include_domains=["reuters.com"],
        search_depth="basic",
    )["results"]
    assert out == [
        {
            "title": "Arista beats",
            "url": "https://x/a",
            "content": "Revenue up 38%",
            "published_date": "2026-09-24",
            "score": 0.0,
            "provider": "exa",
        }
    ]
    assert sent["numResults"] == 3 and sent["category"] == "news"
    assert sent["includeDomains"] == ["reuters.com"] and "startPublishedDate" in sent
    assert "search_depth" not in sent  # Tavily-only options never reach Exa


def test_spent_exa_credit_switches_to_tavily_for_the_rest_of_the_run(monkeypatch):
    calls = {"exa": 0, "tavily": 0}

    def broke(*a, **k):
        calls["exa"] += 1
        raise ClientError("exa: 402", status=402)

    def tavily(query, **kwargs):
        calls["tavily"] += 1
        return {"results": [{"title": "t", "url": "u", "content": "c"}]}

    monkeypatch.setattr(ws._EXA, "post_json", broke)
    monkeypatch.setattr(ws, "_tavily_search", tavily)
    first = ws.WebSearch().search(query="a")["results"]
    ws.WebSearch().search(query="b")
    assert first[0]["title"] == "t" and calls == {"exa": 1, "tavily": 2}


def test_exa_is_paced_at_ten_a_second_and_optional(monkeypatch):
    assert ws._EXA._min_interval == pytest.approx(0.1)
    monkeypatch.delenv("EXA_API_KEY")
    monkeypatch.setattr(ws, "_tavily_search", lambda query, **k: {"results": []})
    assert ws.WebSearch().search(query="x") == {"results": []}
    monkeypatch.delenv("TAVILY_API_KEY")
    assert ws.client() is None
