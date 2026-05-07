"""Macro / market-wide news fetch (Tavily) for sentiment synthesis."""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tavily import TavilyClient

from ..logging import get_logger

_TAVILY_MAX_WORKERS = 3

logger = get_logger(__name__)

PREMIUM_NEWS_DOMAINS: list[str] = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "barrons.com",
    "marketwatch.com",
    "finance.yahoo.com",
    "seekingalpha.com",
    "investors.com",
    "fool.com",
]


def fetch_market_sentiment_news(*, max_results: int = 10) -> list[dict]:
    """Fetch recent US macro/market news for sentiment synthesis."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; returning empty sentiment news")
        return []
    queries = [
        "US stock market today S&P 500 Nasdaq Dow",
        "US economy CPI jobs Fed interest rates this week",
        "geopolitical news affecting US markets today",
    ]
    client = TavilyClient(api_key=api_key)

    def _search(q: str) -> dict[str, Any] | None:
        try:
            return client.search(
                query=q,
                topic="news",
                search_depth="basic",
                max_results=5,
                days=1,
                include_domains=PREMIUM_NEWS_DOMAINS,
            )
        except Exception as e:
            logger.warning("Tavily sentiment query failed (%r): %s", q, e)
            return None

    with ThreadPoolExecutor(max_workers=_TAVILY_MAX_WORKERS) as ex:
        responses = list(ex.map(_search, queries))

    seen: set[str] = set()
    out: list[dict] = []
    for res in responses:
        if res is None:
            continue
        for r in res.get("results", []):
            url = r.get("url")
            title = r.get("title")
            if url and title and url not in seen:
                seen.add(url)
                out.append({"title": title, "snippet": (r.get("content") or "")[:250]})
            if len(out) >= max_results:
                logger.info("Sentiment news: %d items", len(out))
                return out
    logger.info("Sentiment news: %d items", len(out))
    return out
