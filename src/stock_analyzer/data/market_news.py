"""Macro / market-wide news fetch (Tavily) for sentiment synthesis."""

from __future__ import annotations

import os
import time
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
    """Recent US macro/market news for sentiment synthesis: Tavily first,
    Finnhub's general market feed when Tavily returns nothing (quota
    exhausted, outage, or no key) — so the daily sentiment block doesn't
    go blank when one news source is down."""
    items = _tavily_market_news(max_results=max_results)
    if items:
        return items
    items = _finnhub_market_news(max_results=max_results)
    logger.info("Sentiment news: %d items from the Finnhub fallback", len(items))
    return items


def _finnhub_market_news(*, max_results: int, hours: int = 36) -> list[dict]:
    from . import finnhub as finnhub_data

    client = finnhub_data._client()
    if client is None:
        logger.warning("No Finnhub key either; sentiment news is empty")
        return []
    cutoff = time.time() - hours * 3600
    seen: set[str] = set()
    out: list[dict] = []
    for r in sorted(
        finnhub_data.fetch_general_news(client),
        key=lambda r: r.get("datetime") or 0,
        reverse=True,
    ):
        title = (r.get("headline") or "").strip()
        if not title or (r.get("datetime") or 0) < cutoff or title in seen:
            continue
        seen.add(title)
        out.append({"title": title, "snippet": (r.get("summary") or "")[:250]})
        if len(out) >= max_results:
            break
    return out


def _tavily_market_news(*, max_results: int) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; skipping Tavily market news")
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
