"""Recent insider trading coverage via Tavily."""
from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from ..logging import get_logger

logger = get_logger(__name__)

INSIDER_DOMAINS: list[str] = [
    "insidermonkey.com",
    "openinsider.com",
    "quiverquant.com",
    "unusualwhales.com",
]


def fetch_insider_trades(days: int = 5, max_results: int = 20) -> list[dict[str, Any]]:
    """Fetch recent insider trade coverage, biased to InsiderMonkey + OpenInsider."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; insider fetch returns empty")
        return []

    client = TavilyClient(api_key=api_key)
    queries = [
        "insider buying SEC Form 4 recent transactions",
        "CEO CFO insider stock purchase this week",
        "notable insider buying clusters",
        "large insider sales 10b5-1",
        "billionaire hedge fund stock picks recent",
    ]

    seen_urls: set[str] = set()
    out: list[dict[str, Any]] = []

    for q in queries:
        try:
            res = client.search(
                query=q,
                search_depth="advanced",
                max_results=8,
                days=days,
                include_domains=INSIDER_DOMAINS,
            )
        except Exception as e:
            logger.warning("Insider query failed (%r): %s", q, e)
            continue

        for r in res.get("results", []):
            url = r.get("url")
            title = r.get("title")
            if not (url and title) or url in seen_urls:
                continue
            seen_urls.add(url)
            out.append(
                {
                    "title": title,
                    "link": url,
                    "snippet": (r.get("content") or "")[:400],
                    "score": round(float(r.get("score") or 0), 3),
                }
            )

    out.sort(key=lambda x: x["score"], reverse=True)
    logger.info("Insider fetch: %d items", len(out))
    return out[:max_results]
