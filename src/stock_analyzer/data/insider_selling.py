"""Recent insider SELLING coverage via Tavily.

Mirrors data/insider.py (which is buy-biased). After fetching coverage, we
extract ticker mentions using the same regex+blacklist from discover.universe
so the output is ticker-keyed rather than article-keyed.

Used by the discover pipeline as a bearish flag — heavy recent insider
selling on a candidate is a major red signal the ranker should weigh.
"""
from __future__ import annotations

import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tavily import TavilyClient

from ..discover.universe import _extract_tickers
from ..logging import get_logger

logger = get_logger(__name__)

_TAVILY_MAX_WORKERS = 4

INSIDER_DOMAINS: list[str] = [
    "insidermonkey.com",
    "openinsider.com",
    "quiverquant.com",
    "unusualwhales.com",
]


def fetch_insider_selling_coverage(
    days: int = 14, max_results: int = 20
) -> list[dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; insider-selling fetch returns empty")
        return []
    client = TavilyClient(api_key=api_key)
    queries = [
        "large insider selling SEC Form 4 recent",
        "CEO CFO insider stock sale this week",
        "executive insider sells stock cluster",
        "heavy insider selling notable companies",
    ]

    def _search(q: str) -> dict[str, Any] | None:
        try:
            return client.search(
                query=q,
                search_depth="advanced",
                max_results=6,
                days=days,
                include_domains=INSIDER_DOMAINS,
            )
        except Exception as e:
            logger.warning("insider-selling query failed (%r): %s", q, e)
            return None

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=_TAVILY_MAX_WORKERS) as ex:
        for r in ex.map(_search, queries):
            if r:
                results.extend(r.get("results", []))

    # Dedupe by URL
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url")
        if url and url not in seen:
            seen.add(url)
            deduped.append(r)
    return deduped[:max_results]


def insider_selling_mentions(
    survivor_tickers: set[str], days: int = 14
) -> dict[str, int]:
    """Return {ticker: mention_count} for survivors that appear in recent
    insider-selling coverage. Missing tickers = no signal (no rows)."""
    items = fetch_insider_selling_coverage(days=days)
    counter: Counter[str] = Counter()
    for item in items:
        text = " ".join((item.get("title") or "", item.get("content") or ""))
        for t in _extract_tickers(text):
            if t in survivor_tickers:
                counter[t] += 1
    return dict(counter)
