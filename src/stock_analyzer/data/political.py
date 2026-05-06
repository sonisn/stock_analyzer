"""Recent congressional trade coverage via Tavily."""
from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from ..logging import get_logger

logger = get_logger(__name__)

# Proxy filter for "high net worth + beating SPY". True net-worth and historical
# performance vs SPY require separate APIs (Quiver Quantitative, OpenSecrets).
# This watchlist captures politicians whose trades are most widely tracked for alpha.
HIGH_PROFILE_POLITICIANS: tuple[str, ...] = (
    "Nancy Pelosi",
    "Paul Pelosi",
    "Tommy Tuberville",
    "Dan Crenshaw",
    "Ro Khanna",
    "Josh Gottheimer",
    "Susie Lee",
    "Marjorie Taylor Greene",
    "Brian Mast",
    "Earl Blumenauer",
    "Michael McCaul",
    "Mark Green",
    "Shelley Moore Capito",
    "Kathy Castor",
    "Blake Moore",
    "Debbie Wasserman Schultz",
    "Garret Graves",
)

POLITICAL_DOMAINS: list[str] = [
    "capitoltrades.com",
    "quiverquant.com",
    "insidermonkey.com",
]


def _mentioned_politicians(text: str) -> list[str]:
    text_lower = text.lower()
    return [p for p in HIGH_PROFILE_POLITICIANS if p.lower() in text_lower]


def fetch_political_trades(days: int = 5, max_results: int = 20) -> list[dict[str, Any]]:
    """Fetch recent congressional trade disclosures.

    Note: STOCK Act disclosures lag the actual trade by up to 45 days, so even
    "last 5 days of articles" will reference older trade dates.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; political fetch returns empty")
        return []

    client = TavilyClient(api_key=api_key)
    queries = [
        "Nancy Pelosi recent stock trade disclosure",
        "Tommy Tuberville recent stock trade disclosure",
        "Dan Crenshaw recent stock trade disclosure",
        "Ro Khanna recent stock trade disclosure",
        "Josh Gottheimer recent stock trade disclosure",
        "Marjorie Taylor Greene recent stock trade disclosure",
        "congressional stock trades disclosure this week",
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
                include_domains=POLITICAL_DOMAINS,
            )
        except Exception as e:
            logger.warning("Political query failed (%r): %s", q, e)
            continue

        for r in res.get("results", []):
            url = r.get("url")
            title = r.get("title")
            if not (url and title) or url in seen_urls:
                continue
            content = r.get("content") or ""
            mentioned = _mentioned_politicians(title + " " + content)
            seen_urls.add(url)
            out.append(
                {
                    "title": title,
                    "link": url,
                    "snippet": content[:400],
                    "politicians": mentioned,
                    "score": round(float(r.get("score") or 0), 3),
                    "is_watchlist": bool(mentioned),
                }
            )

    out.sort(key=lambda x: (not x["is_watchlist"], -x["score"]))
    logger.info("Political fetch: %d items (%d watchlist matches)", len(out), sum(x["is_watchlist"] for x in out))
    return out[:max_results]
