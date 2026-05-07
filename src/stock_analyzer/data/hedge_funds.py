"""Recent top hedge fund trade coverage via Tavily."""
from __future__ import annotations

import os
from typing import Any

from tavily import TavilyClient

from ..logging import get_logger

logger = get_logger(__name__)

# Billionaire investors with proven track records of significant profit —
# personal net worth largely built from investing, with documented wins.
# Excludes faceless quant shops, retired/family-office-only managers
# (Soros), funds with mixed recent performance (Tiger Global), and
# managers who have stepped back (Simons — deceased; Dalio — retired).
# Manager name is matched alongside fund name since coverage often
# references the person rather than the firm.
HIGH_PROFILE_FUNDS: tuple[tuple[str, str], ...] = (
    ("Berkshire Hathaway", "Warren Buffett"),
    ("Icahn Enterprises", "Carl Icahn"),
    ("Pershing Square", "Bill Ackman"),
    ("Appaloosa", "David Tepper"),
    ("Scion Asset Management", "Michael Burry"),
    ("Duquesne Family Office", "Stan Druckenmiller"),
    ("Baupost", "Seth Klarman"),
    ("Third Point", "Daniel Loeb"),
    ("Elliott Management", "Paul Singer"),
    ("Greenlight Capital", "David Einhorn"),
    ("Citadel", "Ken Griffin"),
    ("Point72", "Steve Cohen"),
    ("Millennium", "Israel Englander"),
)

HEDGE_FUND_DOMAINS: list[str] = [
    "insidermonkey.com",
    "whalewisdom.com",
    "hedgefollow.com",
    "dataroma.com",
    "13f.info",
    "quiverquant.com",
    "unusualwhales.com",
    "stockcircle.com",
]


def _mentioned_funds(text: str) -> list[str]:
    text_lower = text.lower()
    matches: list[str] = []
    for fund, manager in HIGH_PROFILE_FUNDS:
        if fund.lower() in text_lower or (manager and manager.lower() in text_lower):
            matches.append(fund)
    return matches


def fetch_hedge_fund_trades(days: int = 5, max_results: int = 20) -> list[dict[str, Any]]:
    """Fetch recent top hedge fund position changes and 13F coverage.

    13F filings lag actual trades by up to 45 days; press coverage and
    activist letters are more timely than the filings themselves.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set; hedge fund fetch returns empty")
        return []

    client = TavilyClient(api_key=api_key)
    queries = [
        "Warren Buffett Berkshire Hathaway recent stock buy sell",
        "Bill Ackman Pershing Square new position",
        "Michael Burry Scion 13F filing recent",
        "David Tepper Appaloosa stock pick recent",
        "Stan Druckenmiller Duquesne new position",
        "Carl Icahn activist position recent",
        "Daniel Loeb Third Point new investment",
        "billionaire hedge fund manager top stock pick recent",
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
                include_domains=HEDGE_FUND_DOMAINS,
            )
        except Exception as e:
            logger.warning("Hedge fund query failed (%r): %s", q, e)
            continue

        for r in res.get("results", []):
            url = r.get("url")
            title = r.get("title")
            if not (url and title) or url in seen_urls:
                continue
            content = r.get("content") or ""
            mentioned = _mentioned_funds(title + " " + content)
            seen_urls.add(url)
            out.append(
                {
                    "title": title,
                    "link": url,
                    "snippet": content[:400],
                    "funds": mentioned,
                    "score": round(float(r.get("score") or 0), 3),
                    "is_watchlist": bool(mentioned),
                }
            )

    out.sort(key=lambda x: (not x["is_watchlist"], -x["score"]))
    logger.info(
        "Hedge fund fetch: %d items (%d watchlist matches)",
        len(out),
        sum(x["is_watchlist"] for x in out),
    )
    return out[:max_results]
