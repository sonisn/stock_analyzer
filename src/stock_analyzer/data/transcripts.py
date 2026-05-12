"""Earnings call transcript snippets via Tavily.

Earnings transcripts aren't reliably available via any free API, but Motley
Fool and Seeking Alpha publish them for major tickers. We use Tavily to
search those domains for the most recent call and extract the relevant
narrative (guidance + Q&A excerpts, capped to ~4k chars per ticker).

Quality varies wildly by ticker:
  - Large caps (NVDA, AAPL, MSFT, etc.): excellent coverage
  - Mid caps: usually findable
  - Small caps / newer listings: often nothing

Graceful degradation: any failure returns None and the pipeline keeps going.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tavily import TavilyClient

from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 3
_MAX_TRANSCRIPT_CHARS = 4000

TRANSCRIPT_DOMAINS: list[str] = [
    "fool.com",
    "seekingalpha.com",
    "investing.com",
    "businesswire.com",
    "globenewswire.com",
]


def fetch_transcript_snippet(ticker: str) -> dict[str, Any] | None:
    """Search Tavily for the latest earnings call transcript, return a
    capped snippet of the narrative. None if nothing useful found."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None

    client = TavilyClient(api_key=api_key)
    query = f"{ticker} most recent earnings call transcript guidance"
    try:
        resp = client.search(
            query=query,
            search_depth="advanced",
            max_results=3,
            days=120,  # last 4 months — covers most recent quarter
            include_domains=TRANSCRIPT_DOMAINS,
            include_raw_content=False,
        )
    except Exception as e:
        logger.debug("transcript search failed for %s: %s", ticker, e)
        return None

    results = (resp or {}).get("results", [])
    if not results:
        return None

    # Take the top result; prefer ones with longer content snippets.
    results.sort(key=lambda r: len(r.get("content") or ""), reverse=True)
    top = results[0]
    content = (top.get("content") or "").strip()
    if len(content) < 300:
        # Not substantive — likely a search-page summary, not real transcript text.
        return None

    return {
        "ticker": ticker,
        "url": top.get("url"),
        "title": top.get("title"),
        "published_date": top.get("published_date"),
        "snippet": content[:_MAX_TRANSCRIPT_CHARS],
    }


def batch_transcript_snippets(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch transcript snippets for a list of tickers. Sequential per ticker
    (Tavily rate limits favor lower concurrency) but capped at _MAX_WORKERS."""
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, r in zip(tickers, ex.map(fetch_transcript_snippet, tickers)):
            if r:
                results[ticker] = r
    return results
