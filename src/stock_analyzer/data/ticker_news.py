"""Per-ticker recent news with dates and article snippets (Tavily).

yfinance's `.news` gives the pipeline three bare headlines with no date or
body, which is too thin for the Analyst/Reviewer to name a forward
catalyst from. This fetch returns the last N days of dated coverage with a
content snippet and a short stable id ("N1", "N2", ...) that the LLM must
cite, so `discover/catalysts.py` can drop any catalyst whose source doesn't
exist.

Cost: one Tavily basic search per ticker (a second, unfiltered one only
when the premium-domain search finds nothing).
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from urllib.parse import urlparse

from tavily import TavilyClient

from ..logging import get_logger
from .market_news import PREMIUM_NEWS_DOMAINS

logger = get_logger(__name__)

_TAVILY_MAX_WORKERS = 3
_SNIPPET_CHARS = 400


class TavilyQuotaExceeded(RuntimeError):
    """The Tavily plan's usage limit is exhausted — every further call fails."""


def _is_quota_error(e: Exception) -> bool:
    text = str(e).lower()
    return "usage limit" in text or "exceeds your plan" in text


def _query(ticker: str, company_name: str | None) -> str:
    if company_name:
        return f"{company_name} ({ticker}) stock news guidance outlook"
    return f"{ticker} stock news guidance outlook"


def _normalize(results: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for r in results:
        url = r.get("url")
        title = r.get("title")
        if not url or not title or url in seen:
            continue
        seen.add(url)
        items.append(
            {
                "title": title,
                "published_date": r.get("published_date"),
                "source": urlparse(url).netloc.removeprefix("www."),
                "snippet": (r.get("content") or "")[:_SNIPPET_CHARS],
            }
        )
    # Newest first; undated items sort last.
    items.sort(key=lambda it: it["published_date"] or "", reverse=True)
    items = items[:max_results]
    for i, it in enumerate(items, start=1):
        it["id"] = f"N{i}"
    return items


def fetch_ticker_news(
    ticker: str,
    company_name: str | None = None,
    *,
    days: int = 30,
    max_results: int = 6,
    client: TavilyClient | None = None,
) -> list[dict[str, Any]]:
    """Recent dated news for one ticker, newest first, each with an `id`.

    Raises TavilyQuotaExceeded when the plan limit is hit, so a batch can
    stop instead of failing once per ticker; any other error returns []."""
    if client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return []
        client = TavilyClient(api_key=api_key)

    def _search(domains: list[str] | None) -> list[dict[str, Any]] | None:
        kwargs: dict[str, Any] = {
            "query": _query(ticker, company_name),
            "topic": "news",
            "search_depth": "basic",
            "max_results": max_results,
            "days": days,
        }
        if domains:
            kwargs["include_domains"] = domains
        try:
            return client.search(**kwargs).get("results", []) or []
        except Exception as e:
            if _is_quota_error(e):
                raise TavilyQuotaExceeded(str(e)) from e
            logger.warning("Tavily ticker news failed for %s: %s", ticker, e)
            return None

    results = _search(PREMIUM_NEWS_DOMAINS)
    if results == []:
        # Smaller names often have no premium-outlet coverage in the window.
        # Not retried after an error — that would just spend a second credit.
        results = _search(None)
    return _normalize(results or [], max_results)


def batch_ticker_news(
    tickers: list[str],
    names: dict[str, str | None] | None = None,
    *,
    days: int = 30,
) -> dict[str, list[dict[str, Any]]]:
    """{ticker: [news item, ...]} for every ticker. Empty dict without a key."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or not tickers:
        if not api_key:
            logger.warning("TAVILY_API_KEY not set; catalyst news fetch returns empty")
        return {}
    client = TavilyClient(api_key=api_key)
    names = names or {}
    quota_hit = threading.Event()

    def _one(ticker: str) -> tuple[str, list[dict[str, Any]]]:
        if quota_hit.is_set():
            return ticker, []
        try:
            return ticker, fetch_ticker_news(ticker, names.get(ticker), days=days, client=client)
        except TavilyQuotaExceeded as e:
            if not quota_hit.is_set():
                quota_hit.set()
                logger.error(
                    "Tavily plan usage limit reached — skipping catalyst news for the "
                    "rest of this run; catalysts will come from filings only (%s)",
                    e,
                )
            return ticker, []

    with ThreadPoolExecutor(max_workers=_TAVILY_MAX_WORKERS) as ex:
        out = dict(ex.map(_one, tickers))
    covered = sum(1 for v in out.values() if v)
    logger.info("Catalyst news: %d/%d tickers with recent coverage", covered, len(tickers))
    return out
