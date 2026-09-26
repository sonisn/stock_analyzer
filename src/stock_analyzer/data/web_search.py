"""Web search for news, catalysts and filings coverage: Exa first, Tavily second.

Exa's free credit ($10 a month, about $0.007 a search with article text —
~1,400 searches) is used first; Tavily's free 1,000 a month takes over when
Exa's credit runs out, its key is rejected, or a call fails. Both answer in
Tavily's shape — {"results": [{title, url, content, published_date, score}]}
— so callers written against TavilyClient.search keep working unchanged,
and each result says which provider found it.

Exa is called at most EXA_QPS times a second across every thread (one
shared, throttled HTTP client), under Exa's own per-key limit.

A Tavily quota error is raised as-is: callers already recognize it and stop
asking for the rest of the batch.
"""

from __future__ import annotations

import atexit
import os
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from ..http_client import HttpClient, HttpClientError
from ..logging import get_logger

logger = get_logger(__name__)

EXA_URL = "https://api.exa.ai/search"
EXA_QPS = 10
# Article text per result. Tavily's `content` runs a few hundred to ~1,500
# characters; the transcript lookup keeps the longest, so give it room.
EXA_TEXT_CHARS = 2000

_EXA = HttpClient(timeout=30.0, rate_limit_per_min=EXA_QPS * 60, name="exa")
# Set when Exa's credit is spent or its key rejected: stop asking for the
# rest of the process instead of failing once per query.
_exa_off = threading.Event()
_lock = threading.Lock()
_usage = {"exa": 0, "tavily": 0, "exa_cost_usd": 0.0}


def available() -> bool:
    return bool(os.getenv("EXA_API_KEY") or os.getenv("TAVILY_API_KEY"))


def client() -> WebSearch | None:
    """A drop-in for TavilyClient, or None when neither key is set."""
    return WebSearch() if available() else None


def usage() -> dict[str, float]:
    with _lock:
        return dict(_usage)


def _count(provider: str, cost: float = 0.0) -> None:
    with _lock:
        _usage[provider] += 1
        _usage["exa_cost_usd"] += cost


@atexit.register
def _log_usage() -> None:
    """One line per run that searched, so the logs say what each command
    spends against the free allowances."""
    u = usage()
    if u["exa"] or u["tavily"]:
        logger.info(
            "Web search usage: Exa %d ($%.3f of the $10/month credit), Tavily %d",
            u["exa"],
            u["exa_cost_usd"],
            u["tavily"],
        )


class WebSearch:
    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        days: int | None = None,
        include_domains: list[str] | None = None,
        topic: str | None = None,
        **tavily_only: Any,
    ) -> dict[str, Any]:
        """Tavily's `search` signature; `search_depth` and the like only
        reach Tavily."""
        exa_key = os.getenv("EXA_API_KEY")
        if exa_key and not _exa_off.is_set():
            try:
                return {
                    "results": _exa_search(
                        exa_key,
                        query,
                        max_results=max_results,
                        days=days,
                        include_domains=include_domains,
                        news=topic == "news",
                    )
                }
            except HttpClientError as e:
                if e.status in (401, 402, 403):
                    _exa_off.set()
                    logger.warning(
                        "Exa unavailable for the rest of this run (HTTP %s: credit spent or "
                        "key rejected) — using Tavily",
                        e.status,
                    )
                else:
                    logger.warning("Exa search failed (%s) — trying Tavily", e)
            except Exception as e:  # noqa: BLE001 — any Exa failure falls back
                logger.warning("Exa search failed (%s) — trying Tavily", e)
        return _tavily_search(
            query,
            max_results=max_results,
            days=days,
            include_domains=include_domains,
            topic=topic,
            **tavily_only,
        )


def _exa_search(
    key: str,
    query: str,
    *,
    max_results: int,
    days: int | None,
    include_domains: list[str] | None,
    news: bool,
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {
        "query": query,
        "numResults": max_results,
        "type": "auto",
        "contents": {"text": {"maxCharacters": EXA_TEXT_CHARS}},
    }
    if days:
        since = datetime.now(UTC) - timedelta(days=days)
        body["startPublishedDate"] = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    if include_domains:
        body["includeDomains"] = include_domains
    if news:
        body["category"] = "news"
    data = _EXA.post_json(EXA_URL, json=body, headers={"x-api-key": key})
    _count("exa", float((data.get("costDollars") or {}).get("total") or 0.0))
    return [
        {
            "title": r.get("title") or "",
            "url": r.get("url") or "",
            "content": (r.get("text") or r.get("summary") or "").strip(),
            "published_date": (r.get("publishedDate") or "")[:10] or None,
            "score": float(r.get("score") or 0.0),
            "provider": "exa",
        }
        for r in data.get("results") or []
        if r.get("url")
    ]


def _tavily_search(query: str, **kwargs: Any) -> dict[str, Any]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return {"results": []}
    from tavily import TavilyClient

    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    resp = TavilyClient(api_key=key).search(query=query, **kwargs)
    _count("tavily")
    for r in resp.get("results") or []:
        r.setdefault("provider", "tavily")
    return resp
