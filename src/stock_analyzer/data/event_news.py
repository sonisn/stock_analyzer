"""Company events that arrive as press releases or government notices, not
SEC filings: a watched stock's new customer, a regulator's approval, a
contract, a milestone.

Why a news feed: Oklo filed three 8-Ks between June and October 2026 (a
vote, an officer change, the ATM sales agreement) while its customers,
DOE approvals and reactor milestones went out as Business Wire releases
only. Finnhub's company feed for it was 155 syndicated articles in sixty
days, none of them a release. Google News' search feed carries the wire
releases, the agencies' own pages (energy.gov) and the trade press
(Utility Dive's FERC ruling), free and without a key.

Most of the feed is still commentary ("Is Oklo the next trillion-dollar
stock?"), so `looks_like_event` keeps a headline only when it names an
event and does not read as an opinion piece. A model reads what is left
(reporting/event_watch.py).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote_plus

from ..http_client import HttpClient, HttpClientError
from ..logging import get_logger

logger = get_logger(__name__)

GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
LOOKBACK_DAYS = 3

_HTTP = HttpClient(
    default_headers={"User-Agent": "Mozilla/5.0 (stock-analyzer event watch)"},
    timeout=20.0,
    rate_limit_per_min=20,
    name="google-news",
)

# A headline that names something happening to the company.
EVENT_WORDS = re.compile(
    r"\b(agreement|agrees|signs?|signed|selected|selects|awarded?|awards|contract|"
    r"customer|order|deal|partner(ship|s)?|collaborat\w*|memorandum|mou|ppa|"
    r"power purchase|offtake|supply|acquires?|acquisition|merger|joint venture|"
    r"approv\w*|license[sd]?|licensing|permit\w*|authoriz\w*|clearance|certif\w*|"
    r"nrc|doe|ferc|department of energy|nuclear regulatory|criticality|"
    r"construction|groundbreaking|breaks ground|milestone|completes?|completed|"
    r"launch(es|ed)?|commission\w*|grant|funding|loan|offering|raises?|"
    r"rejects?|denie[sd]|den(y|ial)|withdraw\w*|terminat\w*|cancel\w*|delay\w*|"
    r"lawsuit|sues|investigation|subpoena|recall|appoints?|names|ceo|cfo|resign\w*|"
    r"results|guidance|quarter|revenue|sales|posts|reports)\b",
    re.IGNORECASE,
)
# A headline that reads as commentary, a ranking or a price move.
OPINION = re.compile(
    r"(\?|\bvs\.?\s|\bversus\b|\bprediction\b|\bshould you\b|\bbuy now\b|\bbetter buy\b|"
    r"\bstocks? to\b|\bstock (price|quote|forecast)\b|\b\d+ (stocks|reasons|things)\b|"
    r"\bhere'?s (why|how|where|what)\b|\bwhy (shares|the stock|investors)\b|"
    r"\b(soars?|jumps?|plunges?|sinks?|falls?|drops?|rall(y|ies)|climbs?|tumbles?|"
    r"slides?|surges?)\b.*\d+%|\b52-week\b|\bprice target\b|\b(upgrade|downgrade|"
    r"reiterates?|initiates? coverage|overweight|underweight)\w*\b|\bmotley fool\b|"
    r"\bseeking alpha\b|\bmillionaire\b|\btrillion-dollar\b|\bcould be\b|\bis it\b)",
    re.IGNORECASE,
)


def looks_like_event(title: str) -> bool:
    return bool(EVENT_WORDS.search(title)) and not OPINION.search(title)


def _strip_source(title: str, source: str) -> str:
    """Google News appends " - Source" to every title."""
    suffix = f" - {source}"
    return title[: -len(suffix)] if source and title.endswith(suffix) else title


def parse_feed(xml_text: str) -> list[dict[str, Any]]:
    """Items of a Google News RSS feed: title, source, url, published (date)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("Google News feed unreadable (%s)", e)
        return []
    out = []
    for item in root.iter("item"):
        source_el = item.find("source")
        source = (source_el.text or "").strip() if source_el is not None else ""
        title = _strip_source((item.findtext("title") or "").strip(), source)
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").date()
        except TypeError, ValueError:
            published = None
        out.append(
            {
                "title": title,
                "source": source,
                "url": (item.findtext("link") or "").strip(),
                "published": published,
            }
        )
    return out


def company_query(name: str) -> str:
    """The search phrase for a company: its name without the legal suffix,
    quoted ("Oklo Inc." -> "Oklo")."""
    words = [w for w in re.split(r"[\s,]+", name) if w]
    while words and words[-1].lower().rstrip(".") in {
        "inc",
        "corp",
        "corporation",
        "co",
        "ltd",
        "plc",
        "holdings",
        "group",
        "company",
        "sa",
        "nv",
        "ag",
    }:
        words.pop()
    return f'"{" ".join(words)}"'


def fetch_events(name: str, *, today: date, days: int = LOOKBACK_DAYS) -> list[dict[str, Any]]:
    """Headlines naming `name` from the last `days` days that look like an
    event, newest first, one per title. [] when the feed fails."""
    q = quote_plus(f"{company_query(name)} when:{days}d")
    try:
        text = _HTTP.get(GOOGLE_NEWS_URL.format(q=q)).text
    except HttpClientError as e:
        logger.warning("Google News for %s failed (%s)", name, e)
        return []
    since = today - timedelta(days=days)
    key = company_query(name).strip('"').lower()
    seen: set[str] = set()
    out = []
    for item in sorted(
        parse_feed(text), key=lambda i: i["published"] or datetime.min.date(), reverse=True
    ):
        title = item["title"]
        norm = re.sub(r"\W+", " ", title.lower()).strip()
        if norm in seen or key not in title.lower():
            continue
        if item["published"] is not None and item["published"] < since:
            continue
        if not looks_like_event(title):
            continue
        seen.add(norm)
        out.append(item)
    return out


__all__ = ["company_query", "fetch_events", "looks_like_event", "parse_feed"]
