"""SEC EDGAR client — fetch latest 10-K risk factors.

Zero-dep: plain `requests` + regex. SEC requires a User-Agent with contact
info (per their fair-access policy) or returns 403. Parsing 10-K HTML is
inherently fragile (filings vary widely) — any failure returns None rather
than crashing the pipeline; the LLM works with what it has.

Endpoints used (all free, no API key):
  - company_tickers.json   ticker → CIK map (cached for the process)
  - submissions/CIK*.json  list of filings per CIK
  - Archives/edgar/data/   the filing document HTML
"""
from __future__ import annotations

import html
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests

from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 3   # SEC limits 10 req/sec; stay polite
_USER_AGENT = "stock-analyzer research-bot (snehal.soni@farohealth.com)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

_TICKER_TO_CIK: dict[str, int] | None = None


def _load_ticker_map() -> dict[str, int]:
    global _TICKER_TO_CIK
    if _TICKER_TO_CIK is not None:
        return _TICKER_TO_CIK
    try:
        resp = requests.get(_TICKERS_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _TICKER_TO_CIK = {
            row["ticker"].upper(): int(row["cik_str"]) for row in data.values()
        }
    except Exception as e:
        logger.warning("SEC ticker map fetch failed: %s", e)
        _TICKER_TO_CIK = {}
    return _TICKER_TO_CIK


def _latest_10k_url(cik: int) -> tuple[str, str] | None:
    """Return (filing_date, primary_doc_url) for the latest 10-K, or None."""
    try:
        resp = requests.get(
            _SUBMISSIONS_URL.format(cik=cik), headers=_HEADERS, timeout=15
        )
        resp.raise_for_status()
        sub = resp.json()
    except Exception as e:
        logger.warning("SEC submissions fetch failed for CIK %s: %s", cik, e)
        return None
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    for form, acc, doc, date in zip(forms, accessions, docs, dates):
        if form == "10-K":
            acc_clean = acc.replace("-", "")
            return (
                date,
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc}",
            )
    return None


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ITEM_1A_RE = re.compile(r"item\s*1a\.?\s+risk\s+factors?", re.IGNORECASE)
_ITEM_1B_RE = re.compile(r"item\s*1b\.?\s", re.IGNORECASE)
_ITEM_2_RE = re.compile(r"item\s*2\.?\s+properties", re.IGNORECASE)


def _strip_html(raw_html: str) -> str:
    text = _TAG_RE.sub(" ", raw_html)
    text = html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def _extract_item_1a(text: str, max_chars: int = 6000) -> str | None:
    # Pick the LAST occurrence — the first is usually the table-of-contents
    # reference; the second is the real section header.
    matches = list(_ITEM_1A_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    end_match = _ITEM_1B_RE.search(text, start) or _ITEM_2_RE.search(text, start)
    end = end_match.start() if end_match else start + max_chars
    section = text[start:end].strip()
    if len(section) < 200:
        return None
    return section[:max_chars]


def fetch_risk_factors(ticker: str) -> dict[str, Any] | None:
    """Best-effort latest-10-K Item 1A risk factors. None on any failure."""
    mapping = _load_ticker_map()
    cik = mapping.get(ticker.upper())
    if cik is None:
        logger.debug("SEC: no CIK for %s", ticker)
        return None
    pair = _latest_10k_url(cik)
    if pair is None:
        return None
    filing_date, url = pair
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("SEC 10-K fetch failed for %s: %s", ticker, e)
        return None
    section = _extract_item_1a(_strip_html(resp.text))
    if not section:
        logger.debug("SEC: Item 1A not extractable for %s", ticker)
        return None
    return {
        "ticker": ticker,
        "cik": cik,
        "filing_date": filing_date,
        "filing_url": url,
        "risk_factors": section,
    }


def batch_risk_factors(tickers: list[str]) -> dict[str, dict[str, Any]]:
    _load_ticker_map()
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, r in zip(tickers, ex.map(fetch_risk_factors, tickers)):
            if r:
                results[ticker] = r
    return results
