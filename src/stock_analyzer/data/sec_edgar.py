"""SEC EDGAR client — fetch latest 10-K risk factors.

Talks via our shared http_client + regex. SEC requires a User-Agent with contact
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

from ..http_client import HttpClient, HttpClientError
from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 3   # SEC limits 10 req/sec; stay polite
_USER_AGENT = "stock-analyzer research-bot (snehal.soni@farohealth.com)"
_HEADERS = {"User-Agent": _USER_AGENT, "Accept-Encoding": "gzip, deflate"}
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# SEC enforces 10 req/sec; cap to 8/sec (= 480/min) to stay comfortably under.
_HTTP = HttpClient(
    default_headers=_HEADERS,
    timeout=30.0,
    rate_limit_per_min=480,
    name="sec-edgar",
)

_TICKER_TO_CIK: dict[str, int] | None = None


def load_ticker_cik_map() -> dict[str, int]:
    """Public alias — returns the SEC's authoritative ticker→CIK mapping.
    Used by the discover universe builder to filter regex-extracted noise."""
    return _load_ticker_map()


def _load_ticker_map() -> dict[str, int]:
    global _TICKER_TO_CIK
    if _TICKER_TO_CIK is not None:
        return _TICKER_TO_CIK
    try:
        data = _HTTP.get_json(_TICKERS_URL)
        _TICKER_TO_CIK = {
            row["ticker"].upper(): int(row["cik_str"]) for row in data.values()
        }
    except HttpClientError as e:
        logger.warning("SEC ticker map fetch failed: %s", e)
        _TICKER_TO_CIK = {}
    return _TICKER_TO_CIK


def _latest_filing_url(cik: int, form_type: str) -> tuple[str, str] | None:
    """Return (filing_date, primary_doc_url) for the latest filing of the
    given form type (e.g. '10-K' or '10-Q'), or None."""
    try:
        sub = _HTTP.get_json(_SUBMISSIONS_URL.format(cik=cik))
    except HttpClientError as e:
        logger.warning("SEC submissions fetch failed for CIK %s: %s", cik, e)
        return None
    recent = sub.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])
    dates = recent.get("filingDate", [])
    for form, acc, doc, date in zip(forms, accessions, docs, dates, strict=False):
        if form == form_type:
            acc_clean = acc.replace("-", "")
            return (
                date,
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc}",
            )
    return None


def _latest_10k_url(cik: int) -> tuple[str, str] | None:
    return _latest_filing_url(cik, "10-K")


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
        resp = _HTTP.get(url)
    except HttpClientError as e:
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
        for ticker, r in zip(tickers, ex.map(fetch_risk_factors, tickers), strict=False):
            if r:
                results[ticker] = r
    return results


# --- 10-Q MD&A (Item 2: Management's Discussion and Analysis) ---------------
# The MD&A is the current-quarter forward-looking narrative — much more
# current than the annual 10-K. We extract Item 2 between the next item
# header (typically "Item 3. Quantitative and Qualitative Disclosures").

_ITEM_2_MDA_RE = re.compile(
    r"item\s*2[.\s]+management.{0,40}discussion", re.IGNORECASE
)
_ITEM_3_RE = re.compile(r"item\s*3\.?\s+quantitative", re.IGNORECASE)
_ITEM_4_RE = re.compile(r"item\s*4\.?\s+controls", re.IGNORECASE)


def _extract_item_2_mda(text: str, max_chars: int = 6000) -> str | None:
    matches = list(_ITEM_2_MDA_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    end_match = (
        _ITEM_3_RE.search(text, start) or _ITEM_4_RE.search(text, start)
    )
    end = end_match.start() if end_match else start + max_chars
    section = text[start:end].strip()
    if len(section) < 300:
        return None
    return section[:max_chars]


def fetch_quarterly_mda(ticker: str) -> dict[str, Any] | None:
    """Best-effort latest-10-Q Item 2 MD&A. None on any failure."""
    mapping = _load_ticker_map()
    cik = mapping.get(ticker.upper())
    if cik is None:
        return None
    pair = _latest_filing_url(cik, "10-Q")
    if pair is None:
        return None
    filing_date, url = pair
    try:
        resp = _HTTP.get(url)
    except HttpClientError as e:
        logger.warning("SEC 10-Q fetch failed for %s: %s", ticker, e)
        return None
    section = _extract_item_2_mda(_strip_html(resp.text))
    if not section:
        logger.debug("SEC: Item 2 MD&A not extractable for %s", ticker)
        return None
    return {
        "ticker": ticker,
        "filing_date": filing_date,
        "filing_url": url,
        "mda": section,
    }


def batch_quarterly_mda(tickers: list[str]) -> dict[str, dict[str, Any]]:
    _load_ticker_map()
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        for ticker, r in zip(tickers, ex.map(fetch_quarterly_mda, tickers), strict=False):
            if r:
                results[ticker] = r
    return results
