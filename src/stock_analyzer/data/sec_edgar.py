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
from datetime import date, timedelta
from typing import Any

from ..http_client import HttpClient, HttpClientError
from ..logging import get_logger

logger = get_logger(__name__)

_MAX_WORKERS = 3  # SEC limits 10 req/sec; stay polite
_USER_AGENT = "stock-analyzer research-bot (soni.snehal@gmail.com)"
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
        _TICKER_TO_CIK = {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}
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
    for form, acc, doc, filed in zip(forms, accessions, docs, dates, strict=False):
        if form == form_type:
            acc_clean = acc.replace("-", "")
            return (
                filed,
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{doc}",
            )
    return None


def fetch_recent_filings(
    ticker: str,
    *,
    forms: tuple[str, ...] = ("8-K", "10-Q", "10-K"),
    days: int = 30,
    today: date | None = None,
) -> list[dict[str, Any]]:
    """Filings of the given forms from the last `days` days, newest first.

    One submissions fetch per ticker. Unlike a news feed this is about the
    company by construction, which is why the daily email falls back to it
    when a holding's headlines are all syndicated commentary.
    """
    today = today or date.today()
    cik = _load_ticker_map().get(ticker.upper())
    if cik is None:
        return []
    try:
        sub = _HTTP.get_json(_SUBMISSIONS_URL.format(cik=cik))
    except HttpClientError as e:
        logger.warning("SEC submissions fetch failed for %s: %s", ticker, e)
        return []
    recent = sub.get("filings", {}).get("recent", {})
    cutoff = (today - timedelta(days=days)).isoformat()
    out: list[dict[str, Any]] = []
    for form, acc, doc, filed in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("filingDate", []),
        strict=False,
    ):
        if form not in forms or not filed or filed < cutoff:
            continue
        out.append(
            {
                "form": form,
                "filed_on": filed,
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}"
                ),
            }
        )
    return sorted(out, key=lambda f: f["filed_on"], reverse=True)


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

_ITEM_2_MDA_RE = re.compile(r"item\s*2[.\s]+management.{0,40}discussion", re.IGNORECASE)
_ITEM_3_RE = re.compile(r"item\s*3\.?\s+quantitative", re.IGNORECASE)
_ITEM_4_RE = re.compile(r"item\s*4\.?\s+controls", re.IGNORECASE)


def _extract_item_2_mda(text: str, max_chars: int = 6000) -> str | None:
    matches = list(_ITEM_2_MDA_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    end_match = _ITEM_3_RE.search(text, start) or _ITEM_4_RE.search(text, start)
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


# --- whole filings for the filing reader (agents/filing_reader.py) ----------
# The reader wants more than the 6k-char excerpts above: the full MD&A and
# the risk-factor section, with table rows kept as rows so a backlog or
# segment figure still reads as "label | value".

_DROP_BLOCKS_RE = re.compile(r"<(ix:header|script|style|head)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_BREAK_RE = re.compile(r"<br\s*/?>|</(?:p|div|tr|li|h\d|table)>", re.IGNORECASE)
_CELL_RE = re.compile(r"</t[dh]>", re.IGNORECASE)
_SPACES_RE = re.compile(r"[ \t\xa0]+")


def filing_text(raw_html: str) -> str:
    """Readable text of a filing: one line per paragraph or table row,
    cells joined by " | ", inline-XBRL header dropped."""
    body = _DROP_BLOCKS_RE.sub(" ", raw_html)
    body = _BREAK_RE.sub("\n", body)
    body = _CELL_RE.sub(" | ", body)
    body = html.unescape(_TAG_RE.sub(" ", body))
    lines = []
    for line in body.split("\n"):
        line = _SPACES_RE.sub(" ", line).strip(" |")
        # A row of empty cells or a lone page number is layout, not text.
        if len(line) > 3 and not line.isdigit():
            lines.append(line)
    return "\n".join(lines)


# Section headers by form. Each is searched from its LAST match, since the
# first is usually the table of contents; a section runs to the first end
# marker after it.
# "Item 2.", "ITEM 2 —", "Item 2. | Management" (a table cell) all appear
# in real filings.
_SEP = r"[\s.:|\-\u2013\u2014]*"


def _loose(word: str) -> str:
    """`word`, allowing one stray space between letters: filings styled
    letter by letter come out as "RIS K FACTORS" once tags are stripped."""
    return r"\s?".join(re.escape(ch) for ch in word)


_L = {
    w: _loose(w)
    for w in (
        "management",
        "discussion",
        "quantitative",
        "controls",
        "risk",
        "factors",
        "unregistered",
        "defaults",
        "other",
        "exhibits",
        "financial",
        "unresolved",
        "cybersecurity",
        "properties",
        "operating",
        "directors",
        "information",
    )
}


_ITEM = r"i\s?t\s?e\s?m"  # "Ite m 2." happens too
_HEADING = r"^\W*(?:part\s+i+\W*)?"
_SECTIONS: dict[str, dict[str, tuple[str, tuple[str, ...]]]] = {
    "10-Q": {
        "mda": (
            rf"{_ITEM}\s*2{_SEP}{_L['management']}.{{0,5}}s?\s+{_L['discussion']}",
            (rf"{_ITEM}\s*3{_SEP}{_L['quantitative']}", rf"{_ITEM}\s*4{_SEP}{_L['controls']}"),
        ),
        "risks": (
            rf"{_ITEM}\s*1a{_SEP}{_L['risk']}\s+{_L['factors']}",
            (
                rf"{_ITEM}\s*2{_SEP}{_L['unregistered']}",
                rf"{_ITEM}\s*3{_SEP}{_L['defaults']}",
                rf"{_ITEM}\s*5{_SEP}{_L['other']}",
                rf"{_ITEM}\s*6{_SEP}{_L['exhibits']}",
            ),
        ),
    },
    # Foreign private issuers' annual report: Item 5 is their MD&A, and
    # the risk factors sit under Item 3 as "D. Risk Factors".
    "20-F": {
        "mda": (
            rf"{_ITEM}\s*5{_SEP}{_L['operating']}\s+and\s+{_L['financial']}\s+review",
            (rf"{_ITEM}\s*6{_SEP}{_L['directors']}",),
        ),
        "risks": (
            rf"(?:{_ITEM}\s*3{_SEP})?d{_SEP}{_L['risk']}\s+{_L['factors']}",
            (rf"{_ITEM}\s*4{_SEP}{_L['information']}", rf"{_ITEM}\s*4a{_SEP}{_L['unresolved']}"),
        ),
    },
    "10-K": {
        "mda": (
            rf"{_ITEM}\s*7{_SEP}{_L['management']}.{{0,5}}s?\s+{_L['discussion']}",
            (rf"{_ITEM}\s*7a{_SEP}{_L['quantitative']}", rf"{_ITEM}\s*8{_SEP}{_L['financial']}"),
        ),
        "risks": (
            rf"{_ITEM}\s*1a{_SEP}{_L['risk']}\s+{_L['factors']}",
            (
                rf"{_ITEM}\s*1b{_SEP}{_L['unresolved']}",
                rf"{_ITEM}\s*1c{_SEP}{_L['cybersecurity']}",
                rf"{_ITEM}\s*2{_SEP}{_L['properties']}",
            ),
        ),
    },
}


# When no numbered heading leads to a real section: a short line that IS the
# section's title, with no "Item" before it. Some filings number only the
# table of contents (DRVN, DBD), so the numbered match finds the contents
# line and nothing after it.
_UNNUMBERED = {
    "mda": rf"{_L['management']}.{{0,5}}s?\s+{_L['discussion']}\s+and\s+analysis[^\n]{{0,100}}$",
}
_UNNUMBERED_ENDS = (
    rf"{_L['quantitative']}\s+and\s+qualitative\s+disclosures?[^\n]{{0,80}}$",
    rf"{_L['controls']}\s+and\s+procedures[^\n]{{0,40}}$",
)


def _cut(text: str, head: str, ends: tuple[str, ...]) -> str | None:
    """The text after the LAST `head` that is followed by a real section
    (a 10-Q's "Item 1A" in Part II often just points at the 10-K), up to
    the first end marker. Headings start a line: "see Item 3 - Quantitative
    ... included in" mid-sentence is a cross-reference, and ending there
    cut Hershey's MD&A to 3k characters."""
    flags = re.IGNORECASE | re.MULTILINE
    enders = [re.compile(_HEADING + end, flags) for end in ends]
    for m in reversed(list(re.finditer(_HEADING + head, text, flags))):
        stop = len(text)
        for e in enders:
            found = e.search(text, m.end())
            if found:
                stop = min(stop, found.start())
        section = text[m.end() : stop].strip()
        if len(section) >= 400:
            return section
    return None


def filing_sections(text: str, form: str, *, max_chars: dict[str, int]) -> dict[str, str]:
    """{"mda": ..., "risks": ...} cut from a filing's text, each capped at
    `max_chars[name]`. A section whose header isn't found (or is under a
    few hundred characters, i.e. only a cross-reference) is left out."""
    out: dict[str, str] = {}
    for name, (head, ends) in _SECTIONS.get(form.replace("/A", ""), {}).items():
        section = _cut(text, head, ends)
        if section is None and name in _UNNUMBERED:
            section = _cut(text, _UNNUMBERED[name], (*ends, *_UNNUMBERED_ENDS))
        if section:
            out[name] = section[: max_chars.get(name, 40_000)]
    return out


def latest_filing(
    ticker: str, forms: tuple[str, ...] = ("10-Q", "10-K", "20-F")
) -> dict[str, Any] | None:
    """The newest filing of `forms`: accession, form, filed_on, period_end
    and the primary document's URL. None when the ticker has no CIK or the
    SEC is unreachable."""
    cik = _load_ticker_map().get(ticker.upper())
    if cik is None:
        return None
    try:
        sub = _HTTP.get_json(_SUBMISSIONS_URL.format(cik=cik))
    except HttpClientError as e:
        logger.warning("SEC submissions fetch failed for %s: %s", ticker, e)
        return None
    recent = sub.get("filings", {}).get("recent", {})
    for form, acc, doc, filed, period in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        strict=False,
    ):
        if form in forms:
            return {
                "ticker": ticker.upper(),
                "cik": cik,
                "accession": acc,
                "form": form,
                "filed_on": filed,
                "period_end": period or None,
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
            }
    return None


def fetch_filing_text(url: str) -> str | None:
    try:
        return filing_text(_HTTP.get(url).text)
    except HttpClientError as e:
        logger.warning("SEC filing fetch failed (%s): %s", url, e)
        return None


# --- a holding's new filings (reporting/filing_alert.py) --------------------


def filings_since(
    ticker: str,
    since: date,
    *,
    forms: tuple[str, ...] = ("10-Q", "10-K", "20-F", "8-K"),
) -> list[dict[str, Any]]:
    """Filings of `forms` dated `since` or later, oldest first, each with
    its accession, 8-K item codes ("2.02", "5.02", …) and primary-document
    URL. [] on any failure: a missed day is picked up by the next run."""
    cik = _load_ticker_map().get(ticker.upper())
    if cik is None:
        return []
    try:
        sub = _HTTP.get_json(_SUBMISSIONS_URL.format(cik=cik))
    except HttpClientError as e:
        logger.warning("SEC submissions fetch failed for %s: %s", ticker, e)
        return []
    recent = sub.get("filings", {}).get("recent", {})
    out = []
    for form, acc, doc, filed, period, items in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
        recent.get("filingDate", []),
        recent.get("reportDate", []),
        recent.get("items", []),
        strict=False,
    ):
        if form not in forms or not filed or filed < since.isoformat():
            continue
        out.append(
            {
                "ticker": ticker.upper(),
                "cik": cik,
                "accession": acc,
                "form": form,
                "filed_on": filed,
                "period_end": period or None,
                "items": [i.strip() for i in (items or "").split(",") if i.strip()],
                "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/{doc}",
            }
        )
    return sorted(out, key=lambda f: f["filed_on"])


# "ex99", "ex-99.1", "exhibit991" (Tesla) all name the press release.
_EX99_RE = re.compile(r"ex(?:hibit)?[-_]?99", re.IGNORECASE)


def exhibit_99_text(filing: dict[str, Any]) -> str | None:
    """The text of an 8-K's press release — Exhibit 99, where an earnings
    8-K's guidance is — or None when it has none. Found by name ("ex99",
    "ex-99.1"); failing that, the largest other document in the filing
    (NVIDIA files it as "q2fy27pr.htm")."""
    folder, primary = filing["url"].rsplit("/", 1)
    try:
        index = _HTTP.get_json(f"{folder}/index.json")
    except HttpClientError as e:
        logger.info("No filing index for %s (%s)", filing["accession"], e)
        return None
    docs = [
        i
        for i in index.get("directory", {}).get("item", [])
        if i.get("name", "").lower().endswith((".htm", ".html"))
        and i["name"] != primary
        and not re.fullmatch(r"R\d+\.html?", i["name"])  # XBRL viewer pages
        and "-index" not in i["name"]  # EDGAR's own index pages
    ]
    named = sorted(i["name"] for i in docs if _EX99_RE.search(i["name"]))
    if named:
        return fetch_filing_text(f"{folder}/{named[0]}")
    if not docs:
        return None

    def size(i: dict[str, Any]) -> int:
        try:
            return int(i.get("size") or 0)
        except ValueError:
            return 0

    return fetch_filing_text(f"{folder}/{max(docs, key=size)['name']}")
