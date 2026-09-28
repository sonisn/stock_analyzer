"""SEC filings beyond the 10-Q/10-K that an owner should hear about.

All free from EDGAR; a model is used only to summarise the rare filing
whose meaning isn't in its structured fields.

  - late filing (NT 10-Q / NT 10-K / NT 20-F): the company can't file on
    time — rare, and a well-known red flag. No model.
  - dilution: a shelf registration (S-3 / S-3ASR) or an offering
    (424B5 prospectus supplement: an at-the-market program, a stock sale,
    or a debt or convertible issue — the reader says which, and how much),
    with the share count's change over a year from XBRL (`shares_change`).
  - planned insider sale (Form 144): who, how many shares, what value,
    when — parsed from the filing's XML, filed before the Form 4 that
    reports the sale. Only sales worth PLANNED_SALE_MIN_USD or more.
  - activist stake (Schedule 13D): a 5%+ holder who may seek change must
    file within days. Percent and the stated purpose (Item 4) are
    structured; the reader classifies the purpose as activist or not.

Holdings get these in the evening alert (reporting/filing_alert.py). The
13D scan also runs over the whole universe from EDGAR's daily index
(`thirteen_d_targets`, nightly in earnings-watch): an activist target
becomes a discover idea source ("activist_13d", no score bonus), like an
insider-buying cluster.
"""

from __future__ import annotations

import html
import re
from datetime import date, timedelta
from typing import Any

from ..http_client import HttpClientError
from ..logging import get_logger
from .sec_edgar import _HTTP, load_ticker_cik_map

logger = get_logger(__name__)

LATE_FORMS = ("NT 10-Q", "NT 10-K", "NT 20-F")
SHELF_FORMS = ("S-3", "S-3ASR")
OFFERING_FORMS = ("424B5",)
PLANNED_SALE_FORMS = ("144",)
THIRTEEN_D_FORMS = ("SCHEDULE 13D", "SCHEDULE 13D/A", "SC 13D", "SC 13D/A")
EVENT_FORMS = (*LATE_FORMS, *SHELF_FORMS, *OFFERING_FORMS, *PLANNED_SALE_FORMS, *THIRTEEN_D_FORMS)

# AVGO alone files ~35 Form 144s a year; below this they are noise.
PLANNED_SALE_MIN_USD = 1_000_000

_DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{d}.idx"
_SHARES_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/dei/"
    "EntityCommonStockSharesOutstanding.json"
)


def kind_of(form: str) -> str | None:
    if form in LATE_FORMS:
        return "late_filing"
    if form in SHELF_FORMS:
        return "shelf"
    if form in OFFERING_FORMS:
        return "offering"
    if form in PLANNED_SALE_FORMS:
        return "planned_sale"
    if form in THIRTEEN_D_FORMS:
        return "activist"
    return None


def _folder(filing: dict[str, Any]) -> str:
    """The filing's directory. Its primary document can sit one level
    down, under a stylesheet folder ("xslF144X01/primary_doc.xml")."""
    m = re.match(r"(.*/\d{18})/", filing["url"])
    return m.group(1) if m else filing["url"].rsplit("/", 1)[0]


def primary_xml(filing: dict[str, Any]) -> str | None:
    """The filing's structured XML (Form 144, the new Schedule 13D)."""
    try:
        return _HTTP.get(f"{_folder(filing)}/primary_doc.xml").text
    except HttpClientError as e:
        logger.info("No primary_doc.xml for %s (%s)", filing.get("accession"), e)
        return None


def _tag(xml: str, name: str) -> str | None:
    m = re.search(rf"<(?:\w+:)?{name}>(.*?)</(?:\w+:)?{name}>", xml, re.S)
    return html.unescape(re.sub(r"\s+", " ", m.group(1)).strip()) if m else None


def _num(value: str | None) -> float | None:
    try:
        return float(str(value).replace(",", ""))
    except TypeError, ValueError:
        return None


def parse_planned_sale(xml: str) -> dict[str, Any]:
    """Form 144: the seller, their relationship, the sale's size and date."""
    shares, outstanding = _num(_tag(xml, "noOfUnitsSold")), _num(_tag(xml, "noOfUnitsOutstanding"))
    return {
        "seller": _tag(xml, "nameOfPersonForWhoseAccountTheSecuritiesAreToBeSold"),
        "relationship": _tag(xml, "relationshipToIssuer"),
        "shares": shares,
        "value_usd": _num(_tag(xml, "aggregateMarketValue")),
        "pct_of_outstanding": (
            round(100 * shares / outstanding, 3) if shares and outstanding else None
        ),
        "sale_date": _tag(xml, "approxSaleDate"),
        "acquired_as": _tag(xml, "natureOfAcquisitionTransaction"),
    }


def parse_13d(xml: str) -> dict[str, Any]:
    """New-format Schedule 13D: who, how much of the class, and why."""
    names = list(
        dict.fromkeys(
            html.unescape(re.sub(r"\s+", " ", n).strip())
            for n in re.findall(r"<reportingPersonName>(.*?)</reportingPersonName>", xml, re.S)
        )
    )
    percents = [
        p
        for p in (_num(x) for x in re.findall(r"<percentOfClass>(.*?)</percentOfClass>", xml))
        if p is not None
    ]
    # An amendment often leaves Item 4 (purpose) out; Items 1-7 together
    # (funding, swaps, agreements) still say what the holder is doing.
    items = re.search(r"<items1To7>(.*?)</items1To7>", xml, re.S)
    body = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", items.group(1))).strip() if items else ""
    return {
        "holders": names,
        "percent": max(percents) if percents else None,
        "purpose": (_tag(xml, "transactionPurpose") or "")[:4000],
        "items_text": body[:6000],
        "event_date": _tag(xml, "dateOfEvent"),
        "issuer": _tag(xml, "issuerName"),
        "issuer_cik": int(_num(_tag(xml, "issuerCIK")) or 0) or None,
    }


def shares_change(cik: int, *, today: date) -> dict[str, Any] | None:
    """Shares outstanding now vs a year earlier, from the cover-page XBRL
    of the company's own filings ({latest, year_ago, change_pct}), or None."""
    try:
        body = _HTTP.get_json(_SHARES_URL.format(cik=int(cik)))
    except Exception as e:  # noqa: BLE001 — not tagged (foreign filers) is normal
        logger.debug("No share count for CIK %s (%s)", cik, e)
        return None
    facts = sorted(
        (f for f in (body.get("units") or {}).get("shares") or [] if f.get("end") and f.get("val")),
        key=lambda f: (f["end"], f.get("filed", "")),
    )
    if not facts:
        return None
    latest = facts[-1]
    end = date.fromisoformat(latest["end"])
    if (today - end).days > 200:
        return None
    prior = [f for f in facts if 330 <= (end - date.fromisoformat(f["end"])).days <= 400]
    out = {"latest": float(latest["val"]), "as_of": latest["end"]}
    if prior:
        then = float(prior[-1]["val"])
        out["year_ago"] = then
        out["change_pct"] = round(100 * (out["latest"] - then) / then, 1) if then else None
    return out


def thirteen_d_targets(day: date, tickers: set[str]) -> list[dict[str, Any]]:
    """Schedule 13D filings on `day` whose subject company is in `tickers`,
    from EDGAR's daily form index (one request). Each filing is listed once
    per party: the row whose CIK maps to a ticker is the target, the rest
    are the holders."""
    q = (day.month - 1) // 3 + 1
    url = _DAILY_INDEX.format(y=day.year, q=q, d=day.strftime("%Y%m%d"))
    try:
        text = _HTTP.get(url).text
    except HttpClientError as e:
        # No index on weekends and holidays (EDGAR answers 403 or 404).
        if day.weekday() < 5 and getattr(e, "status", None) not in (403, 404):
            logger.warning("EDGAR daily index %s unavailable (%s)", day, e)
        return []
    by_cik = {cik: t for t, cik in load_ticker_cik_map().items() if t in tickers}
    rows: dict[str, list[tuple[str, int]]] = {}
    forms: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith(THIRTEEN_D_FORMS):
            continue
        m = re.match(r"(.+?)\s{2,}(.+?)\s{2,}(\d+)\s+(\d{8})\s+(\S+)", line)
        if not m:
            continue
        form, name, cik, _, path = m.groups()
        acc = path.rsplit("/", 1)[-1].removesuffix(".txt")
        rows.setdefault(acc, []).append((name.strip(), int(cik)))
        forms[acc] = form.strip()
    out = []
    for acc, parties in rows.items():
        # Usually one party maps to a ticker. When the holder is a listed
        # company too (Wells Fargo filing on a fund, Hafnia on TORM), every
        # candidate is returned and `read_event` keeps the one the filing
        # names as its issuer.
        for _, cik in [(n, c) for n, c in parties if c in by_cik]:
            out.append(
                {
                    "ticker": by_cik[cik],
                    "cik": cik,
                    "accession": acc,
                    "form": forms[acc],
                    "filed_on": day.isoformat(),
                    "holders": [n for n, c in parties if c != cik],
                    "url": f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/primary_doc.xml",
                }
            )
    return out


def recent_days(today: date, days: int) -> list[date]:
    return [today - timedelta(days=i) for i in range(days, 0, -1)]
