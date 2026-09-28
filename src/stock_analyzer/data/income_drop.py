"""A filing whose own income numbers fell, as filed — free, from XBRL.

A bulk read that flags nothing stands as the stock's filing facts. In the
2026-09-27 pilot the bulk reader flagged nothing on PANW's 10-Q while
operating income was down 44% and net income 73%; the better reader
caught it. The income statement is tagged in every 10-Q and 10-K, so the
drop can be seen without a model: the filing's own operating and net
income against the same-length period a year earlier. A drop past
DROP_SHARE, or a swing into a loss, sends the filing to the reader model
(one ~$0.0125 read) whatever the bulk read said.

Two requests per filing to the SEC's companyconcept API. Foreign filers
(20-F, IFRS) and companies that don't tag a concept get no check, never
a penalty. A mis-tagged value (ANET's -$2,556M net income, see
discover/data_reconciliation) costs at worst one needless better read.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..logging import get_logger
from .sec_edgar import _HTTP

logger = get_logger(__name__)

_CONCEPT_URL = "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/{concept}.json"
CONCEPTS = (("OperatingIncomeLoss", "operating income"), ("NetIncomeLoss", "net income"))
DROP_SHARE = 0.25
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (350, 380)
_YEAR_AGO = (340, 390)  # period-end gap to the comparison period
# Below this a year-ago profit is too small to measure a drop against.
_MIN_BASE_USD = 5_000_000


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except TypeError, ValueError:
        return None


def _span(f: dict[str, Any]) -> tuple[date, date] | None:
    start, end = _day(f.get("start")), _day(f.get("end"))
    return None if start is None or end is None else (start, end)


def drop_in(facts: list[dict[str, Any]], accession: str) -> tuple[float, float] | None:
    """(this period, same period a year earlier) for the filing
    `accession`: its three-month figure (a 10-Q) or twelve-month one
    (a 10-K), against the latest-filed value for the matching period a
    year before. None when the filing tagged no such figure."""
    own = [f for f in facts if f.get("accn") == accession and f.get("val") is not None]
    for lo, hi in (_QUARTER_DAYS, _YEAR_DAYS):
        current = [f for f in own if (s := _span(f)) and lo <= (s[1] - s[0]).days <= hi]
        if not current:
            continue
        # The filing's own period is its latest-ending one; the older
        # columns in the same filing are the comparisons.
        cur = max(current, key=lambda f: _day(f["end"]) or date.min)
        cur_end = _day(cur["end"])
        assert cur_end is not None
        prior: dict[str, Any] | None = None
        for f in facts:
            s = _span(f)
            if s is None or f.get("val") is None or not lo <= (s[1] - s[0]).days <= hi:
                continue
            if not _YEAR_AGO[0] <= (cur_end - s[1]).days <= _YEAR_AGO[1]:
                continue
            if prior is None or str(f.get("filed", "")) > str(prior.get("filed", "")):
                prior = f
        if prior is not None:
            return float(cur["val"]), float(prior["val"])
    return None


def describe(label: str, now: float, then: float) -> str | None:
    """The reason to re-read, or None when the figure held up."""
    if then < _MIN_BASE_USD:
        return None
    if now < 0:
        return f"{label} swung to a ${now / 1e6:,.0f}M loss from ${then / 1e6:,.0f}M a year earlier"
    change = (now - then) / then
    if change <= -DROP_SHARE:
        return (
            f"{label} {change:+.0%} vs a year earlier (${now / 1e6:,.0f}M from ${then / 1e6:,.0f}M)"
        )
    return None


def income_drops(filing: dict[str, Any]) -> list[str]:
    """Why the filing's income numbers say it needs the better reader
    ([] when they held up, weren't tagged, or the SEC didn't answer)."""
    cik, acc = filing.get("cik"), filing.get("accession")
    if not cik or not acc or filing.get("form") not in ("10-Q", "10-K"):
        return []
    reasons: list[str] = []
    for concept, label in CONCEPTS:
        try:
            body = _HTTP.get_json(_CONCEPT_URL.format(cik=int(cik), concept=concept))
        except Exception as e:  # noqa: BLE001 — untagged is normal; no check, no penalty
            logger.debug("%s: no %s (%s)", filing.get("ticker"), concept, e)
            continue
        pair = drop_in((body.get("units") or {}).get("USD") or [], acc)
        if pair and (reason := describe(label, *pair)):
            reasons.append(f"filed {reason}")
    return reasons
