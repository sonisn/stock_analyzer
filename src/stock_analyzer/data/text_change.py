"""How much a 10-K's risk factors changed from the year before.

"Lazy Prices" (Cohen, Malloy & Nguyen): companies whose annual report
changes a lot, risk factors above all, tend to do worse afterwards, and
the market is slow to notice. Tested here 2026-09-28 on the S&P 500's
10-Ks since 2012 (484 companies, 141 months): the direction held in both
halves, but the effect was too weak to score (risk-factor cosine, 126-day
IC +0.027, t 1.6 after overlap). The paper's effect sits mostly in
smaller companies, so it is recorded for every 10-K read across the $2B+
universe — shown to the deciding models as a fact and kept with each
screened candidate — to be tested again on that wider set.

Two measures, as in the study:
  kept    share of this year's risk-factor sentences (8+ words) found
          verbatim in last year's — readable: typical 0.70, bottom tenth
          under 0.48 (S&P 500 10-Ks since 2020)
  cosine  word-count cosine similarity — the measure that tested best
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from ..logging import get_logger
from .sec_edgar import fetch_filing_text, filing_sections, latest_filings

logger = get_logger(__name__)

ANNUAL_FORMS = ("10-K", "20-F", "40-F")
TYPICAL_KEPT = 0.70
LOW_KEPT = 0.48  # the bottom tenth
RISK_CHARS = 400_000  # the whole section; a 10-K's runs to ~100k
_WORD = re.compile(r"[a-z]{3,}")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str) -> set[str]:
    out = set()
    for s in _SENT.split(text):
        words = _WORD.findall(s.lower())
        if len(words) >= 8:
            out.add(" ".join(words))
    return out


def _cosine(a: Counter, b: Counter) -> float | None:
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if not na or not nb:
        return None
    return sum(v * b.get(k, 0) for k, v in a.items()) / (na * nb)


def compare(now: str, before: str) -> dict[str, float | None] | None:
    """{kept, cosine} of two risk-factor sections, or None if either is
    too short to measure."""
    s_now, s_before = _sentences(now), _sentences(before)
    if len(s_now) < 10 or len(s_before) < 10:
        return None
    cos = _cosine(Counter(_WORD.findall(now.lower())), Counter(_WORD.findall(before.lower())))
    return {
        "kept": round(len(s_now & s_before) / len(s_now), 3),
        "cosine": round(cos, 5) if cos is not None else None,
    }


def risk_sections(text: str, form: str) -> str:
    return filing_sections(text, form, max_chars={"mda": 10, "risks": RISK_CHARS}).get("risks", "")


def risk_change(filing: dict[str, Any], risks_now: str) -> dict[str, Any] | None:
    """The filing's risk factors against the same form a year before (one
    SEC request for the list, one for the document); None for a 10-Q, a
    first annual report, or a section that can't be cut."""
    if filing.get("form") not in ANNUAL_FORMS or not risks_now:
        return None
    earlier = [
        f
        for f in latest_filings(filing["ticker"], 3, forms=(filing["form"],))
        if f["accession"] != filing["accession"] and f["filed_on"] < filing["filed_on"]
    ]
    if not earlier:
        return None
    text = fetch_filing_text(earlier[0]["url"])
    if not text:
        return None
    out = compare(risks_now, risk_sections(text, earlier[0]["form"]))
    if out is None:
        return None
    return {**out, "prior_filed_on": earlier[0]["filed_on"]}


def describe(change: dict[str, Any] | None) -> str | None:
    """One line for the deciding models and the email."""
    if not change or change.get("kept") is None:
        return None
    kept = change["kept"]
    verdict = (
        "heavily rewritten (bottom tenth)"
        if kept < LOW_KEPT
        else "more changed than usual"
        if kept < TYPICAL_KEPT - 0.1
        else "about as usual"
        if kept < TYPICAL_KEPT + 0.08
        else "little changed"
    )
    return (
        f"{kept:.0%} of risk-factor sentences carried over verbatim from the "
        f"{change.get('prior_filed_on', 'prior')} filing (typical {TYPICAL_KEPT:.0%}): {verdict}"
    )
