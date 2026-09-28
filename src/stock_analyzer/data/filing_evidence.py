"""The SEC-filing evidence the deciding models see (`sec_filing`).

What `read-filings` stored (table `filing_facts`), cut down to a pack of
~400-600 tokens per stock: every field's direction and one-line detail,
the key risks, one-off items, and each reported event with its quote.
It stands in for the raw `quarterly_mda` / `risk_factors_10k` excerpts,
which were the first 4,000 / 3,500 characters of each section — mostly
safe-harbour boilerplate, while the pack is read from the whole MD&A and
risk-factor section.

A pack is only used when it is at least as new as the filing the run
fetched itself (`prefer_pack`): a 10-Q filed today and not yet read keeps
the raw excerpt for that stock.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import Any

from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..logging import get_logger
from .text_change import describe as describe_change

logger = get_logger(__name__)

_DIRECTED = ("guidance", "demand", "margins", "backlog")
_CHANGE_FIELDS = ("guidance", "demand", "margins", "backlog", "tone")


def _line(field: dict[str, Any] | None, key: str = "direction") -> str | None:
    if not isinstance(field, dict):
        return None
    head = field.get(key) or field.get("change")
    detail = (field.get("detail") or "").strip()
    if key == "change" and field.get("value_usd_millions") is not None:
        detail = f"{detail} (${field['value_usd_millions']:,}M)".strip()
    # "none given" / "not disclosed" is the filing saying nothing: leave the
    # field out rather than spend the deciding models' tokens on it.
    if head in ("not_disclosed", "none_given") or (head in (None, "unclear") and not detail):
        return None
    return f"{head}: {detail}" if head and detail else (head or detail)


def _direction(facts: dict[str, Any], name: str) -> Any:
    f = facts.get(name)
    if name == "tone":
        return f
    if not isinstance(f, dict):
        return None
    return f.get("change") if name == "backlog" else f.get("direction")


def pack(row: dict[str, Any], prior: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """One stock's evidence from its latest stored read (and the one before,
    for what changed). None when the read had no usable answer."""
    try:
        facts = json.loads(row["facts"] or "null")
    except json.JSONDecodeError:
        return None
    if not isinstance(facts, dict):
        return None
    liquidity = facts.get("liquidity") or {}
    out: dict[str, Any] = {
        "form": row["form"],
        "period_end": row["period_end"],
        "filed_on": row["filed_on"],
        "url": row["url"],
        "quotes_verified": f"{row['quotes_found']}/{row['quotes_checked']}",
        "summary": facts.get("summary"),
        "tone": facts.get("tone"),
    }
    for name in _DIRECTED:
        line = _line(facts.get(name), "change" if name == "backlog" else "direction")
        if line:
            out[name] = line
    if isinstance(liquidity, dict) and (liquidity.get("detail") or liquidity.get("concern")):
        flag = "CONCERN" if liquidity.get("concern") else "no concern"
        out["liquidity"] = f"{flag}: {liquidity.get('detail') or ''}".strip(": ")
    capital = (facts.get("capital_return") or {}).get("detail")
    if capital:
        out["capital_return"] = capital
    change = describe_change({"kept": row.get("risk_kept"), "prior_filed_on": "prior"})
    if change:
        out["risk_factors_vs_last_year"] = change
    out["key_risks"] = [
        r["risk"] for r in facts.get("key_risks") or [] if isinstance(r, dict) and r.get("risk")
    ]
    out["one_offs"] = [
        o["item"] for o in facts.get("one_offs") or [] if isinstance(o, dict) and o.get("item")
    ]
    # Reported events keep their quote: they are what a decision may turn on.
    out["events"] = [
        {k: c.get(k) for k in ("issue", "category", "severity", "quote") if c.get(k)}
        for c in facts.get("caveats") or []
        if isinstance(c, dict) and c.get("issue")
    ]
    if prior is not None:
        try:
            before = json.loads(prior["facts"] or "null")
        except json.JSONDecodeError:
            before = None
        if isinstance(before, dict):
            changed = {
                name: f"{_direction(before, name)} -> {_direction(facts, name)}"
                for name in _CHANGE_FIELDS
                if _direction(before, name) != _direction(facts, name)
                and _direction(facts, name) is not None
            }
            if changed:
                out["vs_prior_filing"] = {"period_end": prior["period_end"], **changed}
    return {k: v for k, v in out.items() if v not in (None, [], "")}


def _latest_reads(db: str, tickers: list[str]) -> dict[str, list[dict[str, Any]]]:
    """{ticker: stored reads, newest first}. Never raises: a missing table
    or bad row just means none."""
    if not tickers:
        return {}
    wanted = sorted({t.upper() for t in tickers})
    try:
        with get_session(db) as session:
            rows = (
                exec_sql(
                    session,
                    text(
                        "SELECT ticker, form, filed_on, period_end, url, facts, quotes_found, "
                        "quotes_checked, reader_model, flag_reasons, risk_kept, risk_cosine FROM filing_facts WHERE ticker IN "
                        f"({', '.join(f':t{i}' for i in range(len(wanted)))}) "
                        "ORDER BY ticker, filed_on DESC"
                    ),
                    {f"t{i}": t for i, t in enumerate(wanted)},
                )
                .mappings()
                .all()
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("SEC filing evidence unavailable (%s)", e)
        return {}
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_ticker.setdefault(r["ticker"], []).append(dict(r))
    return by_ticker


def earnings_releases(db: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: its latest stored earnings 8-K (item 2.02), cut down to the
    headline, guidance and reported numbers}. The deciding models get it as
    `earnings_release` beside `sec_filing`: a 10-Q seldom states guidance,
    the release usually does. Never raises."""
    if not tickers:
        return {}
    wanted = sorted({t.upper() for t in tickers})
    try:
        with get_session(db) as session:
            rows = (
                exec_sql(
                    session,
                    text(
                        "SELECT ticker, filed_on, url, summary FROM eightk_alerts "
                        "WHERE items LIKE '%2.02%' AND summary != '' AND ticker IN "
                        f"({', '.join(f':t{i}' for i in range(len(wanted)))}) "
                        "ORDER BY ticker, filed_on DESC"
                    ),
                    {f"t{i}": t for i, t in enumerate(wanted)},
                )
                .mappings()
                .all()
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Earnings releases unavailable (%s)", e)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["ticker"] in out:
            continue
        try:
            s = json.loads(r["summary"])
        except json.JSONDecodeError:
            continue
        if not isinstance(s, dict):
            continue
        release: dict[str, Any] = {"filed_on": r["filed_on"], "headline": s.get("headline")}
        g = s.get("guidance") or {}
        if isinstance(g, dict) and g.get("direction") not in (None, "none_given"):
            release["guidance"] = f"{g['direction']}: {g.get('detail') or ''}".strip(": ")
        release["numbers"] = [
            " ".join(str(n.get(k)) for k in ("metric", "value", "vs_prior") if n.get(k))
            for n in s.get("numbers") or []
            if isinstance(n, dict)
        ]
        out[r["ticker"]] = {k: v for k, v in release.items() if v not in (None, [], "")}
    return out


def evidence_packs(db: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: pack} for the tickers with a usable stored read. Never raises:
    evidence is context, and a missing table or bad row just means none."""
    out: dict[str, dict[str, Any]] = {}
    for t, reads in _latest_reads(db, tickers).items():
        p = pack(reads[0], reads[1] if len(reads) > 1 else None)
        if p:
            out[t] = p
    return out


# The events the screen scores (screen._score_filing_flags). Only a
# high-severity report counts: on the 2026-09-27 universe read, 1 of the 14
# high ones was wrong (DSGX's 2004 restatement, still in its risk factors),
# against ~1 in 3 of the medium ones — immaterial revisions called
# restatements, a tenant's going-concern doubt, a remediated weakness.
RED_FLAG_CATEGORIES = ("going_concern", "material_weakness", "restatement")
# A filing older than this has been superseded, or the company stopped filing.
RED_FLAG_MAX_AGE_DAYS = 200


def filing_features(db: str, tickers: list[str], *, today: date) -> dict[str, dict[str, float]]:
    """{ticker: numeric facts from its latest filing}, stored with each
    screened candidate (CandidateSnapshot) so a later study can test every
    category — not only the ones the screen scores — against returns:

      filing_<category>       reported events of that category, any severity
      filing_<category>_high  the high-severity ones (what the screen scores)
      filing_income_drop      1 when filed operating/net income fell 25%+
                              (data/income_drop; only checked on reads
                              since 2026-09-27)

    Only tickers with a read under RED_FLAG_MAX_AGE_DAYS old. Never raises."""
    from ..agents.filing_reader import CAVEAT_CATEGORIES

    cutoff = (today - timedelta(days=RED_FLAG_MAX_AGE_DAYS)).isoformat()
    out: dict[str, dict[str, float]] = {}
    for t, reads in _latest_reads(db, tickers).items():
        row = reads[0]
        if (row["filed_on"] or "") < cutoff:
            continue
        try:
            facts = json.loads(row["facts"] or "null")
        except json.JSONDecodeError:
            continue
        if not isinstance(facts, dict):
            continue
        data: dict[str, float] = {}
        for c in facts.get("caveats") or []:
            cat = c.get("category") if isinstance(c, dict) else None
            if cat not in CAVEAT_CATEGORIES:
                continue
            data[f"filing_{cat}"] = data.get(f"filing_{cat}", 0) + 1
            if c.get("severity") == "high":
                data[f"filing_{cat}_high"] = data.get(f"filing_{cat}_high", 0) + 1
        data["filing_income_drop"] = float("filed " in (row.get("flag_reasons") or ""))
        for key in ("risk_kept", "risk_cosine"):
            if row.get(key) is not None:
                data[f"filing_{key}"] = float(row[key])
        out[t] = data
    return out


def red_flags_from(features: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """{ticker: the scored red-flag categories} out of `filing_features`."""
    out: dict[str, list[str]] = {}
    for t, data in features.items():
        cats = [c for c in RED_FLAG_CATEGORIES if data.get(f"filing_{c}_high")]
        if cats:
            out[t] = cats
    return out


def red_flags(db: str, tickers: list[str], *, today: date) -> dict[str, list[str]]:
    """{ticker: red-flag categories in its latest filing}, for the tickers
    that have any. Never raises."""
    return red_flags_from(filing_features(db, tickers, today=today))


def prefer_pack(pack_: dict[str, Any] | None, raw_filed_on: str | None) -> bool:
    """Use the pack unless the run fetched a newer filing than it covers."""
    if not pack_:
        return False
    return not raw_filed_on or raw_filed_on <= (pack_.get("filed_on") or "")
