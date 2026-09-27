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
from typing import Any

from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..logging import get_logger

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


def evidence_packs(db: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: pack} for the tickers with a usable stored read. Never raises:
    evidence is context, and a missing table or bad row just means none."""
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
                        "quotes_checked FROM filing_facts WHERE ticker IN "
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
    out: dict[str, dict[str, Any]] = {}
    for t, reads in by_ticker.items():
        p = pack(reads[0], reads[1] if len(reads) > 1 else None)
        if p:
            out[t] = p
    return out


def prefer_pack(pack_: dict[str, Any] | None, raw_filed_on: str | None) -> bool:
    """Use the pack unless the run fetched a newer filing than it covers."""
    if not pack_:
        return False
    return not raw_filed_on or raw_filed_on <= (pack_.get("filed_on") or "")
