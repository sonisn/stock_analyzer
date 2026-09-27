"""A held stock's new SEC filings, read the evening they appear.

The weekly Saturday run (cli/filings.py) is fine for the universe; for a
stock you own, a filing shouldn't wait up to a week. The after-close run
(cli/portfolio.py --snapshot-only) checks each holding for filings from
the last few days that haven't been read yet:

  - a new 10-Q/10-K/20-F is read on the reader model (GLM-5.3), stored
    in `filing_facts` like the weekly read — so Saturday skips it — and
    summarised with what changed since the filing before;
  - an 8-K is read only when it carries a material item
    (agents/filing_reader.MATERIAL_8K_ITEMS: results, restatement,
    executive change, impairment…), together with its press release —
    where an earnings 8-K's guidance is. Stored in `eightk_alerts`.

Both go into the same evening email as the unusual-drop alert: at most
one email a day, none on a quiet one. A filing accepted after 5:30 PM is
dated the next business day by EDGAR, and anything filed after the run is
picked up the next evening — hence the look-back of a few days.
"""

from __future__ import annotations

import html
from datetime import date, timedelta
from typing import Any

from sqlmodel import select

from ..agents.filing_reader import (
    MATERIAL_8K_ITEMS,
    SECTION_CHARS,
    read_8k,
    read_filing,
)
from ..data.filing_evidence import evidence_packs
from ..data.sec_edgar import exhibit_99_text, fetch_filing_text, filing_sections, filings_since
from ..db.session import get_session
from ..db.tables import EightKAlert, FilingFacts
from ..logging import get_logger
from ..openrouter import OpenRouter
from ..serialization import dumps_compact
from ..usage import BudgetExceededError

logger = get_logger(__name__)

LOOKBACK_DAYS = 4
PERIODIC = ("10-Q", "10-K", "20-F")


def _seen(db: str) -> set[str]:
    with get_session(db) as session:
        periodic = set(session.exec(select(FilingFacts.accession)).all())
        eightks = set(session.exec(select(EightKAlert.accession)).all())
    return periodic | eightks


def new_holding_filings(
    client: OpenRouter,
    db: str,
    tickers: list[str],
    *,
    today: date,
    model: str,
) -> list[dict[str, Any]]:
    """Read the holdings' unread filings; one dict per filing for the email.
    Stops quietly at the daily OpenRouter cap (the rest comes tomorrow)."""
    from ..cli.filings import store

    seen = _seen(db)
    since = today - timedelta(days=LOOKBACK_DAYS)
    found: list[dict[str, Any]] = []
    for ticker in tickers:
        for f in filings_since(ticker, since):
            if f["accession"] in seen:
                continue
            seen.add(f["accession"])  # share classes file one document
            try:
                if f["form"] in PERIODIC:
                    item = _periodic(client, db, f, model=model, today=today, store=store)
                elif set(f["items"]) & set(MATERIAL_8K_ITEMS):
                    item = _eightk(client, db, f, model=model, today=today)
                else:
                    continue
            except BudgetExceededError as e:
                logger.warning("Holding filings: stopped at the OpenRouter cap (%s)", e)
                return found
            except Exception as e:  # noqa: BLE001 — one filing never sinks the alert
                logger.warning("Holding filing %s %s failed (%s)", ticker, f["form"], e)
                continue
            if item:
                found.append(item)
    return found


def _periodic(client, db, f, *, model, today, store) -> dict[str, Any] | None:
    text = fetch_filing_text(f["url"])
    sections = filing_sections(text or "", f["form"], max_chars=SECTION_CHARS)
    if "mda" not in sections:
        logger.info("%s %s: MD&A not found, left to the weekly read", f["ticker"], f["form"])
        return None
    read = read_filing(client, f, sections, model=model)
    store(db, read, tier="A", today=today)
    pack = evidence_packs(db, [f["ticker"]]).get(f["ticker"])
    return {"kind": "periodic", "filing": f, "pack": pack} if pack else None


def _eightk(client, db, f, *, model, today) -> dict[str, Any] | None:
    body = fetch_filing_text(f["url"])
    if not body:
        return None
    exhibit = exhibit_99_text(f)
    read = read_8k(client, f, body, exhibit, model=model)
    with get_session(db) as session:
        session.merge(
            EightKAlert(
                accession=f["accession"],
                ticker=f["ticker"],
                filed_on=f["filed_on"],
                items=",".join(f["items"]),
                url=f["url"],
                read_on=today.isoformat(),
                reader_model=model,
                summary=dumps_compact(read.summary) if read.summary else "",
                quotes_checked=read.quotes_checked,
                quotes_found=read.quotes_found,
                cost_usd=round(read.cost_usd, 6),
            )
        )
    return {"kind": "8-K", "filing": f, "summary": read.summary} if read.summary else None


# --- the email -----------------------------------------------------------------


def subject_part(item: dict[str, Any]) -> str:
    f = item["filing"]
    if item["kind"] == "periodic":
        return f"{f['ticker']} {f['form']}"
    labels = [MATERIAL_8K_ITEMS[i] for i in f["items"] if i in MATERIAL_8K_ITEMS]
    return f"{f['ticker']} 8-K ({', '.join(labels[:2])})"


def _e(x: Any) -> str:
    return html.escape(str(x or ""))


def _events(events: list[dict[str, Any]] | None) -> list[str]:
    return [
        f"<b>{'⚠ ' if e.get('severity') == 'high' else ''}{_e(e.get('issue'))}</b> "
        f"[{_e(e.get('category'))}]"
        + (f"<br><i>“{_e(e.get('quote'))}”</i>" if e.get("quote") else "")
        for e in events or []
    ]


def filing_block(item: dict[str, Any]) -> str:
    f = item["filing"]
    link = f'<a href="{_e(f["url"])}">{_e(f["form"])} filed {_e(f["filed_on"])}</a>'
    if item["kind"] == "periodic":
        p = item["pack"]
        rows = [
            f"<b>{_e(f['ticker'])}</b> — {link}, for the period ended {_e(p.get('period_end'))}"
        ]
        if p.get("summary"):
            rows.append(_e(p["summary"]))
        for name in ("guidance", "demand", "margins", "backlog", "liquidity"):
            if p.get(name):
                rows.append(f"{name.capitalize()}: {_e(p[name])}")
        if p.get("vs_prior_filing"):
            changes = {k: v for k, v in p["vs_prior_filing"].items() if k != "period_end"}
            rows.append(
                "Changed since the filing before: "
                + "; ".join(f"{_e(k)} {_e(v)}" for k, v in changes.items())
            )
        rows += _events(p.get("events"))
        rows.append(f"<small>Quotes verified in the filing: {_e(p.get('quotes_verified'))}</small>")
    else:
        s = item["summary"]
        items = ", ".join(f"{i} {MATERIAL_8K_ITEMS.get(i, '')}".strip() for i in f["items"])
        rows = [f"<b>{_e(f['ticker'])}</b> — {link} (items {_e(items)})"]
        if s.get("headline"):
            rows.append(f"<b>{_e(s['headline'])}</b>")
        if s.get("what_happened"):
            rows.append(_e(s["what_happened"]))
        g = s.get("guidance") or {}
        if g.get("direction") and g["direction"] != "none_given":
            rows.append(f"Guidance: <b>{_e(g['direction'])}</b> — {_e(g.get('detail'))}")
        for n in s.get("numbers") or []:
            rows.append(
                f"{_e(n.get('metric'))}: {_e(n.get('value'))}"
                + (f" ({_e(n.get('vs_prior'))})" if n.get("vs_prior") else "")
            )
        rows += _events(s.get("events"))
    return "<p>" + "<br>".join(rows) + "</p>"


def summary_line(item: dict[str, Any]) -> str:
    """One line for logs and tests."""
    s = item.get("pack") or item.get("summary") or {}
    return f"{subject_part(item)}: {s.get('headline') or s.get('summary') or ''}"
