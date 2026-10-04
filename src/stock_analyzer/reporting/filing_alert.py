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

from sqlalchemy import select

from ..agents.filing_reader import (
    MATERIAL_8K_ITEMS,
    read_8k,
    read_filing,
)
from ..data.filing_evidence import evidence_packs
from ..data.sec_edgar import (
    exhibit_99_text,
    fetch_filing_text,
    filings_since,
)
from ..data.sec_events import (
    EVENT_FORMS,
    PLANNED_SALE_MIN_USD,
    kind_of,
    parse_13d,
    parse_planned_sale,
    primary_xml,
    shares_change,
)
from ..data.text_change import ANNUAL_FORMS, risk_change, risk_sections
from ..db.session import get_session
from ..db.tables import EightKAlert, FilingFacts, SecEvent
from ..logging import get_logger
from ..openrouter import OpenRouter
from ..serialization import dumps_compact
from ..usage import BudgetExceededError

logger = get_logger(__name__)

LOOKBACK_DAYS = 4
PERIODIC = ("10-Q", "10-K", "20-F", "40-F")


def _seen(db: str) -> set[str]:
    with get_session(db) as session:
        periodic = set(session.scalars(select(FilingFacts.accession)).all())
        eightks = set(session.scalars(select(EightKAlert.accession)).all())
        events = set(session.scalars(select(SecEvent.accession)).all())
    return periodic | eightks | events


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
        for f in filings_since(ticker, since, forms=(*PERIODIC, "8-K", *EVENT_FORMS)):
            if f["accession"] in seen:
                continue
            seen.add(f["accession"])  # share classes file one document
            try:
                if kind_of(f["form"]):
                    item = read_event(client, db, f, model=model, today=today)
                elif f["form"] in PERIODIC:
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
    from ..cli.filings import foreign_aware_sections

    sections = foreign_aware_sections(f, text or "")
    if text and f["form"] in ANNUAL_FORMS:
        try:
            f["risk_change"] = risk_change(f, risk_sections(text, f["form"]))
        except Exception as e:  # noqa: BLE001
            logger.info("%s: risk-factor comparison failed (%s)", f["ticker"], e)
    if "mda" not in sections:
        logger.info("%s %s: MD&A not found, left to the weekly read", f["ticker"], f["form"])
        return None
    read = read_filing(client, f, sections, model=model)
    store(db, read, tier="A", today=today)
    pack = evidence_packs(db, [f["ticker"]]).get(f["ticker"])
    return {"kind": "periodic", "filing": f, "pack": pack} if pack else None


EARNINGS_ITEM = "2.02"  # Results of Operations and Financial Condition
EARNINGS_LOOKBACK_DAYS = 100  # one quarter: older releases are superseded


def earnings_releases(
    client: OpenRouter, db: str, tickers: list[str], *, today: date, model: str
) -> list[dict[str, Any]]:
    """Each ticker's latest earnings 8-K from the last quarter, read with
    its press release if not read yet — the weekly run does this for the
    stocks acted on (tier A), since a 10-Q seldom states guidance and the
    release usually does. Stored in `eightk_alerts` like a holding's, never
    emailed. Raises BudgetExceededError at the cap (the caller stops)."""
    seen = _seen(db)
    since = today - timedelta(days=EARNINGS_LOOKBACK_DAYS)
    out: list[dict[str, Any]] = []
    for ticker in tickers:
        releases = [
            f for f in filings_since(ticker, since, forms=("8-K",)) if EARNINGS_ITEM in f["items"]
        ]
        if not releases or releases[-1]["accession"] in seen:
            continue
        f = releases[-1]
        seen.add(f["accession"])
        try:
            item = _eightk(client, db, f, model=model, today=today)
        except BudgetExceededError:
            raise
        except Exception as e:  # noqa: BLE001 — one release never sinks the batch
            logger.warning("Earnings release %s %s failed (%s)", ticker, f["filed_on"], e)
            continue
        if item:
            out.append(item)
    return out


def read_event(
    client: OpenRouter, db: str, f: dict[str, Any], *, model: str, today: date
) -> dict[str, Any] | None:
    """Read and store one SEC event filing (data/sec_events); the email item
    when it meets the alert bar, else None. A Form 144 below
    PLANNED_SALE_MIN_USD and a 13D the reader doesn't call activist are
    recorded but not alerted. What the reads cost is in openrouter_spend."""
    from ..agents.event_reader import read_13d, read_offering

    kind = kind_of(f["form"]) or ""
    facts: dict[str, Any] = {}
    alert = True
    used_model = ""
    if kind in ("shelf", "offering"):
        change = shares_change(f["cik"], today=today) if f.get("cik") else None
        if change:
            facts["shares"] = change
        if kind == "offering":
            body = fetch_filing_text(f["url"])
            if body:
                facts["offering"] = read_offering(client, f, body, model=model)
                used_model = model
    elif kind == "planned_sale":
        xml = primary_xml(f)
        facts.update(parse_planned_sale(xml) if xml else {})
        alert = (facts.get("value_usd") or 0) >= PLANNED_SALE_MIN_USD
    elif kind == "activist":
        xml = primary_xml(f)
        parsed = parse_13d(xml) if xml else {}
        issuer = parsed.get("issuer_cik")
        if issuer and f.get("cik") and int(issuer) != int(f["cik"]):
            return None  # this ticker is the holder, not the target
        facts.update({k: v for k, v in parsed.items() if k not in ("purpose", "items_text")})
        if parsed:
            facts["read"] = read_13d(client, f, parsed, model=model)
            used_model = model
        # "unclear" is kept: an amendment by an activist fund often doesn't
        # restate its demands (Elliott at Seadrill, 2026-09-21). The discover
        # idea source (`activist_targets`) takes "activist" only.
        alert = ((facts.get("read") or {}).get("stance")) in ("activist", "unclear")
    with get_session(db) as session:
        session.merge(
            SecEvent(
                accession=f["accession"],
                ticker=f["ticker"],
                form=f["form"],
                kind=kind,
                filed_on=f["filed_on"],
                url=f["url"],
                read_on=today.isoformat(),
                reader_model=used_model,
                facts=dumps_compact(facts),
                alerted=alert,
            )
        )
    return {"kind": "event", "event": kind, "filing": f, "facts": facts} if alert else None


SCAN_DAYS = 5  # a weekend plus a missed night


def activist_scan(
    client: OpenRouter, db: str, tickers: set[str], *, today: date, model: str
) -> list[dict[str, Any]]:
    """Schedule 13Ds filed on any of `tickers` in the last SCAN_DAYS (EDGAR
    daily index, one request a day), each read once. Returns the activist
    ones. Nightly in earnings-watch; stops quietly at the OpenRouter cap."""
    from ..data.sec_events import recent_days, thirteen_d_targets

    seen = _seen(db)
    tried: set[tuple[str, str]] = set()  # one filing, several listed parties
    found = []
    for day in recent_days(today + timedelta(days=1), SCAN_DAYS):
        for f in thirteen_d_targets(day, tickers):
            key = (f["accession"], f["ticker"])
            if f["accession"] in seen or key in tried:
                continue
            tried.add(key)
            try:
                item = read_event(client, db, f, model=model, today=today)
            except BudgetExceededError as e:
                logger.warning("13D scan: stopped at the OpenRouter cap (%s)", e)
                return found
            except Exception as e:  # noqa: BLE001 — one filing never sinks the scan
                logger.warning("13D %s %s failed (%s)", f["ticker"], f["accession"], e)
                continue
            if item:
                found.append(item)
    return found


def activist_targets(db: str, *, today: date, days: int = 60) -> list[str]:
    """Stocks with an activist 13D filed in the last `days` — a discover
    idea source ("activist_13d"), not a score."""
    import json

    since = (today - timedelta(days=days)).isoformat()
    with get_session(db) as session:
        rows = session.execute(
            select(SecEvent.ticker, SecEvent.facts).where(
                SecEvent.kind == "activist", SecEvent.filed_on >= since
            )
        ).all()
    out = set()
    for ticker, facts in rows:
        try:
            read = (json.loads(facts or "{}") or {}).get("read") or {}
        except json.JSONDecodeError:
            continue
        if read.get("stance") == "activist":
            out.add(ticker)
    return sorted(out)


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


EVENT_LABELS = {
    "late_filing": "late-filing notice",
    "shelf": "shelf registration",
    "offering": "offering",
    "planned_sale": "planned insider sale",
    "activist": "activist stake",
}


def subject_part(item: dict[str, Any]) -> str:
    f = item["filing"]
    if item["kind"] == "event":
        return f"{f['ticker']} {EVENT_LABELS.get(item['event'], f['form'])}"
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


def _money(v: Any) -> str:
    try:
        v = float(v)
    except TypeError, ValueError:
        return "?"
    return f"${v / 1e9:,.2f}B" if v >= 1e9 else f"${v / 1e6:,.1f}M"


def event_rows(item: dict[str, Any]) -> list[str]:
    """The lines for one SEC event (data/sec_events) in the alert email."""
    f, facts, kind = item["filing"], item.get("facts") or {}, item["event"]
    link = f'<a href="{_e(f["url"])}">{_e(f["form"])} filed {_e(f["filed_on"])}</a>'
    head = f"<b>{_e(f['ticker'])}</b> — {EVENT_LABELS.get(kind, kind)}: {link}"
    rows = [head]
    if kind == "late_filing":
        rows.append(
            "⚠ The company told the SEC it cannot file its report on time — often a sign "
            "of accounting trouble. Read the notice for the reason."
        )
    elif kind == "planned_sale":
        pct = facts.get("pct_of_outstanding")
        rows.append(
            f"{_e(facts.get('seller'))} ({_e(facts.get('relationship'))}) plans to sell "
            f"{(facts.get('shares') or 0):,.0f} shares, {_money(facts.get('value_usd'))}"
            + (f" ({pct}% of the company)" if pct else "")
            + f", around {_e(facts.get('sale_date'))}"
            + (f"; acquired as {_e(facts.get('acquired_as'))}" if facts.get("acquired_as") else "")
        )
    elif kind == "activist":
        read = facts.get("read") or {}
        who = ", ".join(facts.get("holders") or []) or "a holder"
        rows.append(f"{_e(who)} — {_e(facts.get('percent'))}% of the class")
        if read.get("headline"):
            rows.append(f"<b>{_e(read['headline'])}</b>")
        if read.get("demands"):
            rows.append(f"Seeks: {_e(read['demands'])}")
        if read.get("quote"):
            rows.append(f"<i>“{_e(read['quote'])}”</i>")
    else:  # shelf / offering
        o = facts.get("offering") or {}
        if o.get("headline"):
            rows.append(f"<b>{_e(o['headline'])}</b>")
        if o:
            size = (
                _money(o["amount_usd_millions"] * 1e6)
                if o.get("amount_usd_millions")
                else "size not stated"
            )
            rows.append(
                f"{_e(o.get('security'))}{' (at-the-market program)' if o.get('at_the_market') else ''}"
                f", {size}"
                + (f"; proceeds for {_e(o['use_of_proceeds'])}" if o.get("use_of_proceeds") else "")
            )
        elif kind == "shelf":
            rows.append("Registers securities the company may sell later (no sale yet).")
        sh = facts.get("shares") or {}
        if sh.get("change_pct") is not None:
            rows.append(
                f"Shares outstanding {sh['change_pct']:+.1f}% in a year "
                f"({sh.get('year_ago', 0):,.0f} → {sh['latest']:,.0f})"
            )
    return rows


def filing_block(item: dict[str, Any]) -> str:
    f = item["filing"]
    link = f'<a href="{_e(f["url"])}">{_e(f["form"])} filed {_e(f["filed_on"])}</a>'
    if item["kind"] == "event":
        return "<p>" + "<br>".join(event_rows(item)) + "</p>"
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
    if item["kind"] == "event":
        facts = item.get("facts") or {}
        s = facts.get("read") or facts.get("offering") or {}
        if item["event"] == "planned_sale" and facts.get("seller"):
            s = {
                "headline": f"{facts['seller']} ({facts.get('relationship') or '?'}), "
                f"{_money(facts.get('value_usd'))} around {facts.get('sale_date') or '?'}"
            }
    return f"{subject_part(item)}: {s.get('headline') or s.get('summary') or ''}"
