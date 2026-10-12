"""Company events for the watched stocks (EVENT_WATCH_TICKERS), in the
evening holding alert: a new customer, an approval, a contract, a denial.

Each evening (cli/portfolio.holding_alerts) for every watched ticker:
  1. the last few days of Google News headlines naming the company that
     look like an event (data/event_news.looks_like_event);
  2. the ones not seen before are read in ONE call on the bulk open model
     (OPENROUTER_BULK_MODEL, GLM-5.3-Flash, ~$0.0005), which keeps only
     those that report something happening to the company — not a sector
     move, a price move or commentary — groups repeats of the same event,
     and says in a line what happened and why it could matter;
  3. every headline read is stored in `watch_news`, so none is read or
     emailed twice.

Headlines only: Google News links are redirects, and a line from the
headline is enough to decide whether to open it. The model is told to
report only what a headline says.
"""

from __future__ import annotations

import hashlib
import html
import re
from datetime import date
from typing import Any

from sqlmodel import select

from ..agents.filing_reader import READER_EXTRA
from ..data.event_news import fetch_events
from ..db.session import get_session
from ..db.tables import WatchNews
from ..logging import get_logger
from ..openrouter import OpenRouter, parse_json_object
from ..usage import BudgetExceededError

logger = get_logger(__name__)

MAX_TOKENS = 4_000
KINDS = (
    "customer",
    "regulatory",
    "contract",
    "partnership",
    "milestone",
    "financing",
    "legal",
    "management",
    "results",
    "other",
)

INSTRUCTIONS = f"""\
You screen news headlines about one company for a long-term (3-5 year)
shareholder. You see headlines only. Report only what a headline itself
says; never add facts, numbers or names it does not contain.

Keep a headline only when it reports something that HAPPENED TO OR WAS
DONE BY THIS company: a customer or order, a regulator's or agency's
decision, a contract, a partnership, a project milestone, a financing or
share sale, a lawsuit, a management change, results. Drop: price moves,
sector moves driven by other companies, analyst ratings and targets,
rankings, predictions and opinion pieces.

Several headlines about the same event are one event: keep the clearest,
list the others in "same_as".

Return ONE JSON object, no prose:
{{"events": [{{"i": <headline number>, "same_as": [<numbers>],
  "kind": "{"|".join(KINDS)}",
  "direction": "positive|negative|mixed",
  "what": "<=25 words: what happened, from the headline",
  "why": "<=30 words: why it could matter to the business over years"}}]}}
Return {{"events": []}} when nothing qualifies."""


def _key(ticker: str, title: str) -> str:
    norm = re.sub(r"\W+", " ", title.lower()).strip()
    return hashlib.sha1(f"{ticker}|{norm}".encode()).hexdigest()


def _seen(db: str, keys: list[str]) -> set[str]:
    if not keys:
        return set()
    with get_session(db) as session:
        rows = session.exec(
            select(WatchNews.key).where(WatchNews.key.in_(keys))  # ty: ignore[unresolved-attribute]
        ).all()
    return set(rows)


def read_headlines(
    client: OpenRouter, ticker: str, name: str, items: list[dict[str, Any]], *, model: str
) -> tuple[list[dict[str, Any]], float]:
    """(events kept, cost). Each event is its headline item plus kind,
    direction, what, why and the repeats (`also`)."""
    lines = [
        f"{n}. [{i['published'] or '?'}] {i['source']}: {i['title']}"
        for n, i in enumerate(items, 1)
    ]
    prompt = f"Company: {name} ({ticker})\n\nHeadlines:\n" + "\n".join(lines)
    reply = client.complete(
        "EventWatch", model, INSTRUCTIONS, prompt, max_tokens=MAX_TOKENS, extra=READER_EXTRA
    )
    parsed = parse_json_object(reply.text) or {}
    out = []
    for e in parsed.get("events") or []:
        if not isinstance(e, dict):
            continue
        try:
            n = int(e.get("i"))
        except TypeError, ValueError:
            continue
        if not 1 <= n <= len(items):
            continue
        also = []
        for m in e.get("same_as") or []:
            try:
                m = int(m)
            except TypeError, ValueError:
                continue
            if 1 <= m <= len(items) and m != n:
                also.append(items[m - 1])
        out.append(
            {
                **items[n - 1],
                "ticker": ticker,
                "kind": e.get("kind") if e.get("kind") in KINDS else "other",
                "direction": e.get("direction") or "",
                "what": str(e.get("what") or "")[:300],
                "why": str(e.get("why") or "")[:300],
                "also": also,
            }
        )
    return out, reply.cost_usd


def new_watch_events(
    client: OpenRouter,
    db: str,
    names: dict[str, str],
    *,
    today: date,
    model: str,
) -> list[dict[str, Any]]:
    """The watched companies' new events. `names` maps ticker -> company
    name. Stops quietly at the OpenRouter cap; one failing ticker never
    stops the others."""
    found: list[dict[str, Any]] = []
    for ticker, name in names.items():
        items = fetch_events(name, today=today)
        keys = {_key(ticker, i["title"]): i for i in items}
        seen = _seen(db, list(keys))
        fresh = [i for k, i in keys.items() if k not in seen]
        if not fresh:
            continue
        try:
            events, cost = read_headlines(client, ticker, name, fresh, model=model)
        except BudgetExceededError as e:
            logger.warning("Event watch: stopped at the OpenRouter cap (%s)", e)
            return found
        except Exception as e:  # noqa: BLE001 — unread headlines are retried tomorrow
            logger.warning("Event watch for %s failed (%s)", ticker, e)
            continue
        kept = {_key(ticker, e["title"]): e for e in events}
        repeats = {_key(ticker, a["title"]) for e in events for a in e["also"]}
        with get_session(db) as session:
            for k, i in ((_key(ticker, i["title"]), i) for i in fresh):
                e = kept.get(k)
                session.merge(
                    WatchNews(
                        key=k,
                        ticker=ticker,
                        published=i["published"].isoformat() if i["published"] else "",
                        title=i["title"],
                        source=i["source"],
                        url=i["url"],
                        read_on=today.isoformat(),
                        reader_model=model,
                        event=e is not None or k in repeats,
                        kind=e["kind"] if e else "",
                        what=e["what"] if e else "",
                        why=e["why"] if e else "",
                    )
                )
        logger.info(
            "Event watch %s: %d new headline(s), %d event(s), $%.4f",
            ticker,
            len(fresh),
            len(events),
            cost,
        )
        found += events
    return found


# --- the email -----------------------------------------------------------------


def subject_part(event: dict[str, Any]) -> str:
    return f"{event['ticker']} {event['kind']}"


def event_block(event: dict[str, Any]) -> str:
    def e(x: Any) -> str:
        return html.escape(str(x or ""))

    arrow = {"positive": "▲ ", "negative": "▼ ", "mixed": "◆ "}.get(event["direction"], "")
    rows = [
        f"<b>{e(event['ticker'])} — {arrow}{e(event['kind'])}</b>: {e(event['what'])}",
        f"<i>Why it could matter:</i> {e(event['why'])}" if event["why"] else "",
        f'<a href="{e(event["url"])}">{e(event["title"])}</a> '
        f"<small>({e(event['source'])}, {e(event['published'])})</small>",
    ]
    if event["also"]:
        rows.append(
            "<small>Also reported by: "
            + ", ".join(f'<a href="{e(a["url"])}">{e(a["source"])}</a>' for a in event["also"])
            + "</small>"
        )
    return "<p>" + "<br>".join(r for r in rows if r) + "</p>"


__all__ = ["event_block", "new_watch_events", "read_headlines", "subject_part"]
