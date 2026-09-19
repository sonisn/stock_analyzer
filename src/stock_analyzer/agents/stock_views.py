"""Daily per-stock blocks: facts formatted in code, the long-term view
reused until something changes.

Every weekday the email used to spend an LLM call per holding (plus a news
re-rank call) mostly to copy numbers from JSON into text. The numbers are
now formatted here, so they are fresh every day for free; news ranking is
one batched call for the whole portfolio; and the model is asked for a
stock's 2-3 sentence long-term view only when:

  - there is no stored view for the stock,
  - the stored view is STOCK_VIEW_MAX_AGE_DAYS old (default 7),
  - the price moved STOCK_VIEW_MOVE_PCT or more since it (default 8%), or
  - the company reported earnings since it was written.

Otherwise yesterday's view is shown with its date. For a 3-5 year view
this changes nothing that matters and removes most daily model calls.

The block also drops headlines an earlier email already carried, and when
nothing company-specific is left it says so and shows what the company
actually did instead (discover/stock_facts.py).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from ..db.session import get_session
from ..db.tables import StockView
from ..logging import get_logger

logger = get_logger(__name__)


def _day(value: Any) -> date | None:
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except TypeError, ValueError:
        return None


def last_earnings_event(data: dict[str, Any], today: date) -> date | None:
    """Most recent earnings date on or before today (Yahoo sometimes leaves
    the reported EPS blank for weeks, so the date alone counts)."""
    days = [
        d
        for row in (data.get("earnings") or {}).get("history") or []
        if (d := _day(row.get("Earnings Date"))) is not None and d <= today
    ]
    return max(days) if days else None


def refresh_reason(
    stored: StockView | None,
    *,
    price: float | None,
    reported_on: date | None,
    today: date,
    max_age_days: int,
    move_pct: float,
) -> str | None:
    """Why the view must be rewritten today, or None to reuse it."""
    if stored is None:
        return "no view yet"
    written = date.fromisoformat(stored.written_on)
    if (today - written).days >= max_age_days:
        return f"{(today - written).days} days old"
    if price and stored.price:
        move = (price / stored.price - 1) * 100
        if abs(move) >= move_pct:
            return f"price moved {move:+.1f}%"
    if reported_on is not None and reported_on > written:
        return f"reported earnings {reported_on:%b %d}"
    return None


def load_view(db_path: str | None, ticker: str) -> StockView | None:
    """The stored view, or None to write a fresh one. Holdings are analyzed
    in parallel threads against one SQLite file, so a busy database must
    cost an extra model call at worst, never the stock's block."""
    if not db_path:
        return None
    try:
        with get_session(db_path) as session:
            row = session.get(StockView, ticker)
            if row is not None:
                session.expunge(row)
            return row
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Could not read the stored view for %s (%s) — writing a fresh one", ticker, e
        )
        return None


def save_view(
    db_path: str | None, ticker: str, *, view: str, price: float | None, today: date
) -> None:
    """Store today's view. A failure here only costs tomorrow's reuse — the
    view was already paid for, so it still goes in the email."""
    if not db_path:
        return
    try:
        with get_session(db_path) as session:
            row = session.get(StockView, ticker) or StockView(ticker=ticker, written_on="", view="")
            row.written_on, row.price, row.view = today.isoformat(), price, view
            session.add(row)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not store the view for %s (%s) — it is rewritten tomorrow", ticker, e)


# Enough history that a headline doing the rounds for a week is caught,
# small enough that the row never grows.
_SHOWN_LINKS_KEPT = 40


def shown_links(stored: StockView | None) -> set[str]:
    """Links this stock's block has already carried."""
    if stored is None or not stored.shown_links:
        return set()
    try:
        return set(json.loads(stored.shown_links))
    except json.JSONDecodeError:
        return set()


def record_shown_news(db_path: str | None, ticker: str, links: list[str], *, today: date) -> None:
    """Remember what went out today, so tomorrow doesn't repeat it."""
    if not db_path or not links:
        return
    try:
        with get_session(db_path) as session:
            row = session.get(StockView, ticker)
            if row is None:
                return
            kept = list(dict.fromkeys(links + json.loads(row.shown_links or "[]")))
            row.shown_links = json.dumps(kept[:_SHOWN_LINKS_KEPT])
            row.news_on = today.isoformat()
            session.add(row)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not store shown headlines for %s (%s)", ticker, e)


def _earnings_line(data: dict[str, Any], today: date) -> str | None:
    rows = (data.get("earnings") or {}).get("history") or []
    parts = []
    reported = [
        r
        for r in rows
        if r.get("Reported EPS") is not None and (_day(r.get("Earnings Date")) or today) <= today
    ]
    if reported:
        r = max(reported, key=lambda r: _day(r.get("Earnings Date")) or date.min)
        surprise = r.get("Surprise(%)")
        sur = f" ({float(surprise):+.1f}%)" if surprise is not None else ""
        parts.append(
            f"Last: EPS {r.get('Reported EPS')} vs est {r.get('EPS Estimate')}{sur} "
            f"on {_day(r.get('Earnings Date'))}"
        )
    upcoming = [r for r in rows if (_day(r.get("Earnings Date")) or date.min) > today]
    if upcoming:
        r = min(upcoming, key=lambda r: _day(r.get("Earnings Date")) or date.max)
        est = f" (est EPS {r.get('EPS Estimate')})" if r.get("EPS Estimate") is not None else ""
        parts.append(f"Next: {_day(r.get('Earnings Date'))}{est}")
    return "; ".join(parts) or None


def format_ticker_block(
    data: dict[str, Any],
    *,
    view: str,
    view_note: str = "",
    news: list[dict[str, Any]] | None = None,
    facts: dict[str, str] | None = None,
    today: date | None = None,
) -> str:
    """The same block the email parser reads (reporting/html.py), built
    from the fetched data; missing fields are left out."""
    today = today or date.today()
    lines = ["-" * 40, "", f"{data.get('symbol')} - {data.get('name') or data.get('symbol')}"]
    pos = data.get("position")
    if pos:
        pl = (
            f" | Unrealized: {pos['unrealized_pl']} ({pos.get('pl_pct') or '—'})"
            if pos.get("unrealized_pl")
            else ""
        )
        lines.append(f"Your Position: {pos['units']} shares @ avg {pos['avg_buy_price']}{pl}")
    if data.get("price"):
        today_s = f" ({data['pct_today']} today)" if data.get("pct_today") else ""
        lines.append(f"Price:       {data['price']}{today_s}")
    for label, key in (
        ("Market Cap", "market_cap"),
        # "Range 52W", not "52W Range": the email's label regex
        # (reporting/html.py) only starts a field on a letter, so the old
        # digit-first label was silently glued onto the Price row instead.
        ("Range 52W", "range_52w"),
        ("P/E", "pe"),
        ("Div Yield", "dividend_yield"),
    ):
        if data.get(key):
            lines.append(f"{label + ':':<13}{data[key]}")
    items = news if news is not None else (data.get("news") or [])[:5]
    if items:
        lines.append("Top News:")
        lines.extend(f"- {n['title']} ({n['link']})" for n in items)
    else:
        # Saying so beats padding the section with stories about other
        # companies (data/news_rank.py); `facts` then carries the morning.
        lines.append("Top News:    Nothing company-specific in today's feed.")
    for label, text in (facts or {}).items():
        lines.append(f"{label + ':':<13}{text}")
    analysts = data.get("analysts") or {}
    if analysts or data.get("analyst_target"):
        buy = analysts.get("strongBuy", 0) + analysts.get("buy", 0)
        sell = analysts.get("sell", 0) + analysts.get("strongSell", 0)
        counts = f"Buy {buy} / Hold {analysts.get('hold', 0)} / Sell {sell}" if analysts else ""
        target = f", mean target {data['analyst_target']}" if data.get("analyst_target") else ""
        lines.append(f"Analysts:    {counts}{target}".rstrip())
    lines.append(f"Long-term view: {view}{view_note}")
    earnings = _earnings_line(data, today)
    if earnings:
        lines.append(f"Earnings:    {earnings}")
    trend = "; ".join(
        f"{label}: {data[key]}"
        for label, key in (
            ("7days", "trend_7days"),
            ("1mo", "trend_1mo"),
            ("3mo", "trend_3mo"),
            ("6mo", "trend_6mo"),
            ("1yr", "trend_1yr"),
        )
        if data.get(key)
    )
    if trend:
        lines.append(f"Trend:       {trend}")
    return "\n".join(lines)
