"""Holding alerts between the weekly emails: the rare days worth a look.

Run from the silent after-close snapshot (cli/portfolio.py --snapshot-only).
Two triggers, both from the day's closes, no model calls:

  1. a holding falls more than DROP_SIGMAS times its own usual daily move
     (the standard deviation of its last 60 daily returns). A flat "down
     5%" rule measured volatility, not news: over the 12 months to
     2026-09-27 it would have fired 206 times on 12 holdings (OKLO alone
     52 times — 5% is an ordinary OKLO day), against 7 for this rule;
  2. a written covered call's strike comes within CALL_NEAR_PCT of the
     price — on the day it first gets there, not every day it stays.

Each alert carries what answers "should I worry?": the move beside SPY and
the stock's sector ETF, the latest company headline, the stored thesis
check, and any reported event from the latest SEC filing read. It says
plainly that nothing needs doing unless the thesis changed: these are
long-term holdings, and a sharp drop is more often a hold (or an add)
than a sale.
"""

from __future__ import annotations

import html
import statistics
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..logging import get_logger

logger = get_logger(__name__)

DROP_SIGMAS = 3.0
WINDOW = 60  # daily returns the usual move is measured over
CALL_NEAR_PCT = 5.0


@dataclass
class Drop:
    ticker: str
    day: date
    change_pct: float
    usual_pct: float  # one standard deviation of daily returns, percent
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def sigmas(self) -> float:
        return abs(self.change_pct) / self.usual_pct if self.usual_pct else 0.0


@dataclass
class CallNear:
    ticker: str
    close: float
    strike: float
    expiry: str
    contracts: int

    @property
    def gap_pct(self) -> float:
        return (self.strike / self.close - 1) * 100


def _returns(closes: list[float]) -> list[float]:
    return [b / a - 1 for a, b in zip(closes, closes[1:], strict=False) if a]


def find_drops(
    series: dict[str, list[tuple[date, float]]],
    *,
    today: date,
    sigmas: float = DROP_SIGMAS,
    window: int = WINDOW,
) -> list[Drop]:
    """Holdings whose close TODAY fell more than `sigmas` usual moves. A
    series whose last bar isn't today (a holiday, a stale feed) is skipped,
    so an old drop can't alert twice."""
    out = []
    for ticker, bars in series.items():
        if len(bars) < window + 2 or bars[-1][0] != today:
            continue
        closes = [c for _, c in bars]
        rets = _returns(closes)
        usual = statistics.pstdev(rets[-window - 1 : -1])
        last = rets[-1]
        if usual > 0 and last <= -sigmas * usual:
            out.append(Drop(ticker, today, round(last * 100, 2), round(usual * 100, 2)))
    return sorted(out, key=lambda d: d.change_pct)


def find_calls_near_strike(
    calls: dict[str, dict[str, Any]],
    series: dict[str, list[tuple[date, float]]],
    *,
    today: date,
    within_pct: float = CALL_NEAR_PCT,
) -> list[CallNear]:
    """Covered calls whose strike came within `within_pct` of the price
    today, having been farther away at yesterday's close."""
    out = []
    for ticker, rec in calls.items():
        bars = series.get(ticker) or []
        if len(bars) < 2 or bars[-1][0] != today:
            continue
        before, now = bars[-2][1], bars[-1][1]
        for leg in rec.get("legs") or []:
            line = leg["strike"] / (1 + within_pct / 100)
            if now >= line > before:
                out.append(CallNear(ticker, now, leg["strike"], leg["expiry"], leg["contracts"]))
    return out


# --- the email -----------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.1f}%"


def _drop_block(d: Drop) -> str:
    c = d.context
    rows = [
        f"<b>{html.escape(d.ticker)}</b> fell <b>{d.change_pct:+.1f}%</b> today, "
        f"{d.sigmas:.1f}x its usual daily move ({d.usual_pct:.1f}%).",
        f"Same day: SPY {_pct(c.get('spy_pct'))}"
        + (
            f", {html.escape(c['sector'])} ({c['etf']}) {_pct(c.get('etf_pct'))}"
            if c.get("etf")
            else ""
        )
        + ".",
    ]
    for n in c.get("news") or []:
        link = html.escape(n.get("url") or "")
        title = html.escape(n.get("title") or "")
        rows.append(
            f"News {html.escape(n.get('published_date') or '')}: "
            + (f'<a href="{link}">{title}</a>' if link else title)
        )
    if not c.get("news"):
        rows.append("No company-specific headline in the last two days.")
    thesis = c.get("thesis")
    if thesis:
        signals = "; ".join(s["text"] for s in thesis.get("signals") or [])
        rows.append(
            f"Thesis check (picked {thesis['pick_date']}): <b>{thesis['status']}</b>"
            + (f" — {html.escape(signals)}" if signals else "")
        )
    else:
        rows.append("No stored pick thesis for this holding.")
    for e in c.get("events") or []:
        rows.append(
            f"Latest SEC filing ({html.escape(c.get('filing') or '')}): "
            f"{html.escape(e.get('issue') or '')} [{html.escape(e.get('category') or '')}]"
        )
    return "<p>" + "<br>".join(rows) + "</p>"


def _call_block(c: CallNear) -> str:
    return (
        f"<p><b>{html.escape(c.ticker)}</b> closed at ${c.close:,.2f}, {c.gap_pct:.1f}% below "
        f"the ${c.strike:,.2f} strike of {c.contracts} written call(s) expiring {c.expiry}. "
        "If it finishes above the strike the shares can be called away: roll the call "
        "up and out, buy it back, or let them go.</p>"
    )


def build_alert(drops: list[Drop], calls: list[CallNear]) -> tuple[str, str] | None:
    """(subject, html body), or None when there is nothing to say."""
    if not drops and not calls:
        return None
    parts = [f"{d.ticker} {d.change_pct:+.0f}%" for d in drops]
    parts += [f"{c.ticker} near call strike" for c in calls]
    subject = "Holding alert: " + ", ".join(parts)
    body = ["<html><body style='font-family:sans-serif;max-width:640px'>"]
    if drops:
        body.append("<h3>Unusual drops</h3>")
        body += [_drop_block(d) for d in drops]
        body.append(
            "<p><i>No action needed unless the thesis changed. These are long-term "
            "holdings: a sharp drop with the thesis intact is usually a hold, or a "
            "chance to add — the Wednesday email has the full view.</i></p>"
        )
    if calls:
        body.append("<h3>Covered calls near their strike</h3>")
        body += [_call_block(c) for c in calls]
    body.append("</body></html>")
    return subject, "\n".join(body)


# --- context (network, guarded) ----------------------------------------------

_SUFFIXES = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "ltd",
    "plc",
    "holdings",
    "group",
    "company",
    "the",
    "class",
    "a",
    "technologies",
    "technology",
}


def about_company(
    items: list[dict[str, Any]], ticker: str, name: str | None
) -> list[dict[str, Any]]:
    """The items that name the company in their headline. Most of a
    company-news feed is syndicated market commentary that merely tags the
    ticker ("Why the Nasdaq refuses to break even…" under GOOGL)."""
    words = [w for w in (name or "").replace(",", " ").split() if w.lower() not in _SUFFIXES]
    keys = {ticker.lower(), *(w.lower() for w in words[:1])}
    return [i for i in items if any(k and k in f" {(i.get('title') or '').lower()} " for k in keys)]


def add_context(
    drops: list[Drop],
    series: dict[str, list[tuple[date, float]]],
    *,
    db: str,
    sectors: dict[str, dict[str, Any]],
    etf_for: dict[str, str],
    thesis: dict[str, dict[str, Any]],
) -> None:
    """Fill each drop's context. Every source is optional: a missing one
    leaves its line out, never the alert."""
    from ..data.filing_evidence import evidence_packs
    from ..data.ticker_news import fetch_finnhub_ticker_news

    def day_change(sym: str, day: date) -> float | None:
        bars = series.get(sym) or []
        for i in range(1, len(bars)):
            if bars[i][0] == day:
                return round((bars[i][1] / bars[i - 1][1] - 1) * 100, 2)
        return None

    try:
        packs = evidence_packs(db, [d.ticker for d in drops])
    except Exception:  # noqa: BLE001
        packs = {}
    for d in drops:
        etf = etf_for.get(d.ticker)
        d.context = {
            "spy_pct": day_change("SPY", d.day),
            "etf": etf,
            "etf_pct": day_change(etf, d.day) if etf else None,
            "sector": (sectors.get(d.ticker) or {}).get("sector") or "",
            "thesis": thesis.get(d.ticker),
        }
        try:
            items = fetch_finnhub_ticker_news(d.ticker, days=2, max_results=20)
            name = (sectors.get(d.ticker) or {}).get("name")
            d.context["news"] = about_company(items, d.ticker, name)[:2]
        except Exception as e:  # noqa: BLE001
            logger.info("No news for the %s alert (%s)", d.ticker, e)
        pack = packs.get(d.ticker) or {}
        if pack.get("events"):
            d.context["events"] = pack["events"]
            d.context["filing"] = f"{pack.get('form')} for {pack.get('period_end')}"
