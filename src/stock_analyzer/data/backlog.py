"""The book: revenue already contracted but not yet delivered.

Everything forward-looking in the pipeline is somebody's opinion —
analyst targets, forward P/E, EPS revisions. Remaining performance
obligations are not an opinion: they are signed orders the company has
told the SEC it has not delivered yet, tagged in every 10-Q and 10-K as
`RevenueRemainingPerformanceObligation`.

It matters because price and book can say opposite things. On
2026-09-20 the daily email called AVGO's thesis BROKEN — below its
200-day average, lagging SPY by 17 points, analysts cutting EPS — while
its contracted book had gone $45.0B → $164.6B → $179.2B over two
quarters. POWL was flagged at -26.4% and offered as a tax-loss sale with
its book up 50% ($1.6B → $2.4B). Both may still be sells; neither should
be sold without that fact in view.

Free, from the SEC's XBRL API through the client `sec_edgar` already
uses. Every fact carries the date it was FILED, so this is point-in-time
by construction — usable in a backtest without look-ahead, which
yfinance's forward estimates are not.

Not every company tags it: ANET's last fact is from 2022, and banks and
retailers have none at all. So it is evidence where present and never a
filter that penalizes absence.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from ..logging import get_logger
from . import yf_gateway
from .sec_edgar import _HTTP, load_ticker_cik_map

logger = get_logger(__name__)

_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/"
    "RevenueRemainingPerformanceObligation.json"
)

# A fact this old is history, not a current book: companies that stopped
# tagging the concept would otherwise look like they still report it.
MAX_FACT_AGE_DAYS = 200
# How far from exactly a year back a fact may sit and still count as the
# year-ago comparison (fiscal quarters drift against the calendar).
_YEAR_AGO_WINDOW = (300, 430)


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except TypeError, ValueError:
        return None


def fetch_rpo(ticker: str, *, as_of: date | None = None) -> dict[str, Any] | None:
    """Contracted-but-undelivered revenue for `ticker`, or None.

    `as_of` keeps only facts already FILED by that date, which is what
    makes this safe to use in a backtest: the book AVGO disclosed on
    2026-09-10 did not exist for anyone on 2026-08-01.
    """
    cik = load_ticker_cik_map().get(ticker.upper())
    if not cik:
        return None
    try:
        body = _HTTP.get_json(_CONCEPT_URL.format(cik=cik))
    except Exception as e:  # noqa: BLE001 — a missing concept is normal
        logger.debug("No RPO concept for %s (%s)", ticker, e)
        return None

    facts = (body.get("units") or {}).get("USD") or []
    # One value per period end, preferring the most recently filed —
    # an amended filing restates the same quarter.
    by_end: dict[date, dict[str, Any]] = {}
    for fact in facts:
        end, filed = _day(fact.get("end")), _day(fact.get("filed"))
        value = fact.get("val")
        if end is None or filed is None or value is None:
            continue
        if as_of and filed > as_of:
            continue
        seen = by_end.get(end)
        if seen is None or filed >= seen["filed"]:
            by_end[end] = {
                "period_end": end,
                "filed": filed,
                "value": float(value),
                "form": fact.get("form"),
            }
    if not by_end:
        return None

    history = [by_end[end] for end in sorted(by_end)]
    latest = history[-1]
    horizon = as_of or date.today()
    if (horizon - latest["period_end"]).days > MAX_FACT_AGE_DAYS:
        logger.debug(
            "%s last tagged an RPO for %s — treating it as no longer reported",
            ticker,
            latest["period_end"],
        )
        return None

    prior = history[-2] if len(history) > 1 else None
    year_ago = None
    for row in reversed(history[:-1]):
        gap = (latest["period_end"] - row["period_end"]).days
        if _YEAR_AGO_WINDOW[0] <= gap <= _YEAR_AGO_WINDOW[1]:
            year_ago = row
            break

    def change(previous: dict[str, Any] | None) -> float | None:
        if not previous or not previous["value"]:
            return None
        return (latest["value"] / previous["value"] - 1) * 100

    return {
        "ticker": ticker.upper(),
        "value": latest["value"],
        "period_end": latest["period_end"].isoformat(),
        "filed": latest["filed"].isoformat(),
        "form": latest["form"],
        "qoq_pct": change(prior),
        "yoy_pct": change(year_ago),
        "history": [
            {**row, "period_end": row["period_end"].isoformat(), "filed": row["filed"].isoformat()}
            for row in history[-8:]
        ],
    }


def batch_rpo(tickers: list[str], *, as_of: date | None = None) -> dict[str, dict[str, Any]]:
    """`fetch_rpo` across tickers, skipping the ones that do not tag it."""
    out: dict[str, dict[str, Any]] = {}
    for ticker, result in yf_gateway.map_symbols(
        lambda t: fetch_rpo(t, as_of=as_of), tickers, workers=3
    ):
        if result:
            out[ticker.upper()] = result
    return out


def coverage_years(rpo_value: float | None, revenue_ttm: float | None) -> float | None:
    """The book measured in years of current revenue.

    A number above 1 means more is already contracted than the company
    sold in the last twelve months.
    """
    if not rpo_value or not revenue_ttm or revenue_ttm <= 0:
        return None
    return rpo_value / revenue_ttm


def _money(value: float) -> str:
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    return f"${value / 1e6:,.0f}M"


def backlog_note(rec: dict[str, Any] | None, *, revenue_ttm: float | None = None) -> str:
    """One line describing the book, or "".

    Deliberately states the growth and the as-of date together: a book is
    a quarterly disclosure, so "up 50%" without "as of 2026-06-30, filed
    2026-08-04" invites reading it as today's news.
    """
    if not rec or not rec.get("value"):
        return ""
    parts = [f"contracted book {_money(rec['value'])} as of {rec['period_end']}"]
    if rec.get("yoy_pct") is not None:
        parts.append(f"{rec['yoy_pct']:+.0f}% over a year")
    elif rec.get("qoq_pct") is not None:
        parts.append(f"{rec['qoq_pct']:+.0f}% over the quarter")
    coverage = coverage_years(rec.get("value"), revenue_ttm)
    if coverage:
        parts.append(f"{coverage:.1f}x trailing revenue")
    return ", ".join(parts) + f" (filed {rec['filed']})"


__all__ = [
    "MAX_FACT_AGE_DAYS",
    "backlog_note",
    "batch_rpo",
    "coverage_years",
    "fetch_rpo",
]


def backlog_block(books: dict[str, dict[str, Any]]) -> str:
    """The order book behind each holding, for the rebalancer's prompt.

    A sale decision reads price, trend and estimates — all of which are
    opinions about the future. Remaining performance obligations are the
    one forward-looking number that is already signed, and they can point
    the other way: AVGO's thesis was called broken while its book grew
    552%. Absence means the company does not tag the concept, never that
    the book is empty, so a name without a row must not be penalised.
    """
    rows = []
    for ticker in sorted(books):
        rec = books[ticker] or {}
        value = rec.get("value")
        if not value:
            continue
        yoy, qoq = rec.get("yoy_pct"), rec.get("qoq_pct")
        parts = [f"  {ticker}: {_money(float(value))} contracted"]
        if yoy is not None:
            parts.append(f"{yoy:+.0f}% YoY")
        if qoq is not None:
            parts.append(f"{qoq:+.0f}% QoQ")
        if rec.get("period_end"):
            parts.append(f"as of {rec['period_end']}")
        rows.append(", ".join(parts))
    if not rows:
        return ""
    return (
        "CONTRACTED BOOK (SEC-filed remaining performance obligations)\n"
        "Revenue already under contract and not yet delivered — signed orders,\n"
        "not an analyst forecast. A book growing fast argues against trimming\n"
        "the name; a shrinking one supports a sale. Holdings with no row do not\n"
        "tag the concept and must NOT be treated as having no backlog.\n"
        + "\n".join(rows)
    )

