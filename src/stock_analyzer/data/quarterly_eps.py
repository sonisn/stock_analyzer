"""Quarterly and annual diluted EPS from SEC filings, and their growth.

The EPS Rating (discover/ibd_ratings) ranks the latest two quarters' EPS
growth over the same quarter a year earlier and the three-year annual
rate. Yahoo keeps four quarters and would cost a request per name on top
of the bars; the SEC's XBRL API has every 10-Q and 10-K figure since
2009, free, one request per company (`EarningsPerShareDiluted`).

A 10-K reports the year, not the fourth quarter, so Q4 is the year less
the three quarters inside it — close, not exact, since the share count
moves during the year. Foreign filers (20-F, IFRS) tag no US-GAAP EPS and
get no rating, like banks without a book in data/backlog.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date
from typing import Any

from ..logging import get_logger
from . import fetch_cache, yf_gateway
from .sec_edgar import _HTTP, load_ticker_cik_map

logger = get_logger(__name__)

_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/"
    "EarningsPerShareDiluted.json"
)
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (350, 380)
_YEAR_AGO = (300, 430)
# A company whose last quarter is older than this has stopped filing.
MAX_AGE_DAYS = 200
# Growth off a base this close to zero is noise, not growth.
MIN_BASE = 0.01
_UNTAGGED: dict[str, Any] = {"untagged": True}


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except TypeError, ValueError:
        return None


def periods(
    facts: list[dict[str, Any]],
) -> tuple[list[tuple[date, float]], list[tuple[date, float]]]:
    """(quarters, years) as (period end, EPS), oldest first; the latest
    filing wins for a restated period, and Q4s are derived from years."""
    latest: dict[tuple[date, date], tuple[date, float]] = {}
    for f in facts:
        start, end, filed = _day(f.get("start")), _day(f.get("end")), _day(f.get("filed"))
        if start is None or end is None or filed is None or f.get("val") is None:
            continue
        seen = latest.get((start, end))
        if seen is None or filed >= seen[0]:
            latest[(start, end)] = (filed, float(f["val"]))
    quarters: dict[date, float] = {}
    years: dict[date, tuple[date, float]] = {}
    for (start, end), (_, val) in latest.items():
        days = (end - start).days
        if _QUARTER_DAYS[0] <= days <= _QUARTER_DAYS[1]:
            quarters[end] = val
        elif _YEAR_DAYS[0] <= days <= _YEAR_DAYS[1]:
            years[end] = (start, val)
    for end, (start, val) in years.items():
        if end in quarters:
            continue
        inside = [q for q in quarters if start < q < end]
        if len(inside) == 3:
            quarters[end] = val - sum(quarters[q] for q in inside)
    return sorted(quarters.items()), sorted((e, v) for e, (_, v) in years.items())


def _growth(now: float, then: float) -> float | None:
    return None if abs(then) < MIN_BASE else (now - then) / abs(then)


def _year_before(series: list[tuple[date, float]], i: int) -> float | None:
    end = series[i][0]
    for other_end, val in reversed(series[:i]):
        if _YEAR_AGO[0] <= (end - other_end).days <= _YEAR_AGO[1]:
            return val
    return None


def eps_growth(
    quarters: list[tuple[date, float]], years: list[tuple[date, float]], *, today: date
) -> dict[str, Any] | None:
    """{"q1_growth", "q2_growth", "cagr_3y", "latest_quarter", "eps"}, or
    None when the latest quarter is missing or stale."""
    if not quarters or (today - quarters[-1][0]).days > MAX_AGE_DAYS:
        return None
    out: dict[str, Any] = {
        "latest_quarter": quarters[-1][0].isoformat(),
        "eps": quarters[-1][1],
        "q1_growth": None,
        "q2_growth": None,
        "cagr_3y": None,
    }
    for key, i in (("q1_growth", len(quarters) - 1), ("q2_growth", len(quarters) - 2)):
        if i >= 0:
            then = _year_before(quarters, i)
            out[key] = None if then is None else _growth(quarters[i][1], then)
    if years:
        last_end, last = years[-1]
        for end, val in years:
            if 3 * 365 - 60 <= (last_end - end).days <= 3 * 365 + 60 and val > 0 and last > 0:
                out["cagr_3y"] = (last / val) ** (1 / 3) - 1
    return out


def fetch_facts(ticker: str, *, strict: bool = False) -> list[dict[str, Any]] | None:
    """Every diluted-EPS fact the company has filed, or None. `strict`
    raises on a failed request so a cache can tell "no EPS" from "could
    not ask"; a 404 (never tagged) is None either way."""
    cik = load_ticker_cik_map().get(ticker.upper())
    if not cik:
        return None
    try:
        body = _HTTP.get_json(_CONCEPT_URL.format(cik=cik))
    except Exception as e:  # noqa: BLE001 — a missing concept is normal
        if strict and getattr(e, "status", None) != 404:
            raise
        return None
    return (body.get("units") or {}).get("USD/shares") or None


def growth_as_of(facts: list[dict[str, Any]], as_of: date) -> dict[str, Any] | None:
    """EPS growth as it stood on `as_of`: only facts filed by then count,
    which is what makes a rebuilt history free of look-ahead."""
    known = [f for f in facts if (_day(f.get("filed")) or date.max) <= as_of]
    quarters, years = periods(known)
    return eps_growth(quarters, years, today=as_of)


def fetch_eps(ticker: str, *, strict: bool = False, today: date | None = None):
    """EPS growth for `ticker` (see `eps_growth`), or None. `strict` raises
    on a failed request so a cache can tell "no EPS" from "could not ask"."""
    facts = fetch_facts(ticker, strict=strict)
    if not facts:
        return None
    quarters, years = periods(facts)
    return eps_growth(quarters, years, today=today or date.today())


def batch_eps(tickers: Iterable[str], *, refresh: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    """`fetch_eps` across tickers through the week-long fetch cache."""
    wanted = [t.upper() for t in tickers]

    def ask(ticker: str) -> dict[str, Any] | None:
        try:
            return fetch_eps(ticker, strict=True) or _UNTAGGED
        except Exception as e:  # noqa: BLE001 — retried next run, not cached
            logger.debug("EPS fetch failed for %s (%s)", ticker, e)
            return None

    def fetch(todo: list[str]) -> Iterator[tuple[str, Any]]:
        yield from yf_gateway.map_symbols(ask, todo, workers=3)

    answers = fetch_cache.fetch_many(
        "quarterly_eps", wanted, fetch, refresh=[t.upper() for t in refresh]
    )
    return {t: r for t, r in answers.items() if not r.get("untagged")}
