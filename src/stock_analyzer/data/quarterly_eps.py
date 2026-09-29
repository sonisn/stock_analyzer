"""Quarterly and annual diluted EPS from SEC filings, and their growth.

The EPS Rating (discover/ibd_ratings) ranks the latest two quarters' EPS
growth over the same quarter a year earlier and the three-year annual
rate. Yahoo keeps four quarters and would cost a request per name on top
of the bars; the SEC's XBRL API has every 10-Q and 10-K figure since
2009, free, one request per company (`EarningsPerShareDiluted`).

A 10-K reports the year, not the fourth quarter, so Q4 is the year less
the three quarters inside it — close, not exact, since the share count
moves during the year.

Two fallbacks, since about a quarter of the universe came back empty:

  - the company-concept API sometimes serves an empty diluted-EPS list
    for a company whose full facts have it, and some companies file
    under another US-GAAP tag; both are read from the company-facts file;
  - foreign filers (20-F, IFRS: annual XBRL only), per-class tags (Visa)
    and partnerships get Yahoo's reported quarterly EPS instead: one
    request per name, adjusted rather than GAAP, marked "source": "yahoo".
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, timedelta
from typing import Any

from ..logging import get_logger
from . import fetch_cache, yf_gateway
from .sec_edgar import _HTTP, load_ticker_cik_map

logger = get_logger(__name__)

_CONCEPT_URL = (
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/us-gaap/"
    "EarningsPerShareDiluted.json"
)
_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
# Where a company files per-share earnings when the diluted-EPS concept
# comes back empty, best first; the tag with the latest period wins.
FALLBACK_TAGS = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
    "EarningsPerShareBasic",
)
# Yahoo's earnings calendar: about six years of reports, enough for the
# three-year rate.
YAHOO_REPORTS = 28
_QUARTER_DAYS = (80, 100)
_YEAR_DAYS = (350, 380)
_YEAR_AGO = (300, 430)
# A company whose last quarter is older than this has stopped filing.
MAX_AGE_DAYS = 200
# Growth off a base this close to zero is noise, not growth.
MIN_BASE = 0.01
# "checked": both fallbacks were tried too.
_UNTAGGED: dict[str, Any] = {"untagged": True, "checked": True}
_FAILED = object()


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


def _usd_per_share(tags: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return ((tags.get(name) or {}).get("units") or {}).get("USD/shares") or []


def facts_from_company(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The per-share earnings facts from a company-facts file: the
    FALLBACK_TAGS entry with the latest period (the earlier-listed on a tie)."""
    tags = (body.get("facts") or {}).get("us-gaap") or {}
    best: tuple[str, int] | None = None
    pick: list[dict[str, Any]] | None = None
    for rank, name in enumerate(FALLBACK_TAGS):
        facts = _usd_per_share(tags, name)
        if not facts:
            continue
        key = (max(str(f.get("end") or "") for f in facts), -rank)
        if best is None or key > best:
            best, pick = key, facts
    return pick


def fetch_facts(ticker: str, *, strict: bool = False) -> list[dict[str, Any]] | None:
    """Every diluted-EPS fact the company has filed, or None. `strict`
    raises on a failed request so a cache can tell "no EPS" from "could
    not ask"; a 404 (never tagged) is None either way. An empty, missing or
    stale concept (GD, MNST moved to another tag) falls back to the
    company-facts file (see facts_from_company)."""
    cik = load_ticker_cik_map().get(ticker.upper())
    if not cik:
        return None
    try:
        body = _HTTP.get_json(_CONCEPT_URL.format(cik=cik))
    except Exception as e:  # noqa: BLE001 — a missing concept is normal
        if strict and getattr(e, "status", None) != 404:
            raise
        body = {}
    facts = (body.get("units") or {}).get("USD/shares") or None
    if facts and _latest_end(facts) >= date.today() - timedelta(days=MAX_AGE_DAYS):
        return facts
    try:
        company = _HTTP.get_json(_FACTS_URL.format(cik=cik))
    except Exception as e:  # noqa: BLE001 — as above
        if strict and getattr(e, "status", None) != 404:
            raise
        return facts
    other = facts_from_company(company or {})
    if other and (not facts or _latest_end(other) > _latest_end(facts)):
        return other
    return facts


def _latest_end(facts: list[dict[str, Any]]) -> date:
    return max((_day(f.get("end")) or date.min for f in facts), default=date.min)


def growth_as_of(facts: list[dict[str, Any]], as_of: date) -> dict[str, Any] | None:
    """EPS growth as it stood on `as_of`: only facts filed by then count,
    which is what makes a rebuilt history free of look-ahead."""
    known = [f for f in facts if (_day(f.get("filed")) or date.max) <= as_of]
    quarters, years = periods(known)
    return eps_growth(quarters, years, today=as_of)


def fetch_eps(ticker: str, *, strict: bool = False, today: date | None = None):
    """EPS growth for `ticker` (see `eps_growth`) from SEC filings, or None.
    `strict` raises on a failed request so a cache can tell "no EPS" from
    "could not ask"."""
    facts = fetch_facts(ticker, strict=strict)
    if not facts:
        return None
    quarters, years = periods(facts)
    got = eps_growth(quarters, years, today=today or date.today())
    return {**got, "source": "sec"} if got else None


def quarter_end_before(day: date) -> date:
    """The last calendar quarter end before a report on `day`."""
    for back in range(1, 120):
        d = day - timedelta(days=back)
        if (d.month, d.day) in ((3, 31), (6, 30), (9, 30), (12, 31)):
            return d
    return day


def yahoo_periods(
    reports: list[tuple[date, float]],
) -> tuple[list[tuple[date, float]], list[tuple[date, float]]]:
    """(quarters, years) from (report day, reported EPS): each report is
    dated to the quarter end before it, and a year is four consecutive
    quarters summed (none for half-yearly reporters)."""
    by_end: dict[date, float] = {}
    for day, eps in sorted(reports):
        by_end[quarter_end_before(day)] = eps
    quarters = sorted(by_end.items())
    years = []
    for i in range(3, len(quarters)):
        four = quarters[i - 3 : i + 1]
        gaps = [(b[0] - a[0]).days for a, b in zip(four, four[1:], strict=False)]
        if all(_QUARTER_DAYS[0] <= g <= _QUARTER_DAYS[1] for g in gaps):
            years.append((quarters[i][0], sum(v for _, v in four)))
    return quarters, years


def yahoo_reports(ticker: str) -> list[tuple[date, float]] | None:
    """(report day, reported EPS) from Yahoo's earnings calendar; [] when
    Yahoo has none, None when it could not be asked."""
    df: Any = yf_gateway.ticker_call(
        ticker,
        "earnings_dates",
        lambda t: t.get_earnings_dates(limit=YAHOO_REPORTS),
        default=_FAILED,
    )
    if df is _FAILED:
        return None
    if df is None or getattr(df, "empty", True) or "Reported EPS" not in df:
        return []
    return [
        (date.fromisoformat(str(when)[:10]), float(eps))
        for when, eps in zip(df.index, df["Reported EPS"], strict=True)
        if eps is not None and eps == eps  # NaN: not reported yet
    ]


class YahooUnavailable(Exception):
    """Yahoo's earnings calendar could not be read (not "no EPS")."""


def fetch_yahoo_eps(ticker: str, *, today: date | None = None) -> dict[str, Any] | None:
    """EPS growth from Yahoo's reported EPS, or None. Raises
    YahooUnavailable when Yahoo could not be asked."""
    reports = yahoo_reports(ticker)
    if reports is None:
        raise YahooUnavailable(ticker)
    quarters, years = yahoo_periods(reports)
    got = eps_growth(quarters, years, today=today or date.today())
    return {**got, "source": "yahoo"} if got else None


def batch_eps(tickers: Iterable[str], *, refresh: Iterable[str] = ()) -> dict[str, dict[str, Any]]:
    """EPS growth across tickers through the week-long fetch cache: SEC
    filings first, Yahoo's reported EPS for the rest."""
    wanted = [t.upper() for t in tickers]

    def ask(ticker: str) -> dict[str, Any] | None:
        try:
            return fetch_eps(ticker, strict=True) or fetch_yahoo_eps(ticker) or _UNTAGGED
        except Exception as e:  # noqa: BLE001 — retried next run, not cached
            logger.debug("EPS fetch failed for %s (%s)", ticker, e)
            return None

    def fetch(todo: list[str]) -> Iterator[tuple[str, Any]]:
        yield from yf_gateway.map_symbols(ask, todo, workers=3)

    # Answers cached as "none" before the fallbacks existed are asked again.
    unchecked = [
        t
        for t, e in fetch_cache.entries("quarterly_eps").items()
        if (e.get("value") or {}).get("untagged") and not (e.get("value") or {}).get("checked")
    ]
    answers = fetch_cache.fetch_many(
        "quarterly_eps", wanted, fetch, refresh=[*(t.upper() for t in refresh), *unchecked]
    )
    return {t: r for t, r in answers.items() if not r.get("untagged")}
