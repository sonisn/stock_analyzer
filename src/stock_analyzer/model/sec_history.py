"""Fundamentals as they were known on each month-end, from SEC filings.

The screen's fundamentals come from Yahoo, which keeps no history, so the
live record is the only test of them — 26 graded names by 2026-10. The
SEC's company-facts file has every 10-Q and 10-K figure each company has
tagged since 2009, and every figure carries the date it was FILED. Reading
only what was filed by a date gives what an investor could have known then:
no look-ahead, and fifteen years of month-ends to test against.

What this module builds, per (month-end, ticker):

  - flows over the trailing twelve months (revenue, operating income, gross
    profit, net income, operating cash flow, capex). A 10-Q reports the
    year to date, so TTM = this year-to-date + the last fiscal year - the
    same year-to-date a year earlier; a 10-K's year is used as is;
  - balance-sheet items at the latest filed date (assets, equity, debt);
  - shares outstanding from the filing cover page (summed across share
    classes), times the actual traded close of the day (`raw_close`: the
    bar store's prices are split- and dividend-adjusted, a filing's share
    count is not) for market cap;
  - a standardized earnings surprise (SUE): the latest quarter's diluted
    EPS less the same quarter a year earlier, over the spread of the last
    eight such changes.

A figure stops counting once it is more than STALE_DAYS old (the company
stopped filing, or changed the tag). Company-facts files are cached
gzipped on disk for FACTS_MAX_AGE_DAYS; one request per company at the
SEC's published rate limit.
"""

from __future__ import annotations

import gzip
import json
import math
import time
from bisect import insort
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from statistics import pstdev
from typing import Any

import polars as pl

from ..data.frames import DATE
from ..logging import get_logger

logger = get_logger(__name__)

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
FACTS_MAX_AGE_DAYS = 30
STALE_DAYS = 200

# Each concept's tags, best first. Companies move between tags (most
# revenue moved to RevenueFromContract... with ASC 606 in 2018), so all are
# read and a period takes the first tag that reports it.
REVENUE = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueGoodsNet",
)
FLOWS: dict[str, tuple[str, ...]] = {
    "revenue": REVENUE,
    "operating_income": ("OperatingIncomeLoss",),
    "gross_profit": ("GrossProfit",),
    "cost_of_revenue": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "cfo": (
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
}
INSTANTS: dict[str, tuple[str, ...]] = {
    "assets": ("Assets",),
    "equity": (
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "long_term_debt": ("LongTermDebt", "LongTermDebtNoncurrent"),
    "current_debt": ("LongTermDebtCurrent", "DebtCurrent"),
    "short_term_borrowings": ("ShortTermBorrowings", "CommercialPaper"),
}
EPS_TAGS = ("EarningsPerShareDiluted", "EarningsPerShareBasicAndDiluted")

_QUARTER = (80, 100)
_YEAR = (350, 380)
_YEAR_AGO = (350, 380)  # end-to-end distance of "the same period a year earlier"
SUE_CHANGES = 8  # quarters of year-on-year EPS changes behind one surprise
SUE_MIN_CHANGES = 6
MIN_EPS_SPREAD = 1e-4  # dollars per share


def _day(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except TypeError, ValueError:
        return None


# --- the company-facts file ---------------------------------------------------


def _cache_path(cache_dir: str, cik: int) -> Path:
    return Path(cache_dir).expanduser() / "sec_facts" / f"CIK{cik:010d}.json.gz"


def company_facts(cik: int, cache_dir: str, *, refresh: bool = False) -> dict[str, Any] | None:
    """The SEC company-facts JSON for `cik`, from the disk cache when fresh
    enough; None when the SEC has none (a 404) or the request fails."""
    from ..data.sec_edgar import _HTTP
    from ..http_client import ClientError, HttpClientError

    path = _cache_path(cache_dir, cik)
    if path.exists() and not refresh:
        age_days = (time.time() - path.stat().st_mtime) / 86400
        if age_days <= FACTS_MAX_AGE_DAYS:
            try:
                with gzip.open(path, "rt") as f:
                    return json.load(f)
            except (OSError, ValueError) as e:
                logger.warning("SEC facts cache unreadable (%s): %s — fetching again", path, e)
    try:
        body = _HTTP.get_json(FACTS_URL.format(cik=cik))
    except ClientError:
        return None  # no XBRL facts for this company
    except HttpClientError as e:
        logger.warning("SEC company facts for CIK %d failed: %s", cik, e)
        return None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(body, f)
        tmp.replace(path)
    except OSError as e:
        logger.warning("Could not cache SEC facts for CIK %d (%s)", cik, e)
    return body


# --- facts as they arrive -----------------------------------------------------


@dataclass
class Fact:
    filed: date
    start: date | None
    end: date
    val: float
    accn: str = field(compare=False, default="")


def _facts(body: dict[str, Any], taxonomy: str, tags: Iterable[str], unit: str) -> list[Fact]:
    """Every fact under `tags` in `unit`; for a period two tags both report,
    the earlier tag in `tags` wins."""
    space = (body.get("facts") or {}).get(taxonomy) or {}
    seen: set[tuple[date | None, date, date]] = set()
    out: list[Fact] = []
    for tag in tags:
        for raw in ((space.get(tag) or {}).get("units") or {}).get(unit) or []:
            end, filed = _day(raw.get("end")), _day(raw.get("filed"))
            if end is None or filed is None or raw.get("val") is None:
                continue
            start = _day(raw.get("start"))
            key = (start, end, filed)
            if key in seen:
                continue
            seen.add(key)
            out.append(Fact(filed, start, end, float(raw["val"]), str(raw.get("accn") or "")))
    out.sort(key=lambda f: (f.filed, f.end, f.start or date.min))
    return out


class Known:
    """What had been filed by a date: per period, the latest value filed so
    far (a restatement replaces the original only once it is filed).
    `advance(day)` takes in every fact filed on or before `day`."""

    def __init__(self, facts: list[Fact]):
        self._facts = facts
        self._i = 0
        self.periods: dict[tuple[date | None, date], float] = {}
        self.ends: list[date] = []  # sorted, distinct period ends known so far

    def advance(self, day: date) -> bool:
        """True when something new arrived."""
        changed = False
        while self._i < len(self._facts) and self._facts[self._i].filed <= day:
            f = self._facts[self._i]
            if (f.start, f.end) not in self.periods and f.end not in self.ends:
                insort(self.ends, f.end)
            self.periods[(f.start, f.end)] = f.val
            self._i += 1
            changed = True
        return changed


def _length(start: date | None, end: date) -> int | None:
    return None if start is None else (end - start).days + 1


def _year_to_date(periods: dict[tuple[date | None, date], float], end: date):
    """(start, value) of the longest period under a year ending at `end`."""
    best = None
    for (start, e), val in periods.items():
        n = _length(start, e)
        if e == end and n is not None and n < _YEAR[0] and (best is None or n > best[2]):
            best = (start, val, n)
    return best


def ttm(periods: dict[tuple[date | None, date], float], end: date) -> float | None:
    """Trailing twelve months of a flow ending at `end`, or None."""
    for (start, e), val in periods.items():
        n = _length(start, e)
        if e == end and n is not None and _YEAR[0] <= n <= _YEAR[1]:
            return val
    ytd = _year_to_date(periods, end)
    if ytd is None:
        return None
    start, ytd_val, n = ytd
    fiscal_year = prior = None
    for (s, e), val in periods.items():
        length = _length(s, e)
        if length is None:
            continue
        if _YEAR[0] <= length <= _YEAR[1] and 0 <= (start - e).days <= 10:
            fiscal_year = val
        elif (
            abs(length - n) <= 15
            and _YEAR_AGO[0] <= (end - e).days <= _YEAR_AGO[1]
            and s is not None
            and 0 <= (start - s).days - 355 <= 20
        ):
            prior = val
    if fiscal_year is None or prior is None:
        return None
    return ytd_val + fiscal_year - prior


def latest_ttm(known: Known, day: date) -> tuple[date, float, float | None] | None:
    """(end, TTM, TTM a year earlier) at the latest period end that yields a
    TTM, skipping a stale one."""
    for end in reversed(known.ends):
        if (day - end).days > STALE_DAYS:
            return None
        now = ttm(known.periods, end)
        if now is None:
            continue
        before = None
        for e in reversed(known.ends):
            if _YEAR_AGO[0] <= (end - e).days <= _YEAR_AGO[1]:
                before = ttm(known.periods, e)
                if before is not None:
                    break
        return end, now, before
    return None


def latest_instant(known: Known, day: date) -> float | None:
    if not known.ends or (day - known.ends[-1]).days > STALE_DAYS:
        return None
    end = known.ends[-1]
    vals = [v for (s, e), v in known.periods.items() if e == end and s is None]
    return vals[-1] if vals else None


def quarterly_eps(periods: dict[tuple[date | None, date], float]) -> list[tuple[date, float]]:
    """(quarter end, EPS), oldest first; Q4 is the year less its three quarters."""
    quarters: dict[date, float] = {}
    years: dict[date, tuple[date, float]] = {}
    for (start, end), val in periods.items():
        n = _length(start, end)
        if n is None:
            continue
        if _QUARTER[0] <= n <= _QUARTER[1]:
            quarters[end] = val
        elif _YEAR[0] <= n <= _YEAR[1] and start is not None:
            years[end] = (start, val)
    for end, (start, val) in years.items():
        if end not in quarters:
            inside = [q for q in quarters if start < q < end]
            if len(inside) == 3:
                quarters[end] = val - sum(quarters[q] for q in inside)
    return sorted(quarters.items())


def sue(quarters: list[tuple[date, float]], day: date) -> float | None:
    """Latest quarter's year-on-year EPS change over the spread of the last
    SUE_CHANGES such changes (the seasonal random-walk surprise)."""
    if not quarters or (day - quarters[-1][0]).days > STALE_DAYS:
        return None
    by_end = dict(quarters)
    changes: list[float] = []
    for end, val in reversed(quarters):
        before = next(
            (by_end[e] for e in by_end if _YEAR_AGO[0] - 15 <= (end - e).days <= _YEAR_AGO[1]),
            None,
        )
        if before is None:
            if not changes:
                return None  # the latest quarter has nothing to compare with
            break
        changes.append(val - before)
        if len(changes) > SUE_CHANGES:
            break
    if len(changes) < SUE_MIN_CHANGES + 1:
        return None
    spread = pstdev(changes[1:])
    # Rounding leaves a flat history a spread of ~1e-16, not zero.
    return None if spread < MIN_EPS_SPREAD else changes[0] / spread


class Shares:
    """Cover-page shares outstanding, summed across share classes within a
    filing; the latest filing's total as of a date, carried through any
    split since the count's own date (AAPL's cover page said 4.28B shares
    for two months after its 4-for-1 split)."""

    def __init__(self, body: dict[str, Any], splits: Sequence[tuple[date, float]] = ()):
        facts = body.get("facts") or {}
        # The cover page first; GOOGL tags none there, only the balance sheet.
        for space, tag in (
            ("dei", "EntityCommonStockSharesOutstanding"),
            ("us-gaap", "CommonStockSharesOutstanding"),
        ):
            self.points = self._read((facts.get(space) or {}).get(tag) or {})
            if self.points:
                break
        self.splits = sorted(splits)

    @staticmethod
    def _read(concept: dict[str, Any]) -> list[tuple[date, date, float]]:
        # A filing lists one count per share class; the same count can also
        # appear twice. Distinct values within a filing are summed.
        by_accn: dict[str, tuple[date, date, set[float]]] = {}
        for raw in (concept.get("units") or {}).get("shares") or []:
            end, filed = _day(raw.get("end")), _day(raw.get("filed"))
            if end is None or filed is None or not raw.get("val"):
                continue
            accn = str(raw.get("accn") or f"{filed}")
            have = by_accn.get(accn)
            if have is None or end > have[1]:
                by_accn[accn] = (filed, end, {float(raw["val"])})
            elif end == have[1]:
                have[2].add(float(raw["val"]))
        return sorted((filed, end, sum(vals)) for filed, end, vals in by_accn.values())

    def at(self, day: date) -> float | None:
        best = None
        for filed, end, total in self.points:
            if filed > day:
                break
            best = (filed, end, total)
        if best is None or (day - best[0]).days > STALE_DAYS:
            return None
        _, counted_on, total = best
        for when, ratio in self.splits:
            if counted_on < when <= day and ratio > 0:
                total *= ratio
        return total


def splits_of(bars: pl.DataFrame) -> list[tuple[date, float]]:
    """(date, ratio) of every split in the bars."""
    if "Stock Splits" not in bars.columns:
        return []
    hits = bars.filter(pl.col("Stock Splits").fill_null(0) > 0)
    return list(zip(hits[DATE].to_list(), hits["Stock Splits"].to_list(), strict=True))


# --- prices as traded -----------------------------------------------------------


def raw_close(bars: pl.DataFrame) -> pl.DataFrame:
    """(date, close) at the price that actually traded. The bar store holds
    split- and dividend-adjusted closes (auto_adjust); walking back from
    today, each split divides and each dividend scales earlier closes, so
    undoing both gives the close a filing's share count matches.
    Checked against AAPL 2020-08-28 ($499.23) and NVDA 2024-06-07
    ($1,208.88), the sessions before their splits."""
    frame = bars.sort(DATE)
    close = frame["Close"].to_list()
    divs = frame["Dividends"].to_list() if "Dividends" in frame.columns else [0.0] * len(close)
    splits = (
        frame["Stock Splits"].to_list() if "Stock Splits" in frame.columns else [0.0] * len(close)
    )
    factor = 1.0
    out = [math.nan] * len(close)
    for i in range(len(close) - 1, -1, -1):
        c = close[i]
        if c is None or not math.isfinite(c):
            continue
        out[i] = c / factor
        # Events dated day i change every close before day i.
        d = divs[i] or 0.0
        if d > 0 and i > 0 and close[i - 1]:
            prev_raw = close[i - 1] / factor + d
            factor *= 1 - d / prev_raw
        s = splits[i] or 0.0
        if s > 0:
            factor /= s
    return pl.DataFrame({DATE: frame[DATE], "close": out})


# --- the monthly panel ------------------------------------------------------------


def _ratio(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return a / b


def company_rows(
    ticker: str,
    body: dict[str, Any],
    closes: dict[date, float],
    month_ends: list[date],
    splits: Sequence[tuple[date, float]] = (),
) -> list[dict[str, Any]]:
    """One row per month-end with what `ticker` had filed by then. `closes`
    maps each month-end to that day's (or the last earlier) traded close."""
    flows = {k: Known(_facts(body, "us-gaap", tags, "USD")) for k, tags in FLOWS.items()}
    instants = {k: Known(_facts(body, "us-gaap", tags, "USD")) for k, tags in INSTANTS.items()}
    eps = Known(_facts(body, "us-gaap", EPS_TAGS, "USD/shares"))
    shares = Shares(body, list(splits))
    rows = []
    for day in month_ends:
        for k in (*flows.values(), *instants.values(), eps):
            k.advance(day)
        t = {k: latest_ttm(known, day) for k, known in flows.items()}
        val = {k: (v[1] if v else None) for k, v in t.items()}
        bal = {k: latest_instant(known, day) for k, known in instants.items()}
        revenue, revenue_before = val["revenue"], (t["revenue"] or (None, None, None))[2]
        gross = val["gross_profit"]
        if gross is None and revenue is not None and val["cost_of_revenue"] is not None:
            gross = revenue - val["cost_of_revenue"]
        fcf = None
        if val["cfo"] is not None:
            fcf = val["cfo"] - abs(val["capex"] or 0.0)
        debt = None
        if any(bal[k] is not None for k in ("long_term_debt", "current_debt")):
            debt = sum(
                bal[k] or 0.0 for k in ("long_term_debt", "current_debt", "short_term_borrowings")
            )
        equity = bal["equity"]
        price, count = closes.get(day), shares.at(day)
        cap = price * count if price and count else None
        growth = None
        if revenue is not None and revenue_before and revenue_before > 0:
            growth = revenue / revenue_before - 1
        rows.append(
            {
                DATE: day,
                "ticker": ticker,
                "market_cap": cap,
                "revenue_growth": growth,
                "operating_margin": _ratio(val["operating_income"], revenue)
                if revenue and revenue > 0
                else None,
                "fcf_yield": _ratio(fcf, cap),
                "free_cash_flow": fcf,
                "operating_cash_flow": val["cfo"],
                "debt_to_equity": _ratio(debt, equity) if equity and equity > 0 else None,
                "roe": _ratio(val["net_income"], equity) if equity and equity > 0 else None,
                "gross_profitability": _ratio(gross, bal["assets"]),
                "sue": sue(quarterly_eps(eps.periods), day),
            }
        )
    return rows


COLUMNS: dict[str, Any] = {
    DATE: pl.Date,
    "ticker": pl.String,
    "market_cap": pl.Float64,
    "revenue_growth": pl.Float64,
    "operating_margin": pl.Float64,
    "fcf_yield": pl.Float64,
    "free_cash_flow": pl.Float64,
    "operating_cash_flow": pl.Float64,
    "debt_to_equity": pl.Float64,
    "roe": pl.Float64,
    "gross_profitability": pl.Float64,
    "sue": pl.Float64,
}


def build_panel(
    tickers: list[str],
    bars: dict[str, pl.DataFrame],
    month_ends: list[date],
    *,
    cache_dir: str,
    refresh: bool = False,
) -> pl.DataFrame:
    """Long frame of COLUMNS: point-in-time fundamentals per (month-end,
    ticker), for each ticker the SEC maps to a CIK and the bar store prices."""
    from ..data.sec_edgar import load_ticker_cik_map

    ciks = load_ticker_cik_map()
    rows: list[dict[str, Any]] = []
    missing = []
    for n, ticker in enumerate(tickers, 1):
        cik = ciks.get(ticker) or ciks.get(ticker.replace("-", "."))
        frame = bars.get(ticker)
        body = company_facts(cik, cache_dir, refresh=refresh) if cik and frame is not None else None
        if body is None or frame is None:
            missing.append(ticker)
            continue
        traded = raw_close(frame).drop_nans("close")
        closes = {}
        if not traded.is_empty():
            days = traded[DATE].to_list()
            values = traded["close"].to_list()
            j = 0
            for day in month_ends:
                while j + 1 < len(days) and days[j + 1] <= day:
                    j += 1
                if days[j] <= day and (day - days[j]).days <= 7:
                    closes[day] = values[j]
        rows.extend(company_rows(ticker, body, closes, month_ends, splits_of(frame)))
        if n % 100 == 0:
            logger.info("SEC fundamentals: %d/%d companies", n, len(tickers))
    if missing:
        logger.info(
            "SEC fundamentals: %d ticker(s) without facts or prices (%s%s)",
            len(missing),
            ", ".join(missing[:10]),
            ", ..." if len(missing) > 10 else "",
        )
    return (
        pl.DataFrame(rows, schema=COLUMNS, orient="row") if rows else pl.DataFrame(schema=COLUMNS)
    )


def month_ends(start: date, end: date) -> list[date]:
    out = []
    day = date(start.year, start.month, 1)
    while True:
        nxt = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
        last = nxt - timedelta(days=1)
        if last > end:
            return out
        if last >= start:
            out.append(last)
        day = nxt


__all__ = [
    "COLUMNS",
    "Known",
    "Shares",
    "build_panel",
    "company_facts",
    "company_rows",
    "latest_ttm",
    "month_ends",
    "quarterly_eps",
    "raw_close",
    "splits_of",
    "sue",
    "ttm",
]
