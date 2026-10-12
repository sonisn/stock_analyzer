"""The moonshot basket: MOONSHOT_TARGET_PCT of the portfolio spread evenly
over MOONSHOT_SLOTS small, fast-growing companies, held for ten years.

Why a basket and not a pick: the 20-bagger study (2026-10-12; 4,530 US
listings, SEC filings as known at each December 2012-2020) found that a
20x over ten years happened to 0.7% of stocks and over five to 0.11%, and
that no combination of fundamentals picked them: the traits that raised
the odds (small, near break-even, growing fast, issuing some shares) also
raised the odds of losing half. A rule cannot find the one; twenty equal
small bets can own it. The same study set the holding rules: the median
20x winner fell 56% on the way, and 40% were still under 2x after three
years — so the basket is never sold on price.

The rules, each from the study:
  - entry (`entry_reasons`): $150M-$2B market cap, revenue growth >= 25%,
    operating margin -20% to +5%, gross margin rising or >= 40%, share
    count up no more than 10% in a year, two years of cash at the current
    burn, $2M a day traded, and revenue (no pre-revenue companies);
  - spread (`fill_slots`): at most 3 per industry, biotech at most a
    quarter of the slots;
  - funding: new cash only — an open slot is named with its size and
    never paid for by a sale;
  - holding: no trim on a drop, no adds (sizes stay equal), no options
    written against a member (a call caps the only upside the basket is
    for); a member past 5% of the whole portfolio is trimmed back to 5%;
  - selling (`thesis_breaks`): only from the filings — under a year of
    cash, a going-concern doubt, shares up 15%+ in each of two years, or
    revenue down two years running.

Kept apart from the daily, weekly and rebalance runs on purpose: nothing
here changes what they suggest. `uv run moonshot check` is where the
sell rules are read; the holding rules are the investor's to keep.

Members are the investor's own choice, kept in the `moonshot_basket`
table (`uv run moonshot add TICKER`); the screen only proposes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from ..logging import get_logger
from ..model.sec_history import (
    FLOWS,
    Known,
    Shares,
    _facts,
    _gross,
    _gross_margin,
    latest_instant,
    latest_ttm,
)

logger = get_logger(__name__)

# --- entry rules ------------------------------------------------------------------

MIN_MARKET_CAP = 150e6
MAX_MARKET_CAP = 2e9
MIN_REVENUE_GROWTH = 0.25
MIN_OPERATING_MARGIN = -0.20
MAX_OPERATING_MARGIN = 0.05
MIN_GROSS_MARGIN = 0.40  # or rising (GM_RISING) from a year earlier
GM_RISING = 0.01
MAX_DILUTION = 0.10  # share count growth over the last year
MIN_RUNWAY_YEARS = 2.0
MIN_DOLLAR_VOLUME = 2e6

# --- spread -----------------------------------------------------------------------

MAX_PER_INDUSTRY = 3
MAX_BIOTECH_SHARE = 0.25
# SIC codes for drugs, biologics, diagnostics and commercial life-science
# research: binary outcomes, so capped as a group.
BIOTECH_SIC = frozenset({2833, 2834, 2835, 2836, 8731})

# --- when the story breaks --------------------------------------------------------

BREAK_RUNWAY_YEARS = 1.0
BREAK_DILUTION = 0.15  # in each of the last two years

# Cash and the short-term securities companies hold it in.
CASH_TAGS = (
    "CashAndCashEquivalentsAtCarryingValue",
    "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    "Cash",
)
SHORT_TERM_INVESTMENT_TAGS = (
    "ShortTermInvestments",
    "MarketableSecuritiesCurrent",
    "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
)


# --- what the filings say today ---------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """Trailing-twelve-month figures as filed by `day`, and the same a year
    and two years earlier. None where a company did not tag the figure."""

    day: date
    revenue: float | None = None
    revenue_1y: float | None = None
    revenue_2y: float | None = None
    operating_margin: float | None = None
    gross_margin: float | None = None
    gross_margin_1y: float | None = None
    free_cash_flow: float | None = None
    cash: float | None = None
    shares: float | None = None
    shares_1y: float | None = None
    shares_2y: float | None = None

    @property
    def revenue_growth(self) -> float | None:
        if self.revenue is None or not self.revenue_1y or self.revenue_1y <= 0:
            return None
        return self.revenue / self.revenue_1y - 1

    @property
    def gross_margin_change(self) -> float | None:
        if self.gross_margin is None or self.gross_margin_1y is None:
            return None
        return self.gross_margin - self.gross_margin_1y

    @property
    def dilution(self) -> float | None:
        """Share count growth over the last year."""
        return _growth(self.shares, self.shares_1y)

    @property
    def dilution_prior_year(self) -> float | None:
        """Share count growth over the year before that."""
        return _growth(self.shares_1y, self.shares_2y)

    @property
    def runway_years(self) -> float | None:
        """Years of cash at the trailing year's burn; infinite when free
        cash flow is positive, None when either figure is missing."""
        if self.free_cash_flow is None or self.cash is None:
            return None
        if self.free_cash_flow >= 0:
            return math.inf
        return self.cash / -self.free_cash_flow


def _growth(now: float | None, then: float | None) -> float | None:
    if now is None or not then or then <= 0:
        return None
    return now / then - 1


def snapshot(
    body: dict[str, Any], *, today: date, splits: Sequence[tuple[date, float]] = ()
) -> Snapshot:
    """`body` is the SEC company-facts JSON; `splits` the stock's splits,
    so a split is not read as a share issue."""
    flows = {k: Known(_facts(body, "us-gaap", tags, "USD")) for k, tags in FLOWS.items()}
    cash = Known(_facts(body, "us-gaap", CASH_TAGS, "USD"))
    securities = Known(_facts(body, "us-gaap", SHORT_TERM_INVESTMENT_TAGS, "USD"))
    shares = Shares(body, list(splits))

    def revenue_at(day: date) -> float | None:
        flows["revenue"].advance(day)
        got = latest_ttm(flows["revenue"], day)
        return got[1] if got else None

    # Known only moves forward: the earliest day first.
    revenue_2y = revenue_at(today - timedelta(days=730))
    revenue_1y = revenue_at(today - timedelta(days=365))
    for known in (*flows.values(), cash, securities):
        known.advance(today)
    t = {k: latest_ttm(known, today) for k, known in flows.items()}
    now = {k: (v[1] if v else None) for k, v in t.items()}
    before = {k: (v[2] if v else None) for k, v in t.items()}
    revenue = now["revenue"]
    if revenue is not None and before["revenue"] is not None:
        revenue_1y = before["revenue"]  # the same quarter-end a year back
    fcf = None
    if now["cfo"] is not None:
        fcf = now["cfo"] - abs(now["capex"] or 0.0)
    held = latest_instant(cash, today)
    extra = latest_instant(securities, today)
    return Snapshot(
        day=today,
        revenue=revenue,
        revenue_1y=revenue_1y,
        revenue_2y=revenue_2y,
        operating_margin=now["operating_income"] / revenue
        if revenue and revenue > 0 and now["operating_income"] is not None
        else None,
        gross_margin=_gross_margin(_gross(now), revenue),
        gross_margin_1y=_gross_margin(_gross(before), before["revenue"]),
        free_cash_flow=fcf,
        cash=None if held is None else held + (extra or 0.0),
        shares=shares.at(today),
        shares_1y=_in_todays_shares(shares, today - timedelta(days=365), today, splits),
        shares_2y=_in_todays_shares(shares, today - timedelta(days=730), today, splits),
    )


def _in_todays_shares(
    shares: Shares, day: date, today: date, splits: Sequence[tuple[date, float]]
) -> float | None:
    """The count on `day`, carried through every split since, so a split
    between then and today is not read as a share issue."""
    count = shares.at(day)
    if count is None:
        return None
    for when, ratio in splits:
        if day < when <= today and ratio > 0:
            count *= ratio
    return count


# --- the entry screen -------------------------------------------------------------


def entry_reasons(snap: Snapshot, *, market_cap: float | None, dollar_volume: float | None):
    """Why a company is NOT a candidate; empty when it passes every rule.
    A figure the filings do not carry fails its rule: an unknown is not
    evidence."""
    out: list[str] = []
    if market_cap is None or not MIN_MARKET_CAP <= market_cap <= MAX_MARKET_CAP:
        out.append("market cap outside $150M-$2B")
    if dollar_volume is None or dollar_volume < MIN_DOLLAR_VOLUME:
        out.append("trades under $2M a day")
    if not snap.revenue or snap.revenue <= 0:
        out.append("no revenue")
        return out
    g = snap.revenue_growth
    if g is None or g < MIN_REVENUE_GROWTH:
        out.append("revenue growth under 25%")
    m = snap.operating_margin
    if m is None or not MIN_OPERATING_MARGIN <= m <= MAX_OPERATING_MARGIN:
        out.append("operating margin outside -20% to +5%")
    gm, change = snap.gross_margin, snap.gross_margin_change
    if not (
        (gm is not None and gm >= MIN_GROSS_MARGIN) or (change is not None and change > GM_RISING)
    ):
        out.append("gross margin under 40% and not rising")
    d = snap.dilution
    if d is None or d > MAX_DILUTION:
        out.append("share count up more than 10% (or unknown)")
    r = snap.runway_years
    if r is None or r < MIN_RUNWAY_YEARS:
        out.append("under two years of cash")
    return out


@dataclass(frozen=True)
class Candidate:
    ticker: str
    name: str
    market_cap: float
    snap: Snapshot
    industry: str = ""
    sic: int | None = None

    @property
    def biotech(self) -> bool:
        return self.sic in BIOTECH_SIC


def fill_slots(
    candidates: Sequence[Candidate],
    *,
    open_slots: int,
    slots: int,
    held: Sequence[Candidate] = (),
) -> list[Candidate]:
    """Up to `open_slots` candidates within the spread rules, counting the
    members already held. Smallest first: in the study the 20x rate was
    highest under $300M (no other trait ranked them)."""
    per_industry: dict[str, int] = {}
    biotech = 0
    for c in held:
        per_industry[c.industry] = per_industry.get(c.industry, 0) + 1
        biotech += c.biotech
    max_biotech = int(slots * MAX_BIOTECH_SHARE)
    out: list[Candidate] = []
    for c in sorted(candidates, key=lambda c: c.market_cap):
        if len(out) >= open_slots:
            break
        if c.industry and per_industry.get(c.industry, 0) >= MAX_PER_INDUSTRY:
            continue
        if c.biotech and biotech >= max_biotech:
            continue
        out.append(c)
        per_industry[c.industry] = per_industry.get(c.industry, 0) + 1
        biotech += c.biotech
    return out


# --- when to sell -----------------------------------------------------------------


def thesis_breaks(snap: Snapshot | None, *, red_flags: Sequence[str] = ()) -> list[str]:
    """The filed reasons to sell a member. Price is never one."""
    out: list[str] = []
    if "going_concern" in red_flags:
        out.append("going-concern doubt in the latest filing")
    if snap is None:
        return out
    r = snap.runway_years
    if r is not None and r < BREAK_RUNWAY_YEARS:
        out.append(f"under a year of cash left (~{r * 12:.0f} months at the current burn)")
    d1, d2 = snap.dilution, snap.dilution_prior_year
    if d1 is not None and d2 is not None and d1 > BREAK_DILUTION and d2 > BREAK_DILUTION:
        out.append(f"share count up {d1:.0%} and {d2:.0%} in the last two years")
    rev, r1, r2 = snap.revenue, snap.revenue_1y, snap.revenue_2y
    if rev is not None and r1 is not None and r2 is not None and rev < r1 < r2:
        out.append(f"revenue down two years running (${r2 / 1e6:,.0f}M -> ${rev / 1e6:,.0f}M)")
    return out


# --- members ----------------------------------------------------------------------


@dataclass(frozen=True)
class Member:
    ticker: str
    added_on: str
    entry_price: float | None = None


# --- stored members ---------------------------------------------------------------


def load_members(db_path: str) -> list[Member]:
    """The current members (removed ones are kept in the table, not here)."""
    from sqlmodel import select

    from ..db.session import get_session
    from ..db.tables import MoonshotMember

    with get_session(db_path) as session:
        rows = session.exec(
            select(MoonshotMember).where(MoonshotMember.removed_on.is_(None))  # ty: ignore[unresolved-attribute]
        ).all()
        return [Member(r.ticker, r.added_on, r.entry_price) for r in rows]


def add_member(db_path: str, ticker: str, *, today: date, entry_price: float | None) -> None:
    from ..db.session import get_session
    from ..db.tables import MoonshotMember

    with get_session(db_path) as session:
        session.merge(
            MoonshotMember(
                ticker=ticker.upper(),
                added_on=today.isoformat(),
                entry_price=entry_price,
                removed_on=None,
                removed_reason=None,
            )
        )


def remove_member(db_path: str, ticker: str, *, today: date, reason: str) -> bool:
    from ..db.session import get_session
    from ..db.tables import MoonshotMember

    with get_session(db_path) as session:
        row = session.get(MoonshotMember, ticker.upper())
        if row is None or row.removed_on is not None:
            return False
        row.removed_on = today.isoformat()
        row.removed_reason = reason
        session.add(row)
        return True


# --- live data ------------------------------------------------------------------

# Members are checked weekly, so their filings are re-read sooner than the
# 30 days the studies use: a break is seen within a week of the filing.
MEMBER_FACTS_MAX_AGE_DAYS = 7


def live_snapshots(
    tickers: Sequence[str],
    *,
    cache_dir: str,
    today: date,
    max_age_days: float | None = None,
) -> dict[str, Snapshot]:
    """{ticker: Snapshot} from the SEC company facts, with each stock's
    splits from the bar store. Tickers the SEC has no facts for are left out."""
    from ..data import yf_gateway
    from ..data.sec_edgar import load_ticker_cik_map
    from ..model.sec_history import FACTS_MAX_AGE_DAYS, company_facts, splits_of

    ciks = load_ticker_cik_map()
    bars = yf_gateway.daily_bars_many(tickers, start=today - timedelta(days=800), what="moonshot")
    out: dict[str, Snapshot] = {}
    for t in tickers:
        cik = ciks.get(t) or ciks.get(t.replace("-", "."))
        if not cik:
            continue
        body = company_facts(
            cik,
            cache_dir,
            max_age_days=FACTS_MAX_AGE_DAYS if max_age_days is None else max_age_days,
        )
        if body is None:
            continue
        frame = bars.get(t)
        out[t] = snapshot(body, today=today, splits=splits_of(frame) if frame is not None else ())
    return out


def member_breaks(
    members: Sequence[Member], *, db_path: str, cache_dir: str, today: date
) -> dict[str, list[str]]:
    """{ticker: filed reasons to sell} for every member (empty list = intact)."""
    tickers = [m.ticker for m in members]
    if not tickers:
        return {}
    from ..data.filing_evidence import red_flags

    snaps = live_snapshots(
        tickers, cache_dir=cache_dir, today=today, max_age_days=MEMBER_FACTS_MAX_AGE_DAYS
    )
    flags = red_flags(db_path, tickers, today=today)
    return {t: thesis_breaks(snaps.get(t), red_flags=flags.get(t, ())) for t in tickers}


__all__ = [
    "Candidate",
    "Member",
    "Snapshot",
    "add_member",
    "entry_reasons",
    "fill_slots",
    "live_snapshots",
    "load_members",
    "member_breaks",
    "remove_member",
    "snapshot",
    "thesis_breaks",
]
