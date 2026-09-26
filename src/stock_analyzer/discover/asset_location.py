"""Which holdings belong in which account.

The same stock costs different amounts of tax depending on where it sits.
In a taxable account its dividends are taxed every year (REIT dividends
at the ordinary rate, most others at the long-term rate), and option
premium written against it is a short-term gain, taxed as income. In a
Traditional IRA or 401(k) none of that is taxed until money comes out;
in a Roth or an HSA, never.

So the tax-inefficient holdings — high yield, REITs, the ones calls and
puts are written on — belong in the tax-advantaged accounts, and the
buy-and-hold, low-yield ones in the taxable account, where growth is only
taxed on sale, at the long-term rate.

Moving a holding costs something: selling it in the taxable account
realizes its gain. So a move is only suggested as a swap that keeps the
overall portfolio the same — sell the inefficient stock in the taxable
account and buy the efficient one there; sell the efficient one in the
IRA and buy the inefficient one there — and only when the yearly tax it
saves pays back the tax on the sale within a few years.

Everything here is arithmetic on data already fetched. No LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

# Real estate investment trusts pay mostly non-qualified dividends.
_ORDINARY_DIVIDEND_SECTORS = {"REAL ESTATE"}
_ORDINARY_DIVIDEND_INDUSTRY_WORDS = ("REIT",)
# A holding this small is not worth a line.
_MIN_POSITION_USD = 1000.0
# A swap partner must carry at most this much tax drag per dollar held
# in a taxable account (0.5% a year) to be worth moving there.
_EFFICIENT_DRAG_PER_DOLLAR = 0.005
# Days after a loss sale in which buying the same stock — in ANY account,
# an IRA included — makes it a wash sale.
WASH_SALE_DAYS = 30


@dataclass
class TickerTaxFacts:
    """What decides a stock's tax cost: its trailing dividend yield
    (a fraction, 0.03 = 3%) and whether those dividends are ordinary income."""

    dividend_yield: float = 0.0
    ordinary_dividends: bool = False


@dataclass
class Placement:
    """One holding in one account."""

    ticker: str
    account: str
    kind: str  # taxable | tax_deferred | tax_free
    value: float
    cost: float
    dividend_yield: float
    ordinary_dividends: bool
    # Premium written on this ticker in this account over the last year.
    option_premium: float = 0.0
    # What holding it in a taxable account costs in tax each year.
    drag_if_taxable: float = 0.0

    @property
    def drag_per_dollar(self) -> float:
        return self.drag_if_taxable / self.value if self.value else 0.0

    @property
    def gain(self) -> float:
        return self.value - self.cost


@dataclass
class Swap:
    """Sell `inefficient` in the taxable account and buy `efficient` there;
    sell `efficient` in `into_account` and buy `inefficient` there."""

    inefficient: str
    efficient: str
    taxable_account: str
    into_account: str
    amount: float
    tax_on_sale: float
    yearly_saving: float
    breakeven_years: float | None
    realized_gain: float
    wash_sale_until: date | None = None


@dataclass
class AssetLocationReport:
    placements: list[Placement] = field(default_factory=list)
    swaps: list[Swap] = field(default_factory=list)
    # Premium written in taxable accounts on stocks an advantaged account
    # also holds 100+ shares of: {ticker: (premium, advantaged account)}.
    options_elsewhere: dict[str, tuple[float, str]] = field(default_factory=dict)
    taxable_drag: float = 0.0
    kinds: dict[str, str] = field(default_factory=dict)


def ordinary_dividends(sector: str | None, industry: str | None) -> bool:
    if (sector or "").upper() in _ORDINARY_DIVIDEND_SECTORS:
        return True
    return any(w in (industry or "").upper() for w in _ORDINARY_DIVIDEND_INDUSTRY_WORDS)


def trailing_yield(bars: Any, *, today: date | None = None) -> float:
    """Dividends paid over the last year over the latest close, from daily
    bars with a `Dividends` column (as `yf_gateway.daily_bars` returns)."""
    if bars is None or getattr(bars, "empty", True) or "Dividends" not in bars:
        return 0.0
    import pandas as pd

    today = today or date.today()
    idx = pd.DatetimeIndex(bars.index)
    since = pd.Timestamp(today - timedelta(days=365)).tz_localize(idx.tz)
    paid = float(bars.loc[idx >= since, "Dividends"].fillna(0).sum())
    close = float(bars["Close"].dropna().iloc[-1]) if "Close" in bars else 0.0
    return paid / close if close > 0 and paid > 0 else 0.0


def yearly_drag(
    value: float,
    facts: TickerTaxFacts,
    option_premium: float,
    *,
    long_term_rate: float,
    short_term_rate: float,
) -> float:
    """Tax a year of holding this in a taxable account costs."""
    div_rate = short_term_rate if facts.ordinary_dividends else long_term_rate
    return value * facts.dividend_yield * div_rate + max(option_premium, 0.0) * short_term_rate


def analyze(
    holdings: dict[str, list[dict[str, Any]]],
    kinds: dict[str, str],
    facts: dict[str, TickerTaxFacts],
    option_premium: dict[tuple[str, str], float],
    *,
    long_term_rate: float,
    short_term_rate: float,
    min_drag_usd: float = 150.0,
    max_breakeven_years: float = 3.0,
    prices: dict[str, float] | None = None,
    today: date | None = None,
) -> AssetLocationReport:
    """Where each holding sits, what that costs, and the swaps worth making.

    `holdings` is {account: brokerage rows}; `kinds` is {account: kind};
    `option_premium` is {(account, ticker): premium written in the last year}.
    """
    today = today or date.today()
    prices = prices or {}
    report = AssetLocationReport(kinds=dict(kinds))
    for account, rows in holdings.items():
        kind = kinds.get(account, "taxable")
        for h in rows:
            ticker = str(h.get("ticker") or "").upper()
            units = float(h.get("units") or 0)
            if not ticker or units <= 0:
                continue
            price = prices.get(ticker) or float(h.get("price") or 0)
            value = units * price
            if value < _MIN_POSITION_USD:
                continue
            f = facts.get(ticker, TickerTaxFacts())
            premium = option_premium.get((account, ticker), 0.0)
            report.placements.append(
                Placement(
                    ticker=ticker,
                    account=account,
                    kind=kind,
                    value=value,
                    cost=units * float(h.get("average_purchase_price") or 0),
                    dividend_yield=f.dividend_yield,
                    ordinary_dividends=f.ordinary_dividends,
                    option_premium=premium,
                    drag_if_taxable=yearly_drag(
                        value,
                        f,
                        premium,
                        long_term_rate=long_term_rate,
                        short_term_rate=short_term_rate,
                    ),
                )
            )
    report.placements.sort(key=lambda p: (p.kind != "taxable", -p.drag_if_taxable))
    report.taxable_drag = sum(p.drag_if_taxable for p in report.placements if p.kind == "taxable")
    report.swaps = _swaps(
        report.placements,
        long_term_rate=long_term_rate,
        min_drag_usd=min_drag_usd,
        max_breakeven_years=max_breakeven_years,
        today=today,
    )
    report.options_elsewhere = _options_elsewhere(holdings, kinds, option_premium)
    return report


def _swaps(
    placements: list[Placement],
    *,
    long_term_rate: float,
    min_drag_usd: float,
    max_breakeven_years: float,
    today: date,
) -> list[Swap]:
    inefficient = [
        p for p in placements if p.kind == "taxable" and p.drag_if_taxable >= min_drag_usd
    ]
    # Candidates to bring into the taxable account: efficient holdings in
    # tax-advantaged accounts, largest first so one swap can absorb a leg.
    partners = sorted(
        (
            p
            for p in placements
            if p.kind != "taxable" and p.drag_per_dollar <= _EFFICIENT_DRAG_PER_DOLLAR
        ),
        key=lambda p: -p.value,
    )
    room = {id(p): p.value for p in partners}
    out: list[Swap] = []
    for leg in sorted(inefficient, key=lambda p: -p.drag_if_taxable):
        partner = next(
            (q for q in partners if q.ticker != leg.ticker and room[id(q)] >= _MIN_POSITION_USD),
            None,
        )
        if partner is None:
            break
        amount = min(leg.value, room[id(partner)])
        share = amount / leg.value
        realized = leg.gain * share
        tax_on_sale = max(realized, 0.0) * long_term_rate
        saving = (leg.drag_per_dollar - partner.drag_per_dollar) * amount
        if saving <= 0:
            continue
        breakeven = tax_on_sale / saving if tax_on_sale > 0 else 0.0
        if breakeven > max_breakeven_years:
            continue
        room[id(partner)] -= amount
        out.append(
            Swap(
                inefficient=leg.ticker,
                efficient=partner.ticker,
                taxable_account=leg.account,
                into_account=partner.account,
                amount=amount,
                tax_on_sale=tax_on_sale,
                yearly_saving=saving,
                breakeven_years=breakeven,
                realized_gain=realized,
                # A loss sale followed by buying the same stock in the IRA
                # within 30 days loses the deduction for good.
                wash_sale_until=today + timedelta(days=WASH_SALE_DAYS + 1)
                if realized < 0
                else None,
            )
        )
    return out


def _options_elsewhere(
    holdings: dict[str, list[dict[str, Any]]],
    kinds: dict[str, str],
    option_premium: dict[tuple[str, str], float],
) -> dict[str, tuple[float, str]]:
    """Premium written in a taxable account on a stock that a tax-advantaged
    account holds a round lot of — the same calls could be written there."""
    lots: dict[str, tuple[float, str]] = {}
    for account, rows in holdings.items():
        if kinds.get(account, "taxable") == "taxable":
            continue
        for h in rows:
            ticker = str(h.get("ticker") or "").upper()
            units = float(h.get("units") or 0)
            if units >= 100 and units > lots.get(ticker, (0.0, ""))[0]:
                lots[ticker] = (units, account)
    out: dict[str, tuple[float, str]] = {}
    for (account, ticker), premium in option_premium.items():
        if kinds.get(account, "taxable") != "taxable" or premium <= 0 or ticker not in lots:
            continue
        prior = out.get(ticker, (0.0, lots[ticker][1]))
        out[ticker] = (prior[0] + premium, lots[ticker][1])
    return out


def option_premium_by_account(contracts: list[Any], *, since: date) -> dict[tuple[str, str], float]:
    """{(account, underlying): net premium} from short options opened on or
    after `since` (`data.options_income.summarize_contracts` rows)."""
    out: dict[tuple[str, str], float] = {}
    for c in contracts:
        if not getattr(c, "opened_short", False) or getattr(c, "day_trade", False):
            continue
        first = getattr(c, "first_day", None)
        if first is None or first < since:
            continue
        key = (c.account, str(c.underlying).upper())
        out[key] = out.get(key, 0.0) + float(c.net_premium)
    return out
