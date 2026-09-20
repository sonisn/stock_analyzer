"""What the written options actually earned, and what they cost.

Selling calls against a holding shows up nowhere in the portfolio's
performance: the shares are valued at the market, the premium lands as
cash, and when a call is assigned the position simply disappears at the
strike. Graded that way a called-away winner looks like a sale into
strength and the premium that paid for it is invisible, which is exactly
backwards for someone running the wheel on long-term holdings.

Every option trade is in the brokerage activity ledger already, told
apart from share trades by its `option_symbol` (`is_option_activity`).
This groups them by contract so each one can be read as a position
rather than a stream of cash: opened short for a credit, then closed by
expiry (keep it all), by a buy-back (keep the difference), or by
assignment (keep it, and lose the shares at the strike).

Long options — bought to open — are reported separately and never
counted as income. Measured on the stored ledger 2026-09-20: $103,349
collected across 25 short sales against $60,728 spent on four long NFLX
positions, which is not $42,621 of "premium" and should never be
summarized as though it were.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..logging import get_logger
from ..models.market import OCCParseError
from .options_symbols import parse_occ
from .transactions import is_option_activity

logger = get_logger(__name__)

# Activity types that close a short option without cash changing hands.
_EXPIRED = "OPTIONEXPIRATION"
_ASSIGNED = "OPTIONASSIGNMENT"


@dataclass
class ContractResult:
    """One option contract, from open to close."""

    symbol: str
    underlying: str
    option_type: str  # "C" or "P"
    strike: float
    expiry: date
    account: str
    contracts: int = 0  # peak short size
    premium_collected: float = 0.0
    premium_paid: float = 0.0
    opened_short: bool = False
    opened_long: bool = False
    assigned: bool = False
    expired: bool = False
    # Opened and closed on one day, so which side came first cannot be
    # read from the ledger. A round trip inside a day is a trade, not
    # premium earned on a holding, and is reported apart from both.
    day_trade: bool = False
    first_day: date | None = None
    last_day: date | None = None

    @property
    def net_premium(self) -> float:
        return self.premium_collected + self.premium_paid

    @property
    def outcome(self) -> str:
        if self.day_trade:
            return "day trade"
        if self.assigned:
            return "assigned"
        if self.expired:
            return "expired"
        if self.premium_paid and self.opened_short:
            return "bought back"
        return "open"

    @property
    def shares_called_away(self) -> float:
        return self.contracts * 100.0 if (self.assigned and self.option_type == "C") else 0.0


def _day(activity: dict[str, Any]) -> date | None:
    raw = activity.get("trade_date") or activity.get("settlement_date")
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw)[:10])
    except TypeError, ValueError:
        return None


def _option_symbol(activity: dict[str, Any]) -> str | None:
    raw = activity.get("option_symbol")
    if isinstance(raw, dict):
        raw = raw.get("ticker") or raw.get("symbol") or raw.get("option_symbol")
    return str(raw) if raw else None


def summarize_contracts(
    by_account: dict[str, list[dict[str, Any]]],
    *,
    start: date | None = None,
    end: date | None = None,
) -> list[ContractResult]:
    """One `ContractResult` per option contract traded in the window.

    A contract is kept when any of its activity falls inside the window,
    so a call sold in May and assigned in September is not split in half.
    """
    # Gather first, fold second. Whether a contract was sold to OPEN (a
    # short) or sold to CLOSE (a long being exited) cannot be read from
    # the sign of one row — only from which trade came first. Folding in
    # arrival order read four long NFLX positions, later sold to close,
    # as 228 short contracts and put a $1,472 trading loss into premium
    # income.
    grouped: dict[tuple[str, str], list[tuple[date, dict[str, Any]]]] = {}
    for account, activities in (by_account or {}).items():
        for activity in activities:
            if not is_option_activity(activity):
                continue
            symbol = _option_symbol(activity)
            day = _day(activity)
            if not symbol or day is None:
                continue
            if (start and day < start) or (end and day > end):
                continue
            grouped.setdefault((account, symbol.strip()), []).append((day, activity))

    results: dict[tuple[str, str], ContractResult] = {}
    for (account, symbol), rows in grouped.items():
        try:
            parsed = parse_occ(symbol)
        except OCCParseError:
            logger.debug("Skipping unparseable option symbol %r", symbol)
            continue
        row = results[(account, symbol)] = ContractResult(
            symbol=symbol,
            underlying=parsed.ticker,
            option_type=parsed.option_type,
            strike=parsed.strike,
            expiry=parsed.expiry,
            account=account,
        )
        position = 0.0
        opened = False
        bought = sold = False
        for day, activity in sorted(rows, key=lambda r: r[0]):
            try:
                units = float(activity.get("units") or 0)
                amount = float(activity.get("amount") or 0)
            except TypeError, ValueError:
                continue
            kind = str(activity.get("type") or "").upper()
            row.first_day = min(row.first_day or day, day)
            row.last_day = max(row.last_day or day, day)

            if kind == _ASSIGNED:
                row.assigned = True
                continue
            if kind == _EXPIRED:
                row.expired = True
                continue
            if units < 0:
                sold = True
            elif units > 0:
                bought = True
            if not opened and units:
                # The first trade decides what this contract is.
                row.opened_short = units < 0
                row.opened_long = units > 0
                opened = True
            position += units
            # Peak exposure, not the biggest single fill: NVDA's
            # 2026-12-18 $285 calls were sold -3 then -1, which is four
            # contracts short, not three.
            row.contracts = max(row.contracts, int(abs(position)))
            if amount >= 0:
                row.premium_collected += amount
            else:
                row.premium_paid += amount
        if bought and sold and row.first_day == row.last_day and abs(position) < 1e-9:
            row.day_trade = True

    return sorted(results.values(), key=lambda r: (r.first_day or date.min, r.symbol))


@dataclass
class OptionIncome:
    """Short-option income for a period, and the longs kept out of it."""

    premium_collected: float = 0.0
    premium_paid_to_close: float = 0.0
    contracts_sold: int = 0
    expired: int = 0
    bought_back: int = 0
    assigned: int = 0
    shares_called_away: float = 0.0
    by_underlying: dict[str, dict[str, float]] = field(default_factory=dict)
    assignments: list[ContractResult] = field(default_factory=list)
    open_short: list[ContractResult] = field(default_factory=list)
    long_positions: list[ContractResult] = field(default_factory=list)
    day_trades: list[ContractResult] = field(default_factory=list)

    @property
    def net_premium(self) -> float:
        return self.premium_collected + self.premium_paid_to_close


def summarize_option_income(
    by_account: dict[str, list[dict[str, Any]]],
    *,
    start: date | None = None,
    end: date | None = None,
) -> OptionIncome:
    """Premium earned from SHORT options over the window, by underlying.

    Long options are listed but never added in: buying a call is a bet,
    not income, and mixing the two turns $103k of collected premium and
    $61k of speculation into a single misleading number.
    """
    income = OptionIncome()
    for row in summarize_contracts(by_account, start=start, end=end):
        if row.day_trade:
            income.day_trades.append(row)
            continue
        if row.opened_long and not row.opened_short:
            income.long_positions.append(row)
            continue
        if not row.opened_short:
            continue
        income.premium_collected += row.premium_collected
        income.premium_paid_to_close += row.premium_paid
        income.contracts_sold += row.contracts
        income.expired += 1 if row.expired else 0
        income.bought_back += 1 if row.outcome == "bought back" else 0
        if row.assigned:
            income.assigned += 1
            income.shares_called_away += row.shares_called_away
            income.assignments.append(row)
        if row.outcome == "open":
            income.open_short.append(row)
        bucket = income.by_underlying.setdefault(
            row.underlying, {"net_premium": 0.0, "contracts": 0.0, "shares_called_away": 0.0}
        )
        bucket["net_premium"] += row.net_premium
        bucket["contracts"] += row.contracts
        bucket["shares_called_away"] += row.shares_called_away
    return income


def fetch_option_income(
    *, start: date | None = None, end: date | None = None, db_path: str | None = None
) -> OptionIncome:
    """`summarize_option_income` over the stored activity ledger."""
    from .transactions import fetch_activities_by_account

    try:
        by_account = fetch_activities_by_account(db_path=db_path)
    except Exception as e:  # noqa: BLE001 — a report section is never worth a run
        logger.warning("Could not read option activity (%s)", e)
        return OptionIncome()
    return summarize_option_income(by_account, start=start, end=end)


__all__ = [
    "ContractResult",
    "OptionIncome",
    "fetch_option_income",
    "summarize_contracts",
    "summarize_option_income",
]
