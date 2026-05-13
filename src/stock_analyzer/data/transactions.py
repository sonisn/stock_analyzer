"""Transaction history → tax lots from SnapTrade.

Pulls BUY/SELL activities over a lookback window and aggregates per-ticker
"tax lots" so the rebalance pipeline can do specific-ID lot selection:
"Sell the lot dated YYYY-MM-DD (long-term, $X gain) — not the one from
last month (short-term, ordinary-income tax)."

US tax treatment encoded:
  - days_held >= 365 → long-term (preferential capital gains rate)
  - days_held <  365 → short-term (ordinary income rate)

This module returns RAW lot data; the LLM reviewer/rebalancer reasons
about which specific lots to sell per recommendation.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..logging import get_logger
from .brokerage import _client, _credentials, _extract_ticker, _unwrap

logger = get_logger(__name__)

# 365 days = long-term holding period for US capital gains.
LONG_TERM_DAYS = 365


def _coerce_date(value: Any) -> date | None:
    """SnapTrade activity dates come back as datetime, date, or ISO string —
    normalize to a plain `date`."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
        except ValueError:
            return None
    return None


class Lot(BaseModel):
    """A single BUY transaction — i.e. one tax lot."""

    model_config = ConfigDict(frozen=True)

    date: str          # ISO date of purchase
    units: float       # shares acquired
    price: float       # per-share purchase price
    total_cost: float  # units * price + fee
    fee: float
    days_held: int
    is_long_term: bool
    account: str

    @classmethod
    def from_activity(
        cls, activity: dict[str, Any], account_name: str, today: date
    ) -> Lot | None:
        try:
            d = _coerce_date(
                activity.get("trade_date") or activity.get("settlement_date")
            )
            if d is None:
                return None
            units = float(activity.get("units") or 0)
            price = float(activity.get("price") or 0)
            fee = float(activity.get("fee") or 0)
            if units <= 0 or price <= 0:
                return None
            days_held = (today - d).days
            return cls(
                date=d.isoformat(),
                units=units,
                price=price,
                total_cost=units * price + fee,
                fee=fee,
                days_held=days_held,
                is_long_term=days_held >= LONG_TERM_DAYS,
                account=account_name,
            )
        except (ValueError, TypeError) as e:
            logger.debug("Could not parse activity: %s", e)
            return None


class TickerTaxSummary(BaseModel):
    ticker: str
    lots: list[Lot] = Field(default_factory=list)
    total_units_bought: float = 0.0
    total_units_sold: float = 0.0
    total_cost_basis: float = 0.0
    short_term_lot_count: int = 0
    long_term_lot_count: int = 0
    short_term_units: float = 0.0
    long_term_units: float = 0.0
    # SELL transactions within the last 60 days — used by the rebalancer
    # for wash-sale awareness (re-buying within 30 days of a loss-sale
    # disallows the loss for tax purposes).
    recent_sells_60d: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def current_units(self) -> float:
        return self.total_units_bought - self.total_units_sold

    @property
    def avg_cost(self) -> float:
        return (
            self.total_cost_basis / self.total_units_bought
            if self.total_units_bought
            else 0
        )

    def to_payload(self) -> dict[str, Any]:
        """Dict suitable for inclusion in the reviewer JSON payload."""
        # Sort lots newest-first so most-recent (shortest-held) appear at top.
        lots_sorted = sorted(self.lots, key=lambda x: x.date, reverse=True)
        return {
            "current_units_held": self.current_units,
            "total_units_bought": self.total_units_bought,
            "total_units_sold": self.total_units_sold,
            "average_cost_basis_per_share": round(self.avg_cost, 4),
            "lot_count": len(self.lots),
            "short_term_lots": self.short_term_lot_count,
            "long_term_lots": self.long_term_lot_count,
            "short_term_units": self.short_term_units,
            "long_term_units": self.long_term_units,
            "lots": [
                {
                    "date": lot.date,
                    "units": lot.units,
                    "price_per_share": round(lot.price, 4),
                    "total_cost": round(lot.total_cost, 2),
                    "days_held": lot.days_held,
                    "treatment": "long_term" if lot.is_long_term else "short_term",
                    "account": lot.account,
                }
                for lot in lots_sorted
            ],
            # Wash-sale flag data. Compare sale_price to avg_cost to estimate
            # whether the sell was at a loss; the LLM uses this to avoid
            # recommending re-purchase within 30 days.
            "recent_sells_60d": sorted(
                self.recent_sells_60d,
                key=lambda x: x.get("date", ""),
                reverse=True,
            ),
        }


def _activity_account_name(
    activity: dict[str, Any], account_id_to_name: dict[str, str]
) -> str:
    """Pull a friendly account name from an activity. The activity's `account`
    field may be a nested dict (AccountSimple) or an id string depending on
    how the SDK deserializes."""
    acc = activity.get("account")
    if isinstance(acc, dict):
        return (
            acc.get("name")
            or acc.get("number")
            or account_id_to_name.get(acc.get("id") or "", "")
            or "unknown"
        )
    if isinstance(acc, str):
        return account_id_to_name.get(acc, acc)
    return "unknown"


def _fetch_account_activities(
    client: Any,
    user_id: str,
    user_secret: str,
    account_id: str,
    start_date_: date,
    end_date_: date,
    *,
    page_size: int = 1000,
    max_pages: int = 20,
) -> list[dict[str, Any]]:
    """Paginate get_account_activities until exhausted. Returns a flat list."""
    all_activities: list[dict[str, Any]] = []
    offset = 0
    for _ in range(max_pages):
        try:
            resp = _unwrap(
                client.account_information.get_account_activities(
                    user_id=user_id,
                    user_secret=user_secret,
                    account_id=account_id,
                    start_date=start_date_,
                    end_date=end_date_,
                    offset=offset,
                    limit=page_size,
                )
            )
        except Exception as e:
            logger.warning(
                "Could not fetch activities for account %s (offset=%d): %s",
                account_id, offset, e,
            )
            return all_activities
        # SnapTrade may return either a paginated dict {data: [...], pagination: ...}
        # or a flat list depending on SDK version. Handle both.
        if isinstance(resp, dict):
            page = resp.get("data") or []
        elif isinstance(resp, list):
            page = resp
        else:
            page = []
        if not page:
            break
        all_activities.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return all_activities


def fetch_transaction_history(years_back: int = 3) -> dict[str, TickerTaxSummary]:
    """Pull BUY/SELL transactions over the lookback window, group by ticker.

    Uses SnapTrade's per-account get_account_activities (the top-level
    get_activities was deprecated and returns 410 Gone). Loops over
    every connected account and paginates within each.
    """
    try:
        user_id, user_secret = _credentials()
        client = _client()
    except RuntimeError as e:
        logger.warning("Cannot fetch transactions: %s", e)
        return {}

    today = date.today()
    start_date_ = today - timedelta(days=years_back * 365)

    try:
        accounts = _unwrap(
            client.account_information.list_user_accounts(
                user_id=user_id, user_secret=user_secret
            )
        ) or []
    except Exception as e:
        logger.warning("Could not list accounts for transactions: %s", e)
        return {}

    account_id_to_name: dict[str, str] = {
        str(a.get("id")): (
            a.get("name")
            or a.get("institution_name")
            or str(a.get("id"))
            or "unknown"
        )
        for a in accounts
        if a.get("id")
    }

    activities: list[dict[str, Any]] = []
    for acc_id, acc_name in account_id_to_name.items():
        acc_activities = _fetch_account_activities(
            client, user_id, user_secret, acc_id, start_date_, today
        )
        logger.info("Account %r: %d activities", acc_name, len(acc_activities))
        activities.extend(acc_activities)

    logger.info(
        "Fetched %d total activities over %d-year lookback (from %s)",
        len(activities),
        years_back,
        start_date_.isoformat(),
    )

    summaries: dict[str, TickerTaxSummary] = {}
    for activity in activities:
        ticker = _extract_ticker(activity)
        if not ticker:
            continue
        activity_type = (activity.get("type") or "").upper()
        account_name = _activity_account_name(activity, account_id_to_name)
        summary = summaries.setdefault(ticker, TickerTaxSummary(ticker=ticker))

        if activity_type == "BUY":
            lot = Lot.from_activity(activity, account_name, today)
            if lot:
                summary.lots.append(lot)
                summary.total_units_bought += lot.units
                summary.total_cost_basis += lot.total_cost
                if lot.is_long_term:
                    summary.long_term_lot_count += 1
                    summary.long_term_units += lot.units
                else:
                    summary.short_term_lot_count += 1
                    summary.short_term_units += lot.units
        elif activity_type == "SELL":
            try:
                units_sold = abs(float(activity.get("units") or 0))
                sell_price = float(activity.get("price") or 0)
            except (ValueError, TypeError):
                continue
            summary.total_units_sold += units_sold
            sell_date = _coerce_date(
                activity.get("trade_date") or activity.get("settlement_date")
            )
            if sell_date and units_sold > 0:
                days_ago = (today - sell_date).days
                if 0 <= days_ago <= 60:
                    summary.recent_sells_60d.append(
                        {
                            "date": sell_date.isoformat(),
                            "units": units_sold,
                            "sale_price": sell_price,
                            "days_ago": days_ago,
                            "account": account_name,
                        }
                    )

    logger.info(
        "Built tax summaries for %d tickers over %d-year lookback",
        len(summaries),
        years_back,
    )
    return summaries


def to_tax_payloads(
    summaries: dict[str, TickerTaxSummary],
) -> dict[str, dict[str, Any]]:
    """Convert summaries to JSON-ready payloads keyed by ticker."""
    return {ticker: s.to_payload() for ticker, s in summaries.items()}
