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

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any

from ..logging import get_logger
from .brokerage import _client, _credentials, _extract_ticker, _unwrap

logger = get_logger(__name__)

# 365 days = long-term holding period for US capital gains.
LONG_TERM_DAYS = 365


@dataclass
class Lot:
    """A single BUY transaction — i.e. one tax lot."""

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
    ) -> "Lot | None":
        try:
            trade_date_str = (
                activity.get("trade_date") or activity.get("settlement_date")
            )
            if not trade_date_str:
                return None
            # Normalize to date (strip time + Z suffix if present).
            d = datetime.fromisoformat(
                trade_date_str.replace("Z", "+00:00")
            ).date()
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


@dataclass
class TickerTaxSummary:
    ticker: str
    lots: list[Lot] = field(default_factory=list)
    total_units_bought: float = 0.0
    total_units_sold: float = 0.0
    total_cost_basis: float = 0.0
    short_term_lot_count: int = 0
    long_term_lot_count: int = 0
    short_term_units: float = 0.0
    long_term_units: float = 0.0

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
        }


def fetch_transaction_history(years_back: int = 3) -> dict[str, TickerTaxSummary]:
    """Pull BUY/SELL transactions over the lookback window, group by ticker."""
    try:
        user_id, user_secret = _credentials()
        client = _client()
    except RuntimeError as e:
        logger.warning("Cannot fetch transactions: %s", e)
        return {}

    today = date.today()
    start_date = today - timedelta(days=years_back * 365)

    try:
        accounts = _unwrap(
            client.account_information.list_user_accounts(
                user_id=user_id, user_secret=user_secret
            )
        ) or []
    except Exception as e:
        logger.warning("Could not list accounts for transactions: %s", e)
        return {}

    summaries: dict[str, TickerTaxSummary] = {}

    for account in accounts:
        account_id = account.get("id")
        account_name = (
            account.get("name")
            or account.get("institution_name")
            or account_id
            or "unknown"
        )
        if not account_id:
            continue
        try:
            activities = _unwrap(
                client.transactions_and_reporting.get_user_account_activities(
                    user_id=user_id,
                    user_secret=user_secret,
                    account_id=account_id,
                    start_date=start_date.isoformat(),
                    end_date=today.isoformat(),
                )
            ) or []
        except Exception as e:
            logger.warning(
                "Could not fetch activities for %r: %s", account_name, e
            )
            continue

        logger.info("Account %r: %d activities", account_name, len(activities))
        for activity in activities:
            ticker = _extract_ticker(activity)
            if not ticker:
                continue
            activity_type = (activity.get("type") or "").upper()
            summary = summaries.setdefault(
                ticker, TickerTaxSummary(ticker=ticker)
            )
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
                    summary.total_units_sold += abs(
                        float(activity.get("units") or 0)
                    )
                except (ValueError, TypeError):
                    continue

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
