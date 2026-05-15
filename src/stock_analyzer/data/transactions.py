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

from ..logging import get_logger
from ..models.portfolio import Lot, TickerTaxSummary, TickerTaxSummaryMut
from .brokerage import _client, _credentials, _extract_ticker, _unwrap

logger = get_logger(__name__)

__all__ = ["fetch_transaction_history", "to_tax_payloads"]


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

    working: dict[str, TickerTaxSummaryMut] = {}
    for activity in activities:
        ticker = _extract_ticker(activity)
        if not ticker:
            continue
        activity_type = (activity.get("type") or "").upper()
        account_name = _activity_account_name(activity, account_id_to_name)
        summary = working.setdefault(ticker, TickerTaxSummaryMut(ticker=ticker))

        if activity_type == "BUY":
            lot = Lot.from_activity(
                activity,
                account_name,
                today,
                coerce_date=_coerce_date,
                logger=logger,
            )
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
        len(working),
        years_back,
    )
    # Freeze each per-ticker aggregate back into the immutable public type.
    return {
        t: TickerTaxSummary.model_validate(mut.model_dump())
        for t, mut in working.items()
    }


def to_tax_payloads(
    summaries: dict[str, TickerTaxSummary],
) -> dict[str, dict[str, Any]]:
    """Convert summaries to JSON-ready payloads keyed by ticker."""
    return {ticker: s.to_payload() for ticker, s in summaries.items()}
