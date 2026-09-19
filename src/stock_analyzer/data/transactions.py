"""Transaction history → tax lots from SnapTrade.

Pulls BUY/SELL activities (dividend reinvestments — REI — count as buys:
they are purchases for lot and wash-sale purposes) over a lookback window
and aggregates per-ticker
"tax lots" so the rebalance pipeline can do specific-ID lot selection:
"Sell the lot dated YYYY-MM-DD (long-term, $X gain) — not the one from
last month (short-term, ordinary-income tax)."

US tax treatment encoded:
  - held MORE than one year (sold on/after `long_term_on`, the day after
    the purchase anniversary) → long-term (preferential rate)
  - otherwise → short-term (ordinary income rate)

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

__all__ = [
    "fetch_transaction_history",
    "to_tax_payloads",
    "fetch_cash_activity",
    "fetch_activities_by_account",
]

# Activity types that are purchases (lots): plain buys and dividend
# reinvestments (a DRIP purchase within 30 days is a wash-sale purchase too).
_BUY_TYPES = {"BUY", "REI"}
# Money or securities moving in/out of the portfolio — not investment
# performance, so time-weighted returns take them out.
_FLOW_TYPES = {"CONTRIBUTION", "WITHDRAWAL", "TRANSFER", "DEPOSIT"}


def is_option_activity(activity: dict[str, Any]) -> bool:
    """Option trades carry the UNDERLYING's symbol on some brokerages
    (Robinhood: an NFLX call reads as "NFLX"), so they must be told apart
    by `option_symbol` — otherwise a $1.23 option premium becomes an NFLX
    share lot at $1.23."""
    return bool(activity.get("option_symbol"))


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


def _activity_account_name(activity: dict[str, Any], account_id_to_name: dict[str, str]) -> str:
    """Pull a friendly account name from an activity. The activity's `account`
    field may be a nested dict (AccountSimple) or an id string depending on
    how the SDK deserializes."""
    acc = activity.get("account")
    if isinstance(acc, dict):
        return (
            account_id_to_name.get(str(acc.get("id") or ""), "")
            or acc.get("name")
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
                account_id,
                offset,
                e,
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


def _live_activities_by_account(
    start: date, end: date, *, since: dict[str, date] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """{account label: activities in [start, end]} straight from SnapTrade;
    `since` overrides the start per account label. {} when SnapTrade is
    unavailable."""
    try:
        user_id, user_secret = _credentials()
        client = _client()
        accounts = (
            _unwrap(
                client.account_information.list_user_accounts(
                    user_id=user_id, user_secret=user_secret
                )
            )
            or []
        )
    except Exception as e:
        logger.warning("Cannot fetch brokerage activities: %s", e)
        return {}

    from .brokerage import account_labels

    out: dict[str, list[dict[str, Any]]] = {}
    for acc_id, label in account_labels(accounts).items():
        rows = _fetch_account_activities(
            client, user_id, user_secret, acc_id, (since or {}).get(label, start), end
        )
        logger.info("Account %r: %d activities", label, len(rows))
        out[label] = rows
    return out


def activities_by_account(
    *, start: date | None, db_path: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """{account label: activities since `start` (None = all)}.

    With `db_path`, the stored activity history is brought up to date
    (only new activity is fetched) and read from the database — full
    history, not just the API window. Without it, SnapTrade is asked
    directly for `start` onward (3 years when `start` is None)."""
    if db_path:
        from .activity_ledger import ledger_activities

        return ledger_activities(db_path, start=start)
    today = date.today()
    return _live_activities_by_account(start or today - timedelta(days=3 * 365), today)


def fetch_transaction_history(
    years_back: int = 3, *, db_path: str | None = None
) -> dict[str, TickerTaxSummary]:
    """Pull BUY/SELL transactions, group by ticker into tax lots.

    With `db_path` every stored activity counts (older purchases don't fall
    out of a 3-year window, so lot splits and FIFO stay right); without
    it, SnapTrade's last `years_back` years.
    """
    today = date.today()
    start = None if db_path else today - timedelta(days=years_back * 365)
    by_account = activities_by_account(start=start, db_path=db_path)
    # Tag each activity with its account label so lots carry the same
    # name everywhere, however the SDK shaped the nested account field.
    activities = [
        {**a, "account": {"name": label}} for label, rows in by_account.items() for a in rows
    ]
    account_id_to_name: dict[str, str] = {}
    logger.info("Building tax lots from %d activities", len(activities))

    working: dict[str, TickerTaxSummaryMut] = {}
    for activity in activities:
        ticker = _extract_ticker(activity)
        if not ticker or is_option_activity(activity):
            continue
        activity_type = (activity.get("type") or "").upper()
        account_name = _activity_account_name(activity, account_id_to_name)
        summary = working.setdefault(ticker, TickerTaxSummaryMut(ticker=ticker))

        if activity_type in _BUY_TYPES:
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
            except ValueError, TypeError:
                continue
            summary.total_units_sold += units_sold
            sell_date = _coerce_date(activity.get("trade_date") or activity.get("settlement_date"))
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

    logger.info("Built tax summaries for %d tickers", len(working))
    # Freeze each per-ticker aggregate back into the immutable public type.
    return {t: TickerTaxSummary.model_validate(mut.model_dump()) for t, mut in working.items()}


def to_tax_payloads(
    summaries: dict[str, TickerTaxSummary],
) -> dict[str, dict[str, Any]]:
    """Convert summaries to JSON-ready payloads keyed by ticker."""
    return {ticker: s.to_payload() for ticker, s in summaries.items()}


def fetch_cash_activity(
    days_back: int = 400, *, db_path: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """External cash flows and dividends over the last `days_back` days.

    {"flows": [{date, amount, account, type}], — CONTRIBUTION / WITHDRAWAL /
                 TRANSFER (incl. securities moved in by ACAT, at value),
     "dividends": [{date, ticker, amount, account, reinvested}]}

    `reinvested` is True when the same account shows a dividend
    reinvestment (REI) of that ticker on the same day. Empty lists when
    no activity is available."""
    out: dict[str, list[dict[str, Any]]] = {"flows": [], "dividends": []}
    start = date.today() - timedelta(days=days_back)
    reinvested: set[tuple[str, str, date]] = set()
    dividends: list[dict[str, Any]] = []
    for label, rows in activities_by_account(start=start, db_path=db_path).items():
        for a in rows:
            kind = (a.get("type") or "").upper()
            day = _coerce_date(a.get("trade_date") or a.get("settlement_date"))
            try:
                amount = float(a.get("amount") or 0)
            except ValueError, TypeError:
                continue
            if day is None or day < start:
                continue
            if kind in _FLOW_TYPES and amount:
                out["flows"].append({"date": day, "amount": amount, "account": label, "type": kind})
            elif kind == "DIVIDEND" and amount > 0:
                ticker = _extract_ticker(a)
                if ticker:
                    dividends.append(
                        {"date": day, "ticker": ticker, "amount": amount, "account": label}
                    )
            elif kind == "REI":
                ticker = _extract_ticker(a)
                if ticker:
                    reinvested.add((label, ticker, day))
    for d in dividends:
        d["reinvested"] = (d["account"], d["ticker"], d["date"]) in reinvested
    out["dividends"] = dividends
    return out


def fetch_activities_by_account(
    years_back: int = 3, *, db_path: str | None = None
) -> dict[str, list[dict[str, Any]]]:
    """{account label: raw activities} — full stored history with
    `db_path`, else SnapTrade's last `years_back` years."""
    start = None if db_path else date.today() - timedelta(days=years_back * 365)
    return activities_by_account(start=start, db_path=db_path)
