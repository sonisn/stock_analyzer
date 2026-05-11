"""SnapTrade brokerage integration — fetch holdings across connected accounts."""
from __future__ import annotations

import os
from typing import Any

from snaptrade_client import SnapTrade

from ..logging import get_logger

logger = get_logger(__name__)


def _client() -> SnapTrade:
    client_id = os.getenv("SNAPTRADE_CLIENT_ID")
    consumer_key = os.getenv("SNAPTRADE_CONSUMER_KEY")
    if not (client_id and consumer_key):
        raise RuntimeError(
            "SNAPTRADE_CLIENT_ID and SNAPTRADE_CONSUMER_KEY must be set"
        )
    return SnapTrade(client_id=client_id, consumer_key=consumer_key)


def _credentials() -> tuple[str, str]:
    user_id = os.getenv("SNAPTRADE_USER_ID")
    user_secret = os.getenv("SNAPTRADE_USER_SECRET")
    if not (user_id and user_secret):
        raise RuntimeError(
            "SNAPTRADE_USER_ID and SNAPTRADE_USER_SECRET must be set"
        )
    return user_id, user_secret


def _unwrap(resp: Any) -> Any:
    return resp.body if hasattr(resp, "body") else resp


def _extract_ticker(position: dict) -> str | None:
    """Walk the SnapTrade position payload to find the underlying ticker symbol."""
    sym = position.get("symbol")
    while isinstance(sym, dict):
        if isinstance(sym.get("symbol"), str):
            return sym["symbol"]
        sym = sym.get("symbol")
    return sym if isinstance(sym, str) else None


def fetch_portfolio_holdings() -> dict[str, list[dict]]:
    """Return holdings keyed by account name across all connected SnapTrade accounts."""
    user_id, user_secret = _credentials()
    client = _client()

    logger.info("Fetching SnapTrade accounts")
    accounts = _unwrap(
        client.account_information.list_user_accounts(
            user_id=user_id, user_secret=user_secret
        )
    ) or []
    logger.info("Found %d SnapTrade accounts", len(accounts))

    out: dict[str, list[dict]] = {}
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

        positions = (
            _unwrap(
                client.account_information.get_user_account_positions(
                    user_id=user_id,
                    user_secret=user_secret,
                    account_id=account_id,
                )
            )
            or []
        )

        holdings: list[dict] = []
        for p in positions:
            ticker = _extract_ticker(p)
            if not ticker:
                continue
            holdings.append(
                {
                    "ticker": ticker,
                    "units": p.get("units"),
                    "price": p.get("price"),
                    "average_purchase_price": p.get("average_purchase_price"),
                }
            )
        logger.info("Account %r: %d positions", account_name, len(holdings))
        out[account_name] = holdings

    return out


def fetch_total_cash() -> float | None:
    """Sum cash balances across all connected SnapTrade accounts.

    Returns total in USD-equivalent or None if the API call fails or no
    balance data is returned. Used by the rebalancer to size BUYs from
    cash + sale proceeds.
    """
    try:
        user_id, user_secret = _credentials()
        client = _client()
        accounts = _unwrap(
            client.account_information.list_user_accounts(
                user_id=user_id, user_secret=user_secret
            )
        ) or []
    except Exception as e:
        logger.warning("Could not list accounts for cash balance: %s", e)
        return None

    total: float = 0.0
    found_any = False
    for account in accounts:
        account_id = account.get("id")
        if not account_id:
            continue
        try:
            balances = (
                _unwrap(
                    client.account_information.get_user_account_balance(
                        user_id=user_id,
                        user_secret=user_secret,
                        account_id=account_id,
                    )
                )
                or []
            )
        except Exception as e:
            logger.warning("Balance fetch failed for account %s: %s", account_id, e)
            continue
        # SnapTrade returns a list of balances per currency. Sum cash entries.
        if isinstance(balances, dict):
            balances = [balances]
        for b in balances:
            cash = b.get("cash") if isinstance(b, dict) else None
            if cash is None:
                continue
            try:
                total += float(cash)
                found_any = True
            except (ValueError, TypeError):
                continue
    return total if found_any else None


def fetch_portfolio_tickers() -> list[str]:
    """Return de-duplicated, sorted list of tickers across all connected accounts."""
    holdings = fetch_portfolio_holdings()
    tickers: set[str] = set()
    for account_holdings in holdings.values():
        for h in account_holdings:
            tickers.add(h["ticker"])
    return sorted(tickers)
