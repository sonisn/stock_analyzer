"""SnapTrade brokerage integration — fetch holdings across connected accounts."""
from __future__ import annotations

import os
from typing import Any, Literal

from snaptrade_client import SnapTrade

from ..logging import get_logger
from .options_symbols import OCCParseError, parse_occ

logger = get_logger(__name__)


TaxStatus = Literal["taxable", "tax_advantaged"]


# Substring patterns that flag a name as a tax-advantaged account when
# SnapTrade's account `type` field isn't conclusive. Order matters — match
# the longest first to avoid e.g. "RothIRA" matching "IRA" then losing
# the Roth signal. Case-insensitive.
_TAX_ADVANTAGED_NAME_PATTERNS = (
    "ROTH IRA", "TRAD IRA", "TRADITIONAL IRA",
    "ROLLOVER IRA", "SEP IRA", "SIMPLE IRA",
    "ROTH", "IRA", "HSA", "401K", "401(K)", "403B", "457",
    "PENSION", "RRSP", "TFSA",
)
# SnapTrade-reported `type` values that map to tax-advantaged.
_TAX_ADVANTAGED_TYPES = {
    "IRA", "ROTH IRA", "TRADITIONAL IRA", "ROLLOVER IRA",
    "SEP IRA", "SIMPLE IRA", "401K", "401(K)", "403B", "457",
    "HSA", "RRSP", "TFSA", "RETIREMENT",
}


def _name_token_match(name_upper: str, pattern: str) -> bool:
    """True if `pattern` appears in `name_upper` as a standalone token —
    i.e. surrounded by non-alphanumeric characters or string boundaries.

    Custom logic instead of `\\b` regex because patterns like '401(K)'
    end with ')' which isn't word-boundary-compatible. We want:
      'Vanguard 401(k)' → matches '401(K)' (preceded by space, followed by EOL)
      'HSAFEcard' → does NOT match 'HSA' (followed by alphanumeric 'F')
      'Schwab HSA' → matches 'HSA' (preceded by space, followed by EOL)
    """
    pattern = pattern.upper()
    idx = 0
    while idx <= len(name_upper) - len(pattern):
        found = name_upper.find(pattern, idx)
        if found < 0:
            return False
        before = name_upper[found - 1] if found > 0 else " "
        after_idx = found + len(pattern)
        after = name_upper[after_idx] if after_idx < len(name_upper) else " "
        if not before.isalnum() and not after.isalnum():
            return True
        idx = found + 1
    return False


def classify_tax_status(
    account_type: str | None, account_name: str | None
) -> TaxStatus:
    """Determine whether trades in this account have tax consequences.

    Strategy:
      1. If SnapTrade returns a `type` field, check it against the
         known tax-advantaged set (covers Robinhood, Fidelity, Schwab,
         Vanguard, etc.).
      2. Fall back to substring match on the account name (catches
         custom names + brokers where `type` is generic 'Investment'
         or missing).

    Defaults to 'taxable' — that's the safer default because applying
    tax-cost analysis to a taxable account is correct; applying it to
    an IRA wastes signal but doesn't lose money. The reverse (skipping
    tax-cost analysis on a taxable account) IS a real risk.
    """
    if account_type:
        upper = account_type.upper().strip()
        if upper in _TAX_ADVANTAGED_TYPES:
            return "tax_advantaged"
    if account_name:
        upper_name = account_name.upper()
        for pattern in _TAX_ADVANTAGED_NAME_PATTERNS:
            if _name_token_match(upper_name, pattern):
                return "tax_advantaged"
    return "taxable"


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


def fetch_account_meta() -> dict[str, dict[str, Any]]:
    """Return {account_name: {id, type, tax_status, institution}} for every
    connected SnapTrade account. Used to tag each position with the
    account's tax treatment so the rebalancer can skip tax-cost analysis
    on IRA / HSA / 401k positions.
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
        logger.warning("Could not list accounts for tax-status meta: %s", e)
        return {}

    out: dict[str, dict[str, Any]] = {}
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
        account_type = (
            account.get("type")
            or account.get("account_type")
            or (account.get("meta") or {}).get("type")
        )
        out[account_name] = {
            "id": account_id,
            "type": account_type,
            "institution": account.get("institution_name"),
            "tax_status": classify_tax_status(account_type, account_name),
        }
    n_advantaged = sum(1 for m in out.values() if m["tax_status"] == "tax_advantaged")
    logger.info(
        "Account tax classification: %d taxable, %d tax-advantaged (out of %d)",
        len(out) - n_advantaged, n_advantaged, len(out),
    )
    return out


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


def fetch_open_option_positions() -> dict[str, int]:
    """Return {underlying_ticker: open_short_call_contracts} across every
    connected SnapTrade account.

    Only SHORT calls (units < 0) are counted — these are the positions
    that reduce the share count available to back NEW covered calls.
    Long calls and short puts are ignored. Returns {} when SnapTrade is
    unavailable or no positions are found (graceful degradation — the
    eligibility filter then simply subtracts zero).
    """
    try:
        user_id, user_secret = _credentials()
        client = _client()
    except Exception as e:
        logger.info("SnapTrade unavailable for option-position lookup: %s", e)
        return {}

    try:
        accounts = _unwrap(
            client.account_information.list_user_accounts(
                user_id=user_id, user_secret=user_secret,
            )
        ) or []
    except Exception as e:
        logger.info("SnapTrade list_user_accounts failed: %s", e)
        return {}

    coverage: dict[str, int] = {}
    for account in accounts:
        account_id = account.get("id") if isinstance(account, dict) else getattr(account, "id", None)
        if not account_id:
            continue
        try:
            positions = _unwrap(
                client.account_information.get_user_account_positions(
                    user_id=user_id,
                    user_secret=user_secret,
                    account_id=account_id,
                )
            ) or []
        except Exception as e:
            logger.info("SnapTrade positions fetch failed for %s: %s", account_id, e)
            continue

        for pos in positions:
            symbol = _extract_ticker(pos)
            if not isinstance(symbol, str):
                continue
            try:
                parsed = parse_occ(symbol)
            except OCCParseError:
                continue  # equity row, skip
            if parsed.option_type != "C":
                continue  # short puts and long puts don't reduce CC coverage
            units = float(pos.get("units") or 0)
            if units >= 0:
                continue  # only SHORT calls reduce coverage
            coverage[parsed.ticker] = coverage.get(parsed.ticker, 0) + int(-units)
    return coverage


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
