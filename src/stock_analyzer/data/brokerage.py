"""SnapTrade brokerage integration — fetch holdings across connected accounts."""

from __future__ import annotations

import os
from typing import Any, Literal

from snaptrade_client import SnapTrade
from snaptrade_client.auth import SnapTradeAuth

from ..logging import get_logger
from ..models.market import OCCParseError, ParsedOCC
from .options_symbols import is_option_symbol, parse_occ

logger = get_logger(__name__)


TaxStatus = Literal["taxable", "tax_advantaged"]


# Substring patterns that flag a name as a tax-advantaged account when
# SnapTrade's account `type` field isn't conclusive. Order matters — match
# the longest first to avoid e.g. "RothIRA" matching "IRA" then losing
# the Roth signal. Case-insensitive.
_TAX_ADVANTAGED_NAME_PATTERNS = (
    "ROTH IRA",
    "TRAD IRA",
    "TRADITIONAL IRA",
    "ROLLOVER IRA",
    "SEP IRA",
    "SIMPLE IRA",
    "ROTH",
    "IRA",
    "HSA",
    "401K",
    "401(K)",
    "403B",
    "457",
    "PENSION",
    "RRSP",
    "TFSA",
)
# SnapTrade-reported `type` values that map to tax-advantaged.
_TAX_ADVANTAGED_TYPES = {
    "IRA",
    "ROTH IRA",
    "TRADITIONAL IRA",
    "ROLLOVER IRA",
    "SEP IRA",
    "SIMPLE IRA",
    "401K",
    "401(K)",
    "403B",
    "457",
    "HSA",
    "RRSP",
    "TFSA",
    "RETIREMENT",
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


def classify_tax_status(account_type: str | None, account_name: str | None) -> TaxStatus:
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
        raise RuntimeError("SNAPTRADE_CLIENT_ID and SNAPTRADE_CONSUMER_KEY must be set")
    # SDK v13 moved credentials off the constructor onto an `auth=` object —
    # passing client_id=/consumer_key= directly now raises TypeError.
    return SnapTrade(
        auth=SnapTradeAuth.commercial_api_key(client_id=client_id, consumer_key=consumer_key)
    )


def _credentials() -> tuple[str, str]:
    user_id = os.getenv("SNAPTRADE_USER_ID")
    user_secret = os.getenv("SNAPTRADE_USER_SECRET")
    if not (user_id and user_secret):
        raise RuntimeError("SNAPTRADE_USER_ID and SNAPTRADE_USER_SECRET must be set")
    return user_id, user_secret


def _unwrap(resp: Any) -> Any:
    return resp.body if hasattr(resp, "body") else resp


def _to_float(v: Any) -> float | None:
    """SDK v13 returns position units/price/cost_basis as strings."""
    if v is None:
        return None
    try:
        return float(v)
    except TypeError, ValueError:
        return None


def _positions_from_response(resp: Any) -> list[dict]:
    """SDK v13's `get_all_account_positions` wraps positions in
    `{"results": [...], "data_freshness": {...}}` rather than returning the
    list directly (the shape `get_user_account_positions` used to return,
    pre-v13). Handle both so this survives another SDK reshuffle the same
    way `transactions.py`'s activities pagination already does."""
    body = _unwrap(resp)
    if isinstance(body, dict):
        return body.get("results") or []
    if isinstance(body, list):
        return body
    return []


def _extract_ticker(position: dict) -> str | None:
    """Find the underlying ticker symbol in a SnapTrade position payload.

    SDK v13's `AccountPosition` nests the symbol under `instrument.symbol`
    (every instrument kind — stock, option, crypto, etc. — carries its own
    `symbol` field directly, no further nesting). Older shapes (pre-v13,
    or `get_user_holdings`'s still-legacy response) put `symbol` directly
    on the position, sometimes as a nested dict — walked for compatibility.
    """
    instrument = position.get("instrument")
    if isinstance(instrument, dict) and isinstance(instrument.get("symbol"), str):
        return instrument["symbol"]
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
        accounts = (
            _unwrap(
                client.account_information.list_user_accounts(
                    user_id=user_id, user_secret=user_secret
                )
            )
            or []
        )
    except Exception as e:
        logger.warning("Could not list accounts for tax-status meta: %s", e)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for account in accounts:
        account_id = account.get("id")
        account_name = (
            account.get("name") or account.get("institution_name") or account_id or "unknown"
        )
        if not account_id:
            continue
        # SDK v13 dropped `type`/`account_type` from the Account model in
        # favor of `raw_type` (closest equivalent); `type`/`account_type`
        # are kept first for backward compatibility with any cached/legacy
        # response shape, `raw_type` is the field real v13 responses use.
        account_type = (
            account.get("type")
            or account.get("account_type")
            or account.get("raw_type")
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
        len(out) - n_advantaged,
        n_advantaged,
        len(out),
    )
    return out


def fetch_portfolio_holdings() -> dict[str, list[dict]]:
    """Return holdings keyed by account name across all connected SnapTrade accounts."""
    user_id, user_secret = _credentials()
    client = _client()

    logger.info("Fetching SnapTrade accounts")
    accounts = (
        _unwrap(
            client.account_information.list_user_accounts(user_id=user_id, user_secret=user_secret)
        )
        or []
    )
    logger.info("Found %d SnapTrade accounts", len(accounts))

    out: dict[str, list[dict]] = {}
    for account in accounts:
        account_id = account.get("id")
        account_name = (
            account.get("name") or account.get("institution_name") or account_id or "unknown"
        )
        if not account_id:
            continue

        positions = _positions_from_response(
            client.account_information.get_all_account_positions(
                user_id=user_id,
                user_secret=user_secret,
                account_id=account_id,
            )
        )

        holdings: list[dict] = []
        option_skip_count: int = 0
        for p in positions:
            ticker = _extract_ticker(p)
            if not ticker:
                continue
            if is_option_symbol(ticker):
                option_skip_count += 1
                continue
            # SDK v13 renamed average_purchase_price -> cost_basis and
            # returns units/price/cost_basis as strings, not numbers; cast
            # here so every downstream consumer keeps getting floats like
            # it always has, instead of hunting down every call site.
            holdings.append(
                {
                    "ticker": ticker,
                    "units": _to_float(p.get("units")),
                    "price": _to_float(p.get("price")),
                    "average_purchase_price": _to_float(
                        p.get("cost_basis", p.get("average_purchase_price"))
                    ),
                }
            )
        logger.info("Account %r: %d positions", account_name, len(holdings))
        if option_skip_count > 0:
            logger.info(
                "Skipped %d option position(s) from %s — handled separately by fetch_open_option_positions",
                option_skip_count,
                account_name,
            )
        out[account_name] = holdings

    return out


def _short_option_positions(option_type: str) -> list[tuple[str, ParsedOCC, int]]:
    """[(account_name, parsed OCC symbol, contracts short), ...] for every
    SHORT option of `option_type` ("C" or "P") across connected accounts.
    Long options are skipped. Returns [] when SnapTrade is unavailable."""
    try:
        user_id, user_secret = _credentials()
        client = _client()
    except Exception as e:
        logger.info("SnapTrade unavailable for option-position lookup: %s", e)
        return []

    try:
        accounts = (
            _unwrap(
                client.account_information.list_user_accounts(
                    user_id=user_id,
                    user_secret=user_secret,
                )
            )
            or []
        )
    except Exception as e:
        logger.info("SnapTrade list_user_accounts failed: %s", e)
        return []

    out: list[tuple[str, ParsedOCC, int]] = []
    for account in accounts:
        if isinstance(account, dict):
            account_id = account.get("id")
            account_name = account.get("name") or account.get("id") or "Unknown"
        else:
            account_id = getattr(account, "id", None)
            account_name = getattr(account, "name", None) or account_id or "Unknown"
        if not account_id:
            continue
        try:
            positions = _positions_from_response(
                client.account_information.get_all_account_positions(
                    user_id=user_id,
                    user_secret=user_secret,
                    account_id=account_id,
                )
            )
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
                continue
            if parsed.option_type != option_type:
                continue
            units = float(pos.get("units") or 0)
            if units >= 0:
                continue
            out.append((account_name, parsed, int(-units)))
    return out


def fetch_open_option_positions() -> dict[str, dict[str, int]]:
    """Return {underlying_ticker: {account_name: short_call_contracts}}.

    Per-account is required because each short call only collateralizes
    shares of the same underlying IN THE SAME ACCOUNT. A short call in
    Account A does NOT reduce CC capacity in Account B.

    Only SHORT calls (units < 0) are counted. Long calls and any puts are
    ignored. Returns {} when SnapTrade is unavailable or no positions are
    found.
    """
    coverage: dict[str, dict[str, int]] = {}
    for account_name, parsed, contracts in _short_option_positions("C"):
        per_account = coverage.setdefault(parsed.ticker, {})
        per_account[account_name] = per_account.get(account_name, 0) + contracts
    return coverage


def fetch_open_short_puts() -> dict[str, dict[str, float]]:
    """Return {underlying_ticker: {"contracts": n, "collateral_usd": x}}
    for puts already sold, summed across accounts.

    `collateral_usd` (strike × 100 × contracts) is cash the broker is
    already holding against possible assignment, so it isn't free for
    new cash-secured puts. Returns {} when SnapTrade is unavailable."""
    out: dict[str, dict[str, float]] = {}
    for _account, parsed, contracts in _short_option_positions("P"):
        rec = out.setdefault(parsed.ticker, {"contracts": 0, "collateral_usd": 0.0})
        rec["contracts"] += contracts
        rec["collateral_usd"] += parsed.strike * 100.0 * contracts
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
        accounts = (
            _unwrap(
                client.account_information.list_user_accounts(
                    user_id=user_id, user_secret=user_secret
                )
            )
            or []
        )
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
            except ValueError, TypeError:
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
