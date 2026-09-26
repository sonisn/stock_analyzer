"""SnapTrade brokerage integration — fetch holdings across connected accounts."""

from __future__ import annotations

import os
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from snaptrade_client import SnapTrade
from snaptrade_client.auth import SnapTradeAuth

from ..logging import get_logger
from ..models.market import OCCParseError, ParsedOCC
from .options_symbols import is_option_symbol, parse_occ

logger = get_logger(__name__)


TaxStatus = Literal["taxable", "tax_advantaged"]
# Finer than TaxStatus, for deciding what belongs where: tax-deferred money
# (Traditional IRA, 401(k)) is taxed as income when withdrawn, tax-free
# money (Roth, HSA spent on medical costs) never is.
AccountKind = Literal["taxable", "tax_deferred", "tax_free"]
_TAX_FREE_PATTERNS = ("ROTH", "HSA", "TFSA")


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


def classify_account_kind(account_type: str | None, account_name: str | None) -> AccountKind:
    """taxable / tax_deferred / tax_free, on the same evidence as
    `classify_tax_status`: Roth, HSA and TFSA accounts grow tax-free, every
    other tax-advantaged account is tax-deferred."""
    if classify_tax_status(account_type, account_name) == "taxable":
        return "taxable"
    for text in (account_type, account_name):
        upper = (text or "").upper()
        if any(_name_token_match(upper, p) for p in _TAX_FREE_PATTERNS):
            return "tax_free"
    return "tax_deferred"


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


# A listed US ticker is letters, at most six of them, with an optional
# class suffix (BRK.B, BF-B). SnapTrade falls back to the CUSIP when a
# holding has no live listing — after the Schwab reconnect on 2026-09-20
# two dead positions arrived as "876214206" (Taronis Technologies, SEC
# registration revoked) and "87621P209" (Taronis Fuels, bankrupt), and a
# 401(k) commingled pool arrived as "FGCCPS" with kind "other". None of
# them exists on any market-data feed, so looking them up wastes a fetch
# and an LLM call and puts a symbol nobody can trade in the email.
_TICKER_RE = re.compile(r"^[A-Z]{1,6}([.-][A-Z]{1,2})?$")
# SnapTrade instrument kinds that trade on a market. Unknown kinds are
# allowed through — a new kind should not silently drop a real holding.
_UNLISTED_KINDS = {"other", "crypto"}


def is_listed_symbol(symbol: str | None, kind: str | None = None) -> bool:
    """Can market data be fetched for this holding's symbol?

    Positions that fail this are still held, valued and taxed like any
    other — they are only kept out of the data fetches and the LLM
    analysis, which have nothing to say about them.
    """
    if not symbol or not _TICKER_RE.match(str(symbol).strip().upper()):
        return False
    return str(kind or "").strip().lower() not in _UNLISTED_KINDS


def listed_tickers(holdings: dict[str, list[dict]]) -> tuple[list[str], list[str]]:
    """(tickers to analyze, symbols skipped) across every account."""
    keep, skip = set(), set()
    for items in holdings.values():
        for h in items:
            ticker = h.get("ticker")
            if not ticker:
                continue
            (keep if is_listed_symbol(ticker, h.get("kind")) else skip).add(str(ticker))
    return sorted(keep), sorted(skip)


def _field(account: Any, key: str) -> Any:
    return account.get(key) if isinstance(account, dict) else getattr(account, key, None)


def account_labels(accounts: list[Any]) -> dict[str, str]:
    """{account_id: label} — the one name every module keys accounts by.

    The account's name (else its institution, else its id). Two accounts
    with the same name (e.g. two brokers' "Individual") get the
    institution, or the id's last 4 characters, appended — otherwise the
    second silently overwrote the first's holdings, cash and options."""
    base: dict[str, str] = {}
    for a in accounts:
        account_id = _field(a, "id")
        if account_id:
            base[str(account_id)] = (
                _field(a, "name") or _field(a, "institution_name") or str(account_id)
            )
    counts: dict[str, int] = {}
    for label in base.values():
        counts[label] = counts.get(label, 0) + 1
    out: dict[str, str] = {}
    for a in accounts:
        account_id = _field(a, "id")
        if not account_id:
            continue
        label = base[str(account_id)]
        if counts[label] > 1:
            institution = _field(a, "institution_name")
            suffix = institution if institution and institution != label else str(account_id)[-4:]
            label = f"{label} ({suffix})"
            if label in out.values():
                label = f"{label} {str(account_id)[-4:]}"
        out[str(account_id)] = label
    return out


# How long a broker connection may go without a successful holdings sync
# before the reports call it out. A long weekend is normal; four days is
# not. On 2026-09-20 the Schwab HSA had last synced 2026-07-06 — 75 days
# — so its share counts, cash and prices were all frozen at July values
# while the other two accounts updated nightly.
STALE_SYNC_DAYS = 4


def _parse_sync_time(value: Any) -> datetime | None:
    """SnapTrade dates holdings syncs to the second and transaction syncs
    to the day; both arrive as strings. Everything is UTC."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=UTC)
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def account_sync_status(
    accounts: list[Any], *, now: datetime | None = None
) -> dict[str, dict[str, Any]]:
    """{label: {holdings_synced_at, transactions_synced_at, days_stale}}.

    `days_stale` is how long ago the broker last refreshed the holdings,
    or None when SnapTrade reports no sync time at all — an unknown sync
    is not evidence of a stale one, so it is never flagged.
    """
    now = now or datetime.now(UTC)
    labels = account_labels(accounts)
    out: dict[str, dict[str, Any]] = {}
    for account in accounts:
        account_id = _field(account, "id")
        if not account_id:
            continue
        sync = _field(account, "sync_status") or {}
        holdings_sync = _parse_sync_time(
            _field(_field(sync, "holdings") or {}, "last_successful_sync")
        )
        txn_sync = _parse_sync_time(
            _field(_field(sync, "transactions") or {}, "last_successful_sync")
        )
        out[labels[str(account_id)]] = {
            "holdings_synced_at": holdings_sync,
            "transactions_synced_at": txn_sync,
            "days_stale": (now - holdings_sync) / timedelta(days=1) if holdings_sync else None,
        }
    return out


def stale_account_notes(
    sync_status: dict[str, dict[str, Any]], *, max_days: float = STALE_SYNC_DAYS
) -> list[str]:
    """One line per account the broker has stopped refreshing.

    A dead connection is not a stale price — the share counts, the cash
    and the transaction history are frozen too, so anything the run
    reports for that account describes the day it stopped syncing.
    """
    notes = []
    for label, status in sorted(sync_status.items()):
        days = status.get("days_stale")
        if days is None or days < max_days:
            continue
        when = status["holdings_synced_at"].date().isoformat()
        notes.append(
            f"{label} last synced {when}, {days:.0f} days ago — its holdings, "
            "cash and prices are frozen; reconnect it in SnapTrade"
        )
    return notes


def fetch_account_sync_status() -> dict[str, dict[str, Any]]:
    """`account_sync_status` for the connected accounts. Never raises — a
    freshness check must not be what takes the daily report down."""
    try:
        user_id, user_secret = _credentials()
        accounts = (
            _unwrap(
                _client().account_information.list_user_accounts(
                    user_id=user_id, user_secret=user_secret
                )
            )
            or []
        )
    except Exception as e:
        logger.warning("Could not check how fresh the brokerage data is: %s", e)
        return {}
    status = account_sync_status(accounts)
    for note in stale_account_notes(status):
        logger.warning("%s", note)
    return status


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

    labels = account_labels(accounts)
    out: dict[str, dict[str, Any]] = {}
    for account in accounts:
        account_id = account.get("id")
        if not account_id:
            continue
        account_name = labels[str(account_id)]
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
            "kind": classify_account_kind(account_type, account_name),
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

    labels = account_labels(accounts)
    out: dict[str, list[dict]] = {}
    for account in accounts:
        account_id = account.get("id")
        if not account_id:
            continue
        account_name = labels[str(account_id)]

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
            instrument = p.get("instrument")
            holdings.append(
                {
                    "ticker": ticker,
                    "kind": (instrument or {}).get("kind")
                    if isinstance(instrument, dict)
                    else None,
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

    labels = account_labels(accounts)
    out: list[tuple[str, ParsedOCC, int]] = []
    for account in accounts:
        account_id = _field(account, "id")
        if not account_id:
            continue
        account_name = labels[str(account_id)]
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


def fetch_open_short_puts() -> dict[str, dict[str, Any]]:
    """Return {underlying_ticker: {"contracts": n, "collateral_usd": x,
    "by_account": {account: collateral_usd}}} for puts already sold.

    `collateral_usd` (strike × 100 × contracts) is cash the broker is
    already holding against possible assignment, so it isn't free for
    new cash-secured puts. Returns {} when SnapTrade is unavailable."""
    out: dict[str, dict[str, Any]] = {}
    for account, parsed, contracts in _short_option_positions("P"):
        rec = out.setdefault(
            parsed.ticker, {"contracts": 0, "collateral_usd": 0.0, "by_account": {}}
        )
        collateral = parsed.strike * 100.0 * contracts
        rec["contracts"] += contracts
        rec["collateral_usd"] += collateral
        rec["by_account"][account] = rec["by_account"].get(account, 0.0) + collateral
    return out


def fetch_covered_call_obligations() -> dict[str, dict[str, Any]]:
    """{ticker: what the short calls on it oblige you to}, with strikes.

    `fetch_open_option_positions` answers "how much capacity is used",
    which is what the rebalancer needs before writing more calls. This
    answers the question every *sell* decision has to ask: how many of
    these shares are already promised to someone else, at what price, and
    until when. Selling shares that back a short call turns it naked, so
    a sale means buying the call back or waiting for assignment.

    Returns {} when SnapTrade is unavailable.
    """
    out: dict[str, dict[str, Any]] = {}
    for account, parsed, contracts in _short_option_positions("C"):
        rec = out.setdefault(
            parsed.ticker,
            {"contracts": 0, "shares_committed": 0.0, "by_account": {}, "legs": []},
        )
        rec["contracts"] += contracts
        rec["shares_committed"] += contracts * 100.0
        rec["by_account"][account] = rec["by_account"].get(account, 0) + contracts
        rec["legs"].append(
            {
                "account": account,
                "contracts": contracts,
                "strike": parsed.strike,
                "expiry": parsed.expiry.isoformat(),
            }
        )
    for rec in out.values():
        # Soonest first: the leg that decides the position is the one
        # expiring next, not the largest.
        rec["legs"].sort(key=lambda leg: (leg["expiry"], leg["strike"]))
        rec["next_expiry"] = rec["legs"][0]["expiry"] if rec["legs"] else None
        rec["lowest_strike"] = min((leg["strike"] for leg in rec["legs"]), default=None)
    return out


def _usd_cash(balances: Any) -> float | None:
    """USD cash from SnapTrade's per-currency balance list. Non-USD
    entries are skipped (no FX conversion) rather than summed as dollars."""
    if isinstance(balances, dict):
        balances = [balances]
    total, found = 0.0, False
    for b in balances or []:
        if not isinstance(b, dict) or b.get("cash") is None:
            continue
        currency = b.get("currency")
        code = currency.get("code") if isinstance(currency, dict) else currency
        if code and str(code).upper() != "USD":
            logger.warning("Skipping %s cash balance (no FX conversion)", code)
            continue
        try:
            total += float(b["cash"])
            found = True
        except ValueError, TypeError:
            continue
    return total if found else None


def fetch_account_cash() -> dict[str, float]:
    """{account label: USD cash} for every connected account with a
    readable balance. Cash only buys (or secures puts) in its own account,
    so plans are checked per account, not against one pooled total."""
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
        return {}

    labels = account_labels(accounts)
    out: dict[str, float] = {}
    for account in accounts:
        account_id = _field(account, "id")
        if not account_id:
            continue
        try:
            balances = _unwrap(
                client.account_information.get_user_account_balance(
                    user_id=user_id,
                    user_secret=user_secret,
                    account_id=account_id,
                )
            )
        except Exception as e:
            logger.warning("Balance fetch failed for account %s: %s", account_id, e)
            continue
        cash = _usd_cash(balances)
        if cash is not None:
            out[labels[str(account_id)]] = cash
    return out


def fetch_total_cash() -> float | None:
    """Sum of USD cash across all connected accounts (None when no balance
    could be read). Prefer `fetch_account_cash` where the account matters."""
    per_account = fetch_account_cash()
    return sum(per_account.values()) if per_account else None


def fetch_portfolio_tickers() -> list[str]:
    """De-duplicated, sorted tickers across all connected accounts, minus
    the ones no market data exists for (see `is_listed_symbol`)."""
    tickers, skipped = listed_tickers(fetch_portfolio_holdings())
    if skipped:
        logger.info("Not market-listed, skipping data for: %s", ", ".join(skipped))
    return tickers
