"""Tax-status classification of brokerage accounts.

Real bug we hit during 5a: `\\b401(K)\\b` regex failed because `)` isn't
a word-boundary character, so "Vanguard 401(K)" wasn't detected as
tax-advantaged. The custom `_name_token_match` replaces `\\b`. These
tests pin that down + the broader detection contract.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from stock_analyzer.data.brokerage import classify_tax_status, fetch_open_option_positions

# --- account `type` wins over name ----------------------------------------


def test_snaptrade_type_ira_is_tax_advantaged():
    """When SnapTrade returns type='IRA', name is irrelevant."""
    assert classify_tax_status("IRA", "My Random Name") == "tax_advantaged"
    assert classify_tax_status("ROTH IRA", "") == "tax_advantaged"
    assert classify_tax_status("HSA", None) == "tax_advantaged"


def test_plain_investment_type_falls_through_to_name_check():
    """Robinhood reports type='Investment' for everything — fall through
    to name match so 'Robinhood HSA' still classifies correctly."""
    assert classify_tax_status("Investment", "Schwab HSA") == "tax_advantaged"
    assert classify_tax_status("Investment", "Brokerage") == "taxable"


# --- name-based detection: the bug we hit ---------------------------------


def test_401k_with_parens_detected():
    """The regression test — `\\b401(K)\\b` failed because `)` isn't a
    word boundary. Custom token matcher must catch this."""
    assert classify_tax_status(None, "Vanguard 401(K)") == "tax_advantaged"
    assert classify_tax_status(None, "401(k)") == "tax_advantaged"


def test_hsa_token_boundary_not_substring():
    """'HSA' must match when standalone but NOT when embedded mid-word.
    A loose substring match would flag 'HSAFEcard' as tax-advantaged."""
    assert classify_tax_status(None, "Schwab HSA") == "tax_advantaged"
    assert classify_tax_status(None, "My HSA Plan") == "tax_advantaged"
    # Embedded inside another word: must NOT match.
    assert classify_tax_status(None, "HSAFEcard Trading") == "taxable"
    assert classify_tax_status(None, "BrokerHSA123") == "taxable"


def test_empty_inputs_default_to_taxable():
    """Safer default: when SnapTrade is silent, assume taxable. The
    docstring is explicit on this: skipping tax-cost analysis on a
    taxable account is the real risk; over-applying it to an IRA only
    wastes signal."""
    assert classify_tax_status(None, None) == "taxable"
    assert classify_tax_status("", "") == "taxable"
    assert classify_tax_status(None, "Brokerage") == "taxable"


# --- open short-call position parsing -----------


def test_fetch_open_option_positions_groups_short_calls_by_underlying():
    """3 short calls NVDA + 2 short calls AAPL + 1 LONG call TSLA + 1 equity.
    Output: only short calls counted; long calls and equity skipped."""
    fake_positions = [
        # 3 contracts short on NVDA Jun-260 call (units = -3)
        {"symbol": {"symbol": {"symbol": "NVDA  260620C00260000"}}, "units": -3},
        # 2 contracts short on AAPL Jul-230 call (units = -2)
        {"symbol": {"symbol": {"symbol": "AAPL  260718C00230000"}}, "units": -2},
        # 1 contract LONG on TSLA Aug-300 call (units = +1) — long, skip
        {"symbol": {"symbol": {"symbol": "TSLA  260815C00300000"}}, "units": 1},
        # Equity row — not an OCC symbol, skip
        {"symbol": {"symbol": {"symbol": "GOOG"}}, "units": 50},
    ]
    fake_accounts = [{"id": "acct-1", "name": "Test Acct"}]

    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = MagicMock(
        body=fake_accounts
    )
    fake_client.account_information.get_user_account_positions.return_value = MagicMock(
        body=fake_positions
    )

    with patch("stock_analyzer.data.brokerage._client", return_value=fake_client), \
         patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")):
        coverage = fetch_open_option_positions()

    assert coverage == {"NVDA": 3, "AAPL": 2}
    assert "TSLA" not in coverage
    assert "GOOG" not in coverage


def test_fetch_open_option_positions_returns_empty_on_credential_error():
    with patch(
        "stock_analyzer.data.brokerage._credentials",
        side_effect=RuntimeError("creds missing"),
    ):
        assert fetch_open_option_positions() == {}


def test_fetch_open_option_positions_returns_empty_when_no_accounts():
    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = MagicMock(body=[])
    with patch("stock_analyzer.data.brokerage._client", return_value=fake_client), \
         patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")):
        assert fetch_open_option_positions() == {}


def test_fetch_portfolio_holdings_skips_option_symbols():
    """Regression: option positions must NOT show up in equity holdings —
    they're handled separately by fetch_open_option_positions. Daily digest
    + discover + rebalance all consume fetch_portfolio_holdings and would
    waste API/LLM calls trying to look up OCC symbols."""
    from stock_analyzer.data.brokerage import fetch_portfolio_holdings

    fake_accounts = [{"id": "acct-1", "name": "Test Acct"}]
    fake_positions = [
        # Equity rows
        {"symbol": {"symbol": {"symbol": "NVDA"}},
         "units": 400, "average_purchase_price": 200.0},
        {"symbol": {"symbol": {"symbol": "AAPL"}},
         "units": 200, "average_purchase_price": 150.0},
        # Option rows — must be filtered out
        {"symbol": {"symbol": {"symbol": "NVDA  260620C00260000"}},
         "units": -3, "average_purchase_price": 2.40},
        {"symbol": {"symbol": {"symbol": "TSLA  260815C00300000"}},
         "units": 1, "average_purchase_price": 5.0},
    ]
    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = MagicMock(
        body=fake_accounts
    )
    fake_client.account_information.get_user_account_positions.return_value = MagicMock(
        body=fake_positions
    )
    with patch("stock_analyzer.data.brokerage._client", return_value=fake_client), \
         patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")):
        holdings = fetch_portfolio_holdings()

    # All accounts collapsed into one for assertion clarity:
    all_tickers = {h["ticker"] for acct in holdings.values() for h in acct}
    assert "NVDA" in all_tickers
    assert "AAPL" in all_tickers
    # Option symbols MUST NOT appear:
    assert not any(" " in t for t in all_tickers), f"Option symbol leaked: {all_tickers}"
