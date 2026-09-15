"""Tax-status classification of brokerage accounts, and SnapTrade SDK v13
response-shape adaptation (position envelope, `instrument.symbol` nesting,
`cost_basis` rename, `raw_type` account field).

Real bug we hit during 5a: `\\b401(K)\\b` regex failed because `)` isn't
a word-boundary character, so "Vanguard 401(K)" wasn't detected as
tax-advantaged. The custom `_name_token_match` replaces `\\b`. These
tests pin that down + the broader detection contract.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stock_analyzer.data import brokerage
from stock_analyzer.data.brokerage import (
    _extract_ticker,
    _positions_from_response,
    _to_float,
    classify_tax_status,
    fetch_account_meta,
    fetch_open_option_positions,
)

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


# --- SDK v13 response-shape adaptation --------------------------------


def test_positions_from_response_unwraps_results_envelope():
    """v13's get_all_account_positions wraps positions in
    {"results": [...], "data_freshness": {...}} instead of returning the
    list directly."""
    resp = MagicMock(body={"results": [{"units": "1"}], "data_freshness": {}})
    assert _positions_from_response(resp) == [{"units": "1"}]


def test_positions_from_response_still_handles_a_bare_list():
    """Backward-compat: a bare list body (pre-v13 shape, or a mocked
    response that skips the envelope) still works."""
    resp = MagicMock(body=[{"units": "1"}])
    assert _positions_from_response(resp) == [{"units": "1"}]


def test_positions_from_response_handles_missing_results_key():
    resp = MagicMock(body={"data_freshness": {}})
    assert _positions_from_response(resp) == []


def test_extract_ticker_reads_v13_instrument_symbol():
    """v13's AccountPosition nests the symbol under instrument.symbol —
    every instrument kind (stock, option, ...) carries it directly."""
    assert _extract_ticker({"instrument": {"kind": "stock", "symbol": "NVDA"}}) == "NVDA"
    assert (
        _extract_ticker({"instrument": {"kind": "option", "symbol": "NVDA  260620C00260000"}})
        == "NVDA  260620C00260000"
    )


def test_extract_ticker_falls_back_to_legacy_nested_symbol_shape():
    """Pre-v13 (or get_user_holdings, which kept the old Position type)
    nested the symbol as {"symbol": {"symbol": {"symbol": "..."}}}."""
    assert _extract_ticker({"symbol": {"symbol": {"symbol": "GOOG"}}}) == "GOOG"
    assert _extract_ticker({"symbol": "AAPL"}) == "AAPL"
    assert _extract_ticker({}) is None


def test_to_float_coerces_v13_string_fields():
    """v13 returns units/price/cost_basis as strings, not numbers."""
    assert _to_float("123.45") == 123.45
    assert _to_float(None) is None
    assert _to_float("not-a-number") is None


def test_fetch_account_meta_falls_back_to_raw_type():
    """SDK v13 dropped `type`/`account_type` from the Account model in
    favor of `raw_type` — the primary tax-status signal must keep working
    once real responses stop sending the old fields."""
    fake_accounts = [{"id": "acct-1", "name": "Vanguard IRA", "raw_type": "IRA"}]
    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = MagicMock(body=fake_accounts)
    with (
        patch("stock_analyzer.data.brokerage._client", return_value=fake_client),
        patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")),
    ):
        meta = fetch_account_meta()
    assert meta["Vanguard IRA"]["type"] == "IRA"
    assert meta["Vanguard IRA"]["tax_status"] == "tax_advantaged"


# --- open short-call position parsing -----------


def test_fetch_open_option_positions_groups_short_calls_by_underlying():
    """3 short calls NVDA + 2 short calls AAPL + 1 LONG call TSLA + 1 equity.
    Output: only short calls counted; long calls and equity skipped."""
    fake_positions = [
        # 3 contracts short on NVDA Jun-260 call (units = -3)
        {"instrument": {"kind": "option", "symbol": "NVDA  260620C00260000"}, "units": "-3"},
        # 2 contracts short on AAPL Jul-230 call (units = -2)
        {"instrument": {"kind": "option", "symbol": "AAPL  260718C00230000"}, "units": "-2"},
        # 1 contract LONG on TSLA Aug-300 call (units = +1) — long, skip
        {"instrument": {"kind": "option", "symbol": "TSLA  260815C00300000"}, "units": "1"},
        # Equity row — not an OCC symbol, skip
        {"instrument": {"kind": "stock", "symbol": "GOOG"}, "units": "50"},
    ]
    fake_accounts = [{"id": "acct-1", "name": "Test Acct"}]

    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = MagicMock(body=fake_accounts)
    fake_client.account_information.get_all_account_positions.return_value = MagicMock(
        body={"results": fake_positions, "data_freshness": {}}
    )

    with (
        patch("stock_analyzer.data.brokerage._client", return_value=fake_client),
        patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")),
    ):
        coverage = fetch_open_option_positions()

    assert coverage == {
        "NVDA": {"Test Acct": 3},
        "AAPL": {"Test Acct": 2},
    }
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
    with (
        patch("stock_analyzer.data.brokerage._client", return_value=fake_client),
        patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")),
    ):
        assert fetch_open_option_positions() == {}


def test_fetch_open_option_positions_returns_per_account_shape(monkeypatch):
    """Per-account map: ticker → {account_name: contracts}.

    A short call in Fidelity IRA must NOT reduce Fidelity Taxable's CC
    capacity. The per-account shape is what makes that correct downstream.

    Uses the legacy flat `symbol` shape (not `instrument.symbol`) deliberately
    — `_extract_ticker`'s fallback path exists precisely for responses that
    don't carry the v13 `instrument` nesting (e.g. get_user_holdings).
    """
    monkeypatch.setattr(brokerage, "_credentials", lambda: ("uid", "secret"))

    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = [
        {"id": "acct-ira", "name": "Fidelity IRA"},
        {"id": "acct-tax", "name": "Fidelity Taxable"},
    ]

    def _positions(*, user_id, user_secret, account_id):
        # Bare list (no .body/.results envelope) — _positions_from_response
        # must handle this shape too.
        if account_id == "acct-ira":
            return [{"symbol": "NVDA  260620C00260000", "units": -1}]
        return [{"symbol": "NVDA  260620C00260000", "units": -2}]

    fake_client.account_information.get_all_account_positions.side_effect = lambda **kw: _positions(
        **kw
    )
    monkeypatch.setattr(brokerage, "_client", lambda: fake_client)

    out = brokerage.fetch_open_option_positions()
    assert out == {
        "NVDA": {"Fidelity IRA": 1, "Fidelity Taxable": 2},
    }


def test_fetch_open_option_positions_empty_when_unavailable(monkeypatch):
    """When credentials are missing, return {}."""

    def _raise():
        raise RuntimeError("no creds")

    monkeypatch.setattr(brokerage, "_credentials", _raise)
    assert brokerage.fetch_open_option_positions() == {}


def test_fetch_portfolio_holdings_skips_option_symbols():
    """Regression: option positions must NOT show up in equity holdings —
    they're handled separately by fetch_open_option_positions. Daily digest
    + discover + rebalance all consume fetch_portfolio_holdings and would
    waste API/LLM calls trying to look up OCC symbols."""
    from stock_analyzer.data.brokerage import fetch_portfolio_holdings

    fake_accounts = [{"id": "acct-1", "name": "Test Acct"}]
    fake_positions = [
        # Equity rows
        {"instrument": {"kind": "stock", "symbol": "NVDA"}, "units": "400", "cost_basis": "200.0"},
        {"instrument": {"kind": "stock", "symbol": "AAPL"}, "units": "200", "cost_basis": "150.0"},
        # Option rows — must be filtered out
        {
            "instrument": {"kind": "option", "symbol": "NVDA  260620C00260000"},
            "units": "-3",
            "cost_basis": "2.40",
        },
        {
            "instrument": {"kind": "option", "symbol": "TSLA  260815C00300000"},
            "units": "1",
            "cost_basis": "5.0",
        },
    ]
    fake_client = MagicMock()
    fake_client.account_information.list_user_accounts.return_value = MagicMock(body=fake_accounts)
    fake_client.account_information.get_all_account_positions.return_value = MagicMock(
        body={"results": fake_positions, "data_freshness": {}}
    )
    with (
        patch("stock_analyzer.data.brokerage._client", return_value=fake_client),
        patch("stock_analyzer.data.brokerage._credentials", return_value=("u", "s")),
    ):
        holdings = fetch_portfolio_holdings()

    # All accounts collapsed into one for assertion clarity:
    all_holdings = [h for acct in holdings.values() for h in acct]
    all_tickers = {h["ticker"] for h in all_holdings}
    assert "NVDA" in all_tickers
    assert "AAPL" in all_tickers
    # Option symbols MUST NOT appear:
    assert not any(" " in t for t in all_tickers), f"Option symbol leaked: {all_tickers}"
    # cost_basis (v13) round-trips through as a float under the legacy
    # "average_purchase_price" key, and units/price are floats not strings.
    nvda = next(h for h in all_holdings if h["ticker"] == "NVDA")
    assert nvda["average_purchase_price"] == 200.0
    assert nvda["units"] == 400.0
