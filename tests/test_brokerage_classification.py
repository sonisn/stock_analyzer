"""Tax-status classification of brokerage accounts.

Real bug we hit during 5a: `\\b401(K)\\b` regex failed because `)` isn't
a word-boundary character, so "Vanguard 401(K)" wasn't detected as
tax-advantaged. The custom `_name_token_match` replaces `\\b`. These
tests pin that down + the broader detection contract.
"""
from __future__ import annotations

from stock_analyzer.data.brokerage import classify_tax_status

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
