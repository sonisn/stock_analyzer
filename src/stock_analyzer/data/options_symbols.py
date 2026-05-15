"""OCC option-symbol parsing.

OCC format (21 chars total):

  ROOT(6, space-padded right) | YY(2) MM(2) DD(2) | TYPE(1: C|P) | STRIKE(8, 3 implied decimals)

Example:
  "NVDA  260620C00250000" = NVDA, 2026-06-20, Call, $250.000 strike

This module deliberately does NOT depend on third-party libs — the
format is fixed-width and small enough to handle by slicing.
"""
from __future__ import annotations

import re
from datetime import date

# Re-export model classes from the canonical models package. Local
# aliases preserve the legacy import path during Phase 1; Group C will
# delete this shim block once every callsite has migrated.
from ..models.market import OCCParseError, ParsedOCC

__all__ = ["OCCParseError", "ParsedOCC",
           "is_option_symbol", "parse_occ"]

_OCC_RE = re.compile(
    r"^([A-Z][A-Z0-9.\-]{0,5})\s+(\d{2})(\d{2})(\d{2})([CP])(\d{8})$"
)


def is_option_symbol(s: str) -> bool:
    """True if `s` looks like an OCC option symbol. Does not raise."""
    try:
        parse_occ(s)
    except OCCParseError:
        return False
    return True


def parse_occ(symbol: str) -> ParsedOCC:
    """Parse an OCC option symbol. Tolerates trailing whitespace and
    runs of spaces between the root and the date (SnapTrade and yfinance
    use slightly different padding). Raises OCCParseError on anything
    that doesn't fit the pattern.

    The fixed-width spec uses 6 chars for the root, space-padded right
    ("NVDA  "). We accept 1-6 chars plus one or more spaces between the
    root and date — brokers that fully strip the padding (no space
    delimiter at all) are not supported; they'd need to repad before
    calling this.
    """
    if not isinstance(symbol, str):
        raise OCCParseError(f"expected str, got {type(symbol).__name__}")
    s = symbol.strip()
    m = _OCC_RE.match(s)
    if not m:
        raise OCCParseError(f"not an OCC symbol: {symbol!r}")
    root, yy, mm, dd, otype, strike_raw = m.groups()
    try:
        expiry = date(2000 + int(yy), int(mm), int(dd))
    except ValueError as e:
        raise OCCParseError(f"bad date in {symbol!r}: {e}") from e
    # Strike has 3 implied decimals: "00250000" = 250.000
    strike = int(strike_raw) / 1000.0
    # Regex guarantees otype ∈ {"C","P"} but the type checker can only
    # see `str`; the ignore is for the Literal["C","P"] assignment.
    return ParsedOCC(
        ticker=root, expiry=expiry, option_type=otype, strike=strike,  # type: ignore[arg-type]
    )
