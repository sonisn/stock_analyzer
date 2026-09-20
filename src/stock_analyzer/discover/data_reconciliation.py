"""Flag disagreement between overlapping data sources before it reaches the LLM.

The discover pipeline pulls analyst price targets from two independent
sources — yfinance (`data/fundamentals.py::analyst_target_mean`) and
Finnhub (`data/finnhub.py::fetch_price_targets`) — and, until now, handed
both straight to the Analyst prompt with no cross-check. Garbage/
conflicting input produces a confidently wrong thesis no amount of prompt
engineering fixes; catching it at the data layer, before generation, is
cheaper than catching it after.
"""

from __future__ import annotations

# Sources routinely disagree by a few percent (different sample dates,
# different analyst panels) — that's noise, not a real conflict. Flag only
# when they diverge enough to plausibly point to a data error.
_DEFAULT_TOLERANCE_PCT = 20.0


def reconcile_price_targets(
    fundamentals: dict | None,
    finnhub_price_targets: dict | None,
    *,
    tolerance_pct: float = _DEFAULT_TOLERANCE_PCT,
) -> str | None:
    """Compare yfinance's and Finnhub's mean analyst price target.

    Returns a warning string when both sources have a value and they
    disagree by more than `tolerance_pct`; None when either source is
    missing (nothing to compare) or they agree within tolerance.
    """
    yf_target = (fundamentals or {}).get("analyst_target_mean")
    fh_target = (finnhub_price_targets or {}).get("mean")
    if yf_target is None or fh_target is None:
        return None
    if yf_target <= 0 or fh_target <= 0:
        return None

    diff_pct = abs(yf_target - fh_target) / min(yf_target, fh_target) * 100
    if diff_pct <= tolerance_pct:
        return None
    return (
        f"Analyst price target disagreement: yfinance mean=${yf_target:,.2f} vs "
        f"Finnhub mean=${fh_target:,.2f} ({diff_pct:.0f}% apart) — treat either "
        f"number with caution rather than picking one silently."
    )


# --- filed fundamentals vs yfinance -----------------------------------------------

# yfinance's margins are its own derivation and routinely sit a point or
# two from the filed GAAP figure; a real conflict is wider than that.
_MARGIN_TOLERANCE_PP = 5.0
_FLOW_TOLERANCE_PCT = 25.0
# Past this, the two sources cannot both be describing the same company —
# one of them has mis-extracted, and which one is not knowable from here.
_MARGIN_IMPLAUSIBLE_PP = 25.0
_FLOW_IMPLAUSIBLE_RATIO = 5.0

_MARGIN_FIELDS = ("gross_margin", "operating_margin", "profit_margin")
# Money amounts read as thousands; a ratio like debt/equity reads as 0.19,
# so "0 vs 0" is what a shared formatter would have printed for GOOGL.
_MONEY_FIELDS = ("free_cash_flow",)
_RATIO_FIELDS = ("debt_to_equity",)
# A US filer's debt/equity above this is not a capital structure, it is a
# unit mismatch: yfinance reports TSM's balance sheet in TWD, which gave
# debt/equity 42.16 and a 32% free-cash-flow yield.
_NON_USD_DEBT_TO_EQUITY = 20.0


def _label(field: str) -> str:
    return field.replace("_", " ")


def reconcile_fundamentals(
    yf_values: dict | None, filed_values: dict | None
) -> tuple[list[str], set[str]]:
    """Compare yfinance's fundamentals with the same figures as filed.

    Returns (warnings, fields to keep yfinance's value for). A moderate
    disagreement is reported and the filed number still wins — it is the
    one with an XBRL citation behind it. A disagreement too large for
    both to describe the same company means one source mis-extracted, and
    since which one is not knowable here, the filed value is *not* taken:
    ANET's 2025-12-31 `NetIncomeLoss` arrives as -$2,556M, which drags a
    38% net margin down to 5% if it is trusted silently.
    """
    warnings: list[str] = []
    rejected: set[str] = set()
    yf_values, filed_values = yf_values or {}, filed_values or {}

    for field in _MARGIN_FIELDS:
        a, b = yf_values.get(field), filed_values.get(field)
        if a is None or b is None:
            continue
        gap_pp = abs(a - b) * 100
        if gap_pp >= _MARGIN_IMPLAUSIBLE_PP or (a > 0) != (b > 0):
            rejected.add(field)
            warnings.append(
                f"{_label(field)} cannot be reconciled: {a * 100:.1f}% (yfinance) vs "
                f"{b * 100:.1f}% (as filed) — one source has mis-extracted it, so "
                f"yfinance's figure is kept and neither should be leaned on."
            )
        elif gap_pp > _MARGIN_TOLERANCE_PP:
            warnings.append(
                f"{_label(field)} {a * 100:.1f}% (yfinance) vs {b * 100:.1f}% (as filed) — "
                f"using the filed figure; the gap is usually GAAP vs adjusted."
            )

    for field in (*_MONEY_FIELDS, *_RATIO_FIELDS):
        a, b = yf_values.get(field), filed_values.get(field)
        if a is None or b is None or a == 0 or b == 0:
            continue
        fmt = "{:,.2f}" if field in _RATIO_FIELDS else "{:,.0f}"
        if (a > 0) != (b > 0):
            rejected.add(field)
            warnings.append(
                f"{_label(field)} has opposite signs: {fmt.format(a)} (yfinance) vs "
                f"{fmt.format(b)} (as filed) — yfinance's figure is kept; check the filing."
            )
            continue
        ratio = max(abs(a), abs(b)) / min(abs(a), abs(b))
        if ratio >= _FLOW_IMPLAUSIBLE_RATIO:
            rejected.add(field)
            warnings.append(
                f"{_label(field)} differs by {ratio:.1f}x: {fmt.format(a)} (yfinance) vs "
                f"{fmt.format(b)} (as filed) — too far apart to pick one, keeping yfinance's."
            )
        elif (ratio - 1) * 100 > _FLOW_TOLERANCE_PCT:
            warnings.append(
                f"{_label(field)} {fmt.format(a)} (yfinance) vs {fmt.format(b)} "
                f"(as filed) — using the filed figure."
            )
    return warnings, rejected


def flag_non_usd_fundamentals(ticker: str, yf_values: dict | None) -> str | None:
    """A foreign issuer's figures, reported in its own currency.

    Wisesheets covers SEC XBRL filers, so a foreign private issuer like
    TSM has no filed figures to cross-check against and its yfinance row
    goes unexamined — with a balance sheet in TWD. Scoring that against
    US peers compares a number to one 30x its size.
    """
    debt_to_equity = (yf_values or {}).get("debt_to_equity")
    if debt_to_equity is None or debt_to_equity < _NON_USD_DEBT_TO_EQUITY:
        return None
    return (
        f"{ticker} debt/equity reads {debt_to_equity:,.1f}, which no US filer sustains — "
        f"the figures are almost certainly in the company's own currency, not USD. "
        f"Compare its margins and growth, not its absolute amounts."
    )
