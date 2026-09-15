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
