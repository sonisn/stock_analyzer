"""Deterministic backfill of OptionWrite entries for orphan WRITE_CALL
actions.

The Opus rebalancer sometimes emits WRITE_CALL actions (with all the
detail in `full_text` prose) without populating the parallel
`option_writes` structured list. When that happens, the validation step
drops the WRITE_CALLs as orphans and the email loses the structured
sections.

This module recovers those structured fields by:
  1. Parsing the action's `sizing` string for ticker / strike / expiry.
  2. Looking up the matching call in the chain data already in state.
  3. Constructing the OptionWrite from bid/ask/delta/iv on the chain row.

Runs BEFORE validation — so validation finds matched pairs and keeps
the WRITE_CALLs.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

from ..logging import get_logger
from ..models.market import OptionChain
from ..models.rebalance import OptionWrite, RebalancePlan

logger = get_logger(__name__)

# Matches sizing strings like:
#   "1 contract $450C expiring 2026-06-18"
#   "3 contracts $260C 2026-06-20"
#   "2 contracts $230.00C 2026-06-20"
#   "5 contracts $1,250C expiring 2026-07-18"
_SIZING_RE = re.compile(
    r"(\d+)\s+contracts?\s+\$?([\d,]+(?:\.\d+)?)C\s+(?:expir\w+\s+)?(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def _parse_sizing(sizing: str) -> tuple[int, float, str] | None:
    """Return (contracts, strike, expiry_iso) or None if not parseable."""
    if not isinstance(sizing, str):
        return None
    m = _SIZING_RE.search(sizing)
    if not m:
        return None
    try:
        contracts = int(m.group(1))
        strike = float(m.group(2).replace(",", ""))
        expiry = m.group(3)
        # Sanity check the date.
        date.fromisoformat(expiry)
        if contracts <= 0 or strike <= 0:
            return None
        return contracts, strike, expiry
    except (ValueError, TypeError):
        return None


def _match_chain_row(
    chain: OptionChain, strike: float, expiry_iso: str,
) -> dict[str, Any] | None:
    """Find the chain row matching strike + expiry. Tolerate small
    floating-point error on strike. Returns a dict of the relevant
    fields for OptionWrite construction, or None."""
    target_expiry = date.fromisoformat(expiry_iso)
    for q in chain.calls:
        if q.expiry != target_expiry:
            continue
        # 1-cent tolerance for strike matching (broker rounding).
        if abs(q.strike - strike) > 0.01:
            continue
        bid = q.bid or 0.0
        ask = q.ask or 0.0
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)
        if mid <= 0 or math.isnan(mid):
            mid = 0.0
        delta = q.delta if q.delta is not None else 0.0
        return {
            "strike": q.strike,
            "expiry": expiry_iso,
            "est_premium_per_share": float(mid),
            "delta": float(delta),
            "iv": q.iv,
        }
    return None


def backfill_option_writes(
    plan: RebalancePlan,
    *,
    chains: dict[str, OptionChain],
) -> RebalancePlan:
    """For every WRITE_CALL action lacking an OptionWrite, synthesize one
    from the action's sizing string + chain data. Returns a new plan
    (frozen models — uses model_copy).

    Best-effort: actions whose sizing is unparseable, whose ticker has
    no chain, or whose strike/expiry has no match in the chain are left
    alone (validation will drop them with a warning, as before).
    """
    existing_ow_tickers = {ow.ticker for ow in plan.option_writes}
    new_writes: list[OptionWrite] = list(plan.option_writes)

    for action in plan.actions:
        if action.action != "WRITE_CALL":
            continue
        if action.ticker in existing_ow_tickers:
            continue

        parsed = _parse_sizing(action.sizing)
        if parsed is None:
            logger.warning(
                "CC backfill: could not parse sizing %r for %s — "
                "OptionWrite will not be synthesized; validation will drop "
                "this WRITE_CALL.",
                action.sizing, action.ticker,
            )
            continue

        contracts, strike, expiry_iso = parsed
        chain = chains.get(action.ticker)
        if chain is None or not chain.calls:
            logger.warning(
                "CC backfill: no chain data for %s — cannot synthesize "
                "OptionWrite (action sizing was parseable).",
                action.ticker,
            )
            continue

        match = _match_chain_row(chain, strike, expiry_iso)
        if match is None:
            logger.warning(
                "CC backfill: no chain row matching %s $%.2fC %s — "
                "Opus may have hallucinated a strike. OptionWrite will "
                "not be synthesized.",
                action.ticker, strike, expiry_iso,
            )
            continue

        delta_val = match["delta"]
        # delta in [0,1] per OptionWrite constraint; clamp defensively.
        delta_val = max(0.0, min(1.0, delta_val))
        new_writes.append(OptionWrite(
            ticker=action.ticker,
            strike=match["strike"],
            expiry=match["expiry"],
            contracts=contracts,
            est_premium_per_share=match["est_premium_per_share"],
            delta=delta_val,
            assignment_probability=delta_val,
            notes="backfilled from chain after Opus omitted option_writes",
        ))
        existing_ow_tickers.add(action.ticker)
        logger.info(
            "CC backfill: synthesized OptionWrite for %s (%d × $%.2fC %s, "
            "premium $%.2f/share, Δ %.2f) from chain data.",
            action.ticker, contracts, match["strike"], expiry_iso,
            match["est_premium_per_share"], delta_val,
        )

    if len(new_writes) == len(plan.option_writes):
        return plan
    return plan.model_copy(update={"option_writes": new_writes})
