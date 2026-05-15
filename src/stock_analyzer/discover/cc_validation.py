"""Post-LLM WRITE_CALL validation.

Runs after Rebalancer.decide() and BEFORE the plan is persisted or
rendered. Guarantees that:

  - every WRITE_CALL action has a matching OptionWrite entry (drops
    orphan actions and orphan option_writes)
  - every OptionWrite ticker is in the eligibility map (drops unknown)
  - contracts × 100 <= available_shares (clamps to max_contracts)

Returns a cleaned plan plus a list of human-readable warning strings,
which the caller logs (loudly) and surfaces in the email summary.
"""
from __future__ import annotations

from ..logging import get_logger
from ..models.portfolio import EligibleHolding
from ..models.rebalance import OptionWrite, RebalanceAction, RebalancePlan

logger = get_logger(__name__)


def validate_option_writes(
    plan: RebalancePlan,
    *,
    eligibility: dict[str, EligibleHolding],
) -> tuple[RebalancePlan, list[str]]:
    """Drop orphan WRITE_CALL actions, drop orphan option_writes, clamp
    oversized `contracts`, drop unknown tickers. Returns a new
    (frozen) plan with the same other fields untouched.
    """
    warnings: list[str] = []

    write_call_tickers = {
        a.ticker for a in plan.actions if a.action == "WRITE_CALL"
    }
    option_write_tickers = {ow.ticker for ow in plan.option_writes}

    kept: set[str] = set()
    cleaned_option_writes: list[OptionWrite] = []
    for ow in plan.option_writes:
        if ow.ticker not in eligibility:
            warnings.append(
                f"OptionWrite for {ow.ticker} dropped: ticker not eligible"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        if ow.ticker not in write_call_tickers:
            warnings.append(
                f"OptionWrite for {ow.ticker} dropped: no matching WRITE_CALL action"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        elig = eligibility[ow.ticker]
        contracts = ow.contracts
        if contracts > elig.max_contracts:
            warnings.append(
                f"OptionWrite for {ow.ticker} clamped from "
                f"{contracts} -> {elig.max_contracts} contracts "
                f"(available_shares={elig.available_shares})"
            )
            logger.warning("CC validation: %s", warnings[-1])
            contracts = elig.max_contracts
        if contracts <= 0:
            warnings.append(
                f"OptionWrite for {ow.ticker} dropped: clamped contracts=0"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        cleaned_option_writes.append(
            ow.model_copy(update={"contracts": contracts})
        )
        kept.add(ow.ticker)

    cleaned_actions: list[RebalanceAction] = []
    for a in plan.actions:
        if a.action == "WRITE_CALL" and a.ticker not in kept:
            warnings.append(
                f"WRITE_CALL action for {a.ticker} dropped: orphan "
                f"(no matching OptionWrite after validation)"
            )
            logger.warning("CC validation: %s", warnings[-1])
            continue
        cleaned_actions.append(a)

    for orphan in write_call_tickers - option_write_tickers:
        warnings.append(
            f"WRITE_CALL action for {orphan} had NO OptionWrite in the plan"
        )
        logger.warning("CC validation: %s", warnings[-1])

    cleaned_plan = plan.model_copy(update={
        "actions": cleaned_actions,
        "option_writes": cleaned_option_writes,
    })
    return cleaned_plan, warnings
