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

from datetime import date

from ..logging import get_logger
from ..models.portfolio import EligibleHolding
from ..models.rebalance import OptionWrite, RebalanceAction, RebalancePlan

logger = get_logger(__name__)


def validate_option_writes(
    plan: RebalancePlan,
    *,
    eligibility: dict[str, list[EligibleHolding]],
    spots: dict[str, float] | None = None,
    delta_max: float | None = None,
    min_upside_pct: float | None = None,
    dte_min: int | None = None,
    dte_max: int | None = None,
    today: date | None = None,
) -> tuple[RebalancePlan, list[str]]:
    """Drop orphan WRITE_CALL actions, drop OptionWrites with unknown
    (ticker, account) pairs, clamp oversized contract counts against the
    matching account's max_contracts. Returns a new (frozen) plan with
    the same other fields untouched.

    The keep-the-shares rules are enforced here rather than trusted to
    the prompt: a call whose delta is above `delta_max`, whose strike
    caps the position less than `min_upside_pct` above spot, or whose
    expiry falls outside the DTE band is dropped. The put path already
    re-read every number from the chain; calls only checked eligibility,
    so a 0.60-delta write a month out would have passed.
    """
    warnings: list[str] = []
    spots = spots or {}
    today = today or date.today()

    # Index eligibility by (ticker, account) for O(1) lookup.
    index: dict[tuple[str, str], EligibleHolding] = {}
    for accounts in eligibility.values():
        for eh in accounts:
            index[(eh.ticker, eh.account)] = eh

    write_call_tickers = {a.ticker for a in plan.actions if a.action == "WRITE_CALL"}

    kept_tickers: set[str] = set()
    cleaned_option_writes: list[OptionWrite] = []
    seen_pairs: set[tuple[str, str]] = set()
    for ow in plan.option_writes:
        key = (ow.ticker, ow.account)
        match = index.get(key)
        if match is None:
            _warn(
                warnings,
                f"OptionWrite for {ow.ticker} in account {ow.account!r} "
                f"dropped: no matching eligibility entry",
            )
            continue
        reason = _rejection(
            ow,
            has_action=ow.ticker in write_call_tickers,
            duplicate=key in seen_pairs,
            spot=spots.get(ow.ticker.upper()),
            delta_max=delta_max,
            min_upside_pct=min_upside_pct,
            dte_min=dte_min,
            dte_max=dte_max,
            today=today,
        )
        if reason is not None:
            _warn(warnings, reason)
            continue
        contracts = ow.contracts
        if contracts > match.max_contracts:
            _warn(
                warnings,
                f"OptionWrite for {ow.ticker} in {ow.account!r} clamped "
                f"from {contracts} -> {match.max_contracts} contracts "
                f"(available_shares={match.available_shares})",
            )
            contracts = match.max_contracts
        if contracts <= 0:
            _warn(
                warnings,
                f"OptionWrite for {ow.ticker} in {ow.account!r} dropped: clamped contracts=0",
            )
            continue
        cleaned_option_writes.append(ow.model_copy(update={"contracts": contracts}))
        seen_pairs.add(key)
        kept_tickers.add(ow.ticker)

    cleaned_plan = plan.model_copy(
        update={
            "actions": _drop_orphan_calls(plan.actions, kept_tickers, warnings),
            "option_writes": cleaned_option_writes,
        }
    )
    return cleaned_plan, warnings


def _warn(warnings: list[str], message: str) -> None:
    warnings.append(message)
    logger.warning("CC validation: %s", message)


def _rejection(
    ow: OptionWrite,
    *,
    has_action: bool,
    duplicate: bool,
    spot: float | None,
    delta_max: float | None,
    min_upside_pct: float | None,
    dte_min: int | None,
    dte_max: int | None,
    today: date,
) -> str | None:
    """Why an eligible write must be dropped, or None to keep it (before
    the contract count is clamped)."""
    if not has_action:
        return f"OptionWrite for {ow.ticker} dropped: no matching WRITE_CALL action"
    if duplicate:
        return f"OptionWrite for {ow.ticker} in {ow.account!r} dropped: duplicate (only first kept)"
    if delta_max is not None and ow.delta > delta_max:
        return (
            f"OptionWrite for {ow.ticker} dropped: delta {ow.delta:.2f} above the "
            f"{delta_max:.2f} ceiling — roughly a {ow.delta:.0%} chance of losing "
            f"the shares"
        )
    if min_upside_pct is not None and spot:
        upside = (ow.strike / spot - 1) * 100
        if upside < min_upside_pct:
            return (
                f"OptionWrite for {ow.ticker} dropped: ${ow.strike:,.2f} strike is "
                f"{upside:+.1f}% from ${spot:,.2f}, inside the {min_upside_pct:.0f}% "
                f"floor — too close to give the position room"
            )
    if dte_min is not None or dte_max is not None:
        try:
            dte = (date.fromisoformat(ow.expiry) - today).days
        except ValueError:
            return f"OptionWrite for {ow.ticker} dropped: unreadable expiry"
        if (dte_min is not None and dte < dte_min) or (dte_max is not None and dte > dte_max):
            return (
                f"OptionWrite for {ow.ticker} dropped: {dte}d to expiry, outside the "
                f"{dte_min}-{dte_max}d band"
            )
    return None


def _drop_orphan_calls(
    actions: list[RebalanceAction], kept_tickers: set[str], warnings: list[str]
) -> list[RebalanceAction]:
    """WRITE_CALL actions whose write did not survive validation go too."""
    cleaned: list[RebalanceAction] = []
    for a in actions:
        if a.action == "WRITE_CALL" and a.ticker not in kept_tickers:
            _warn(
                warnings,
                f"WRITE_CALL action for {a.ticker} dropped: orphan "
                f"(no matching OptionWrite after validation)",
            )
            continue
        cleaned.append(a)
    return cleaned
