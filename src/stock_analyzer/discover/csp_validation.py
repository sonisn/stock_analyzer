"""Post-LLM SELL_PUT backfill and validation.

Runs after Rebalancer.decide() and before the plan is persisted or
rendered — the same compute-then-force-correct shape as
cc_backfill.py + cc_validation.py.

`backfill_csp_writes` rebuilds a missing CashSecuredPut from the
action's sizing string and the chain. `validate_csp_writes` then keeps
only puts that:
  - are on an eligible ticker, with exactly one matching SELL_PUT action
  - name a strike/expiry that exists in the fetched chain (no invented
    strikes); delta and premium are re-read from that chain row
  - are out of the money, inside the DTE band and inside the |Δ| band
  - fit the per-put and total collateral caps (contracts are clamped
    down to fit; a put that can't fit even one contract is dropped)

Surviving SELL_PUT actions get a canonical sizing string, so the action
table always agrees with the structured entry.
"""

from __future__ import annotations

import math
import re
from datetime import date

from ..logging import get_logger
from ..models.market import OptionChain, OptionQuote
from ..models.portfolio import CspCandidate
from ..models.rebalance import CashSecuredPut, RebalanceAction, RebalancePlan

logger = get_logger(__name__)

# Rounding slack on the |Δ| band — quotes move between fetch and answer.
_DELTA_TOLERANCE = 0.01

# Matches sizing strings like:
#   "2 contracts $145P 2026-07-18"
#   "1 contract $1,250.00P expiring 2026-07-18"
#   "2 contracts at $145 strike, exp 2026-07-18"
_SIZING_RE = re.compile(
    r"(?P<contracts>\d+)\s+contracts?\s+(?:at\s+)?\$?(?P<strike>[\d,]+(?:\.\d+)?)"
    r"\s*(?:P\b|strike\b)[,\s]*(?:exp\w*\s+)?(?P<expiry>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def csp_sizing(cp: CashSecuredPut) -> str:
    """Canonical SELL_PUT sizing, e.g. '2 contracts $145P 2026-07-18'."""
    plural = "s" if cp.contracts != 1 else ""
    return f"{cp.contracts} contract{plural} ${cp.strike:,.2f}P {cp.expiry}"


def _parse_sizing(sizing: str) -> tuple[int, float, str] | None:
    m = _SIZING_RE.search(sizing or "")
    if not m:
        return None
    try:
        contracts = int(m.group("contracts"))
        strike = float(m.group("strike").replace(",", ""))
        date.fromisoformat(m.group("expiry"))
    except ValueError:
        return None
    if contracts <= 0 or strike <= 0:
        return None
    return contracts, strike, m.group("expiry")


def _find_put(chain: OptionChain | None, strike: float, expiry: str) -> OptionQuote | None:
    if chain is None:
        return None
    try:
        target = date.fromisoformat(expiry)
    except ValueError:
        return None
    for q in chain.puts:
        if q.expiry == target and abs(q.strike - strike) <= 0.01:
            return q
    return None


def _mid(q: OptionQuote) -> float:
    bid, ask = q.bid or 0.0, q.ask or 0.0
    mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else max(bid, ask)
    return 0.0 if math.isnan(mid) else float(mid)


def backfill_csp_writes(plan: RebalancePlan, *, chains: dict[str, OptionChain]) -> RebalancePlan:
    """For every SELL_PUT action without a CashSecuredPut, build one from
    the sizing string + chain row. Best-effort: unparseable sizings and
    strikes missing from the chain are left for validation to drop."""
    have = {cp.ticker for cp in plan.csp_writes}
    added: list[CashSecuredPut] = []
    for action in plan.actions:
        if action.action != "SELL_PUT" or action.ticker in have:
            continue
        parsed = _parse_sizing(action.sizing)
        if parsed is None:
            logger.warning(
                "CSP backfill: could not parse sizing %r for %s", action.sizing, action.ticker
            )
            continue
        contracts, strike, expiry = parsed
        q = _find_put(chains.get(action.ticker), strike, expiry)
        if q is None or q.delta is None:
            logger.warning(
                "CSP backfill: no chain row for %s $%.2fP %s", action.ticker, strike, expiry
            )
            continue
        added.append(
            CashSecuredPut(
                ticker=action.ticker,
                strike=q.strike,
                expiry=expiry,
                contracts=contracts,
                est_premium_per_share=_mid(q),
                delta=q.delta,
                notes="backfilled from chain after the plan omitted csp_writes",
            )
        )
        have.add(action.ticker)
    if not added:
        return plan
    return plan.model_copy(update={"csp_writes": [*plan.csp_writes, *added]})


def validate_csp_writes(
    plan: RebalancePlan,
    *,
    eligible: dict[str, CspCandidate],
    chains: dict[str, OptionChain],
    cash_budget: float,
    delta_min: float,
    delta_max: float,
    dte_min: int,
    dte_max: int,
    max_pct_total: float,
    today: date | None = None,
) -> tuple[RebalancePlan, list[str]]:
    """Return (cleaned plan, warnings). See the module docstring."""
    today = today or date.today()
    warnings: list[str] = []

    def _warn(msg: str) -> None:
        warnings.append(msg)
        logger.warning("CSP validation: %s", msg)

    action_counts: dict[str, int] = {}
    for a in plan.actions:
        if a.action == "SELL_PUT":
            action_counts[a.ticker] = action_counts.get(a.ticker, 0) + 1

    total_room = cash_budget * max_pct_total
    kept: dict[str, CashSecuredPut] = {}
    for cp in plan.csp_writes:
        t = cp.ticker
        cand = eligible.get(t)
        if cand is None:
            _warn(f"put on {t} dropped: not a put candidate this run")
            continue
        if action_counts.get(t, 0) == 0:
            _warn(f"put on {t} dropped: no matching SELL_PUT action")
            continue
        if t in kept:
            _warn(f"put on {t} dropped: duplicate (only the first is kept)")
            continue
        chain = chains.get(t)
        q = _find_put(chain, cp.strike, cp.expiry)
        if q is None:
            _warn(f"put on {t} dropped: ${cp.strike:,.2f}P {cp.expiry} is not in the chain")
            continue
        dte = (q.expiry - today).days
        if not dte_min <= dte <= dte_max:
            _warn(f"put on {t} dropped: {dte} days to expiry, outside {dte_min}-{dte_max}")
            continue
        if chain is not None and chain.spot > 0 and cp.strike >= chain.spot:
            _warn(
                f"put on {t} dropped: strike ${cp.strike:,.2f} is not below spot ${chain.spot:,.2f}"
            )
            continue
        delta = q.delta if q.delta is not None else cp.delta
        if not delta_min - _DELTA_TOLERANCE <= abs(delta) <= delta_max + _DELTA_TOLERANCE:
            _warn(
                f"put on {t} dropped: |Δ| {abs(delta):.2f} outside {delta_min:.2f}-{delta_max:.2f}"
            )
            continue

        per_contract = cp.strike * 100.0
        fit = int(min(cand.max_csp_cash, total_room) // per_contract)
        contracts = min(cp.contracts, fit)
        if contracts <= 0:
            _warn(
                f"put on {t} dropped: one contract needs ${per_contract:,.0f} of cash, "
                f"only ${min(cand.max_csp_cash, total_room):,.0f} is allowed"
            )
            continue
        if contracts < cp.contracts:
            _warn(
                f"put on {t} cut from {cp.contracts} to {contracts} contract(s) "
                "to fit the cash caps"
            )
        premium = _mid(q) or cp.est_premium_per_share
        fixed = cp.model_copy(
            update={"contracts": contracts, "delta": -abs(delta), "est_premium_per_share": premium}
        )
        kept[t] = fixed
        total_room -= fixed.cash_reserved

    actions: list[RebalanceAction] = []
    for a in plan.actions:
        if a.action != "SELL_PUT":
            actions.append(a)
        elif a.ticker in kept and not any(
            x.ticker == a.ticker and x.action == "SELL_PUT" for x in actions
        ):
            actions.append(a.model_copy(update={"sizing": csp_sizing(kept[a.ticker])}))
        elif a.ticker not in kept:
            _warn(f"SELL_PUT on {a.ticker} dropped: no valid put detail")

    cleaned = plan.model_copy(update={"actions": actions, "csp_writes": list(kept.values())})
    return cleaned, warnings
