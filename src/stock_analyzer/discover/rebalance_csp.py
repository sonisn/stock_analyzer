"""Cash-secured-put data pipeline and plan validation for rebalance runs.

Mirrors rebalance_cc.py: `run_csp_data_pipeline` gathers candidates and
put chains and builds the prompt block; `apply_csp_plan_validation`
cleans the rebalancer's SELL_PUT output; `csp_report_data` feeds the
report section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..config import Settings
from ..logging import get_logger
from ..models.market import OptionChain
from ..models.portfolio import CspCandidate
from ..models.rebalance import RebalancePlan
from .rebalance_cc import earnings_dates_from_signals

logger = get_logger(__name__)


def csp_empty_state() -> dict[str, Any]:
    return {
        "csp_context_block": "",
        "csp_eligibility": {},
        "csp_chains": {},
        "csp_cash_budget": 0.0,
        "csp_account_room": {},
    }


@dataclass
class CspDataResult:
    context_block: str = ""
    eligibility: dict[str, CspCandidate] = field(default_factory=dict)
    chains: dict[str, OptionChain] = field(default_factory=dict)
    cash_budget: float = 0.0
    account_room: dict[str, float] = field(default_factory=dict)
    content: str = ""
    # Why no put is on offer, when candidates existed but none fit. An
    # empty block used to reach the report as silence, which reads as
    # "there was nothing to say" rather than "a cap ruled everything out".
    blocked_note: str = ""
    cheap_premium: dict[str, str] = field(default_factory=dict)


def _earnings_dates(
    tickers: list[str], finnhub_signals: dict[str, Any], db_path: str
) -> dict[str, date]:
    """Finnhub's dates where the run already has them; the cached calendar
    (data/reference.py) for the rest (older picks aren't in this run's set)."""
    from ..data.reference import next_earnings_dates

    out = earnings_dates_from_signals({t: [] for t in tickers}, finnhub_signals)
    missing = [t for t in tickers if t not in out]
    if missing:
        out.update({t: d for t, d in next_earnings_dates(missing, db_path).items() if d})
    return out


def _account_room(
    account_cash: dict[str, float],
    open_puts: dict[str, dict[str, Any]],
    options_accounts: tuple[str, ...],
) -> dict[str, float]:
    """{account: cash free to secure new puts} — the account's cash minus
    collateral its open short puts already hold, for accounts allowed to
    sell puts (`OPTIONS_ACCOUNTS`; empty = all). Assumes the broker's cash
    balance still includes that collateral (the conservative reading)."""
    held: dict[str, float] = {}
    for rec in open_puts.values():
        for acct, amount in (rec.get("by_account") or {}).items():
            held[acct] = held.get(acct, 0.0) + amount
    allowed = set(options_accounts)
    return {
        acct: max(cash - held.get(acct, 0.0), 0.0)
        for acct, cash in account_cash.items()
        if not allowed or acct in allowed
    }


def _round_lot_tickers(position_splits: dict[str, Any] | None) -> set[str] | None:
    """Tickers with 100+ shares in a single account — the ones that can
    back a covered call. None when per-account data is missing."""
    if not position_splits:
        return None
    return {
        t
        for t, info in position_splits.items()
        if any(float(s.get("units") or 0) >= 100 for s in info.get("splits") or [])
    }


def run_csp_data_pipeline(
    state: dict[str, Any],
    settings: Settings,
    recent_picks: list[tuple[str, int, str]],
) -> CspDataResult:
    """Find put candidates among recent picks, fetch their put chains,
    and build the CASH-SECURED PUT CONTEXT block for the rebalancer.

    `recent_picks` is [(ticker, rank, run_at), ...] from past runs; this
    run's own picks (`state["picks"]`) are added here."""
    from ..data.brokerage import fetch_open_short_puts
    from ..data.options_chain import fetch_chains
    from .cc_eligibility import apply_earnings_filter
    from .csp_eligibility import build_csp_context_block, eligible_csp_tickers, fill_put_deltas

    cash = state.get("cash_balance")
    if cash is None:
        return CspDataResult(content="csp_data: cash balance unknown; puts need known cash")

    try:
        open_puts = fetch_open_short_puts()
    except Exception as e:
        logger.warning("open short-put fetch failed: %s", e)
        open_puts = {}
    reserved = sum(rec["collateral_usd"] for rec in open_puts.values())
    account_room = _account_room(
        state.get("account_cash") or {}, open_puts, settings.options_accounts
    )
    budget = sum(account_room.values()) if account_room else max(float(cash) - reserved, 0.0)
    logger.info(
        "CSP: cash $%s, $%s already reserved by %d open short put(s) → budget $%s %s",
        f"{cash:,.0f}",
        f"{reserved:,.0f}",
        len(open_puts),
        f"{budget:,.0f}",
        {a: round(r) for a, r in account_room.items()},
    )

    now = date.today().isoformat()
    picks = [(t, rank, now) for rank, t, _ in state.get("picks") or []] + list(recent_picks)
    thesis = {c["ticker"]: c["status"] for c in state.get("thesis_checks") or []}
    eligible = eligible_csp_tickers(
        picks,
        positions=state.get("holdings_positions") or {},
        cash_budget=budget,
        denylist=settings.options_denylist,
        open_short_puts=open_puts,
        thesis_status=thesis,
        max_pct_per_put=settings.csp_max_pct_per_put,
        max_pct_total=settings.csp_max_pct_total,
        covered_call_tickers=_round_lot_tickers(state.get("position_splits")),
        max_account_room=max(account_room.values()) if account_room else None,
    )
    logger.info("CSP eligibility: %d candidate(s): %s", len(eligible), sorted(eligible))
    if not eligible:
        return CspDataResult(
            cash_budget=budget,
            account_room=account_room,
            content="csp_data: no put candidates this run",
        )

    chains = fetch_chains(
        list(eligible), dte_min=settings.csp_dte_min, dte_max=settings.csp_dte_max, kind="puts"
    )
    earnings = _earnings_dates(
        list(eligible), state.get("finnhub_signals") or {}, settings.discover_db_path
    )
    ready: dict[str, OptionChain] = {}
    for t, chain in chains.items():
        filtered, _ = apply_earnings_filter(chain, earnings_date=earnings.get(t))
        ready[t] = fill_put_deltas(filtered)

    # The put side had no volatility test while the call side did, so a
    # put could be offered at an implied vol below what the stock actually
    # realizes — selling insurance under cost. Same floor, same reasoning.
    from .rebalance_cc import compute_iv_hv_regimes, drop_cheap_premium

    cheap: dict[str, str] = {}
    if settings.csp_min_iv_hv_ratio > 0:
        regimes = compute_iv_hv_regimes({t: [] for t in eligible}, ready)
        kept, cheap = drop_cheap_premium(
            dict.fromkeys(eligible, []), regimes, min_ratio=settings.csp_min_iv_hv_ratio
        )
        if cheap:
            logger.info(
                "CSP: %d candidate(s) dropped for cheap premium: %s",
                len(cheap),
                ", ".join(sorted(cheap)),
            )
        eligible = {t: c for t, c in eligible.items() if t in kept}
        ready = {t: c for t, c in ready.items() if t in kept}
    if not eligible:
        return CspDataResult(
            cash_budget=budget,
            account_room=account_room,
            cheap_premium=cheap,
            blocked_note=(
                "No cash-secured put is worth writing: every candidate's implied "
                f"volatility is below its realized volatility (floor "
                f"{settings.csp_min_iv_hv_ratio:.2f}x), so the premium does not pay "
                "for the risk."
            ),
            content=f"csp_data: all {len(cheap)} candidate(s) dropped as cheap premium",
        )

    block = build_csp_context_block(
        candidates=eligible,
        chains=ready,
        earnings=earnings,
        cash_budget=budget,
        open_put_collateral=reserved,
        account_room=account_room,
        delta_min=settings.csp_target_delta_min,
        delta_max=settings.csp_target_delta_max,
        max_pct_per_put=settings.csp_max_pct_per_put,
        max_pct_total=settings.csp_max_pct_total,
    )
    sources = sorted({c.source for c in chains.values()})
    logger.info("CSP context block built: %d chars; chain sources %s", len(block), sources)
    blocked_note = "" if block else _blocked_note(eligible, ready, budget, settings)
    if blocked_note:
        logger.info("CSP: %s", blocked_note)
    return CspDataResult(
        context_block=block,
        eligibility=eligible,
        chains=ready,
        cash_budget=budget,
        account_room=account_room,
        blocked_note=blocked_note,
        cheap_premium=cheap,
        content=(
            f"csp_data: {len(eligible)} candidate(s); chain sources {sources}; "
            f"budget ${budget:,.0f}; context block {len(block)} chars"
        ),
    )


def _blocked_note(
    eligible: dict[str, CspCandidate],
    chains: dict[str, OptionChain],
    budget: float,
    settings: Settings,
) -> str:
    """Name the binding constraint when candidates existed but none fit.

    Nearly always the per-put collateral cap: a put needs 100 x strike in
    cash, so a 25% cap on a $20k budget only reaches a $52 strike, and
    every liquid name is priced far above that. Saying so is the
    difference between a setting the user can change and silence they
    cannot interpret.
    """
    cap = budget * settings.csp_max_pct_per_put
    cheapest: tuple[str, float] | None = None
    for ticker, chain in chains.items():
        strikes = [q.strike for q in getattr(chain, "puts", []) if q.strike and q.strike > 0]
        if not strikes:
            continue
        need = min(strikes) * 100
        if cheapest is None or need < cheapest[1]:
            cheapest = (ticker, need)
    if cheapest is None:
        return (
            f"No put chain came back for any of the {len(eligible)} candidate(s), so "
            "none could be priced."
        )
    ticker, need = cheapest
    if need > cap:
        return (
            f"No cash-secured put fits. One contract must be fully secured, so the "
            f"{settings.csp_max_pct_per_put:.0%} per-put cap on ${budget:,.0f} allows "
            f"${cap:,.0f} of collateral — a strike of ${cap / 100:,.2f} or less. The "
            f"cheapest put on offer is {ticker} at ${need:,.0f}. Raise "
            f"CSP_MAX_PCT_PER_PUT, or add lower-priced candidates."
        )
    return (
        f"No put sat inside the {settings.csp_target_delta_min:.2f}-"
        f"{settings.csp_target_delta_max:.2f} delta band at "
        f"{settings.csp_dte_min}-{settings.csp_dte_max} days for any of the "
        f"{len(eligible)} candidate(s)."
    )


def apply_csp_plan_validation(
    plan: RebalancePlan,
    *,
    chains: dict[str, OptionChain],
    eligibility: dict[str, CspCandidate],
    cash_budget: float,
    settings: Settings,
    account_room: dict[str, float] | None = None,
    units: dict[str, float] | None = None,
    prices: dict[str, float | None] | None = None,
) -> tuple[RebalancePlan, list[str]]:
    """Backfill, then validate the plan's puts against the cash its own
    BUYs/ADDs leave behind (so buys + collateral can't exceed cash)."""
    from .csp_validation import backfill_csp_writes, cash_left_for_puts, validate_csp_writes

    plan = backfill_csp_writes(plan, chains=chains)
    budget, room, notes = cash_left_for_puts(
        plan,
        cash_budget=cash_budget,
        account_room=account_room or {},
        units=units or {},
        prices=prices or {},
    )
    if plan.csp_writes and budget < cash_budget:
        logger.info(
            "CSP: plan's own trades leave $%s of the $%s put budget",
            f"{budget:,.0f}",
            f"{cash_budget:,.0f}",
        )
    plan, warnings = validate_csp_writes(
        plan,
        eligible=eligibility,
        chains=chains,
        cash_budget=budget,
        delta_min=settings.csp_target_delta_min,
        delta_max=settings.csp_target_delta_max,
        dte_min=settings.csp_dte_min,
        dte_max=settings.csp_dte_max,
        max_pct_total=settings.csp_max_pct_total,
        account_room=room if account_room else None,
    )
    if plan.csp_writes and notes:
        warnings.append("put cash check approximate: " + "; ".join(notes))
    for cp in plan.csp_writes:
        logger.info(
            "  - SELL_PUT %s: %d × $%.2fP %s, Δ %.2f, ~$%s premium, $%s cash reserved",
            cp.ticker,
            cp.contracts,
            cp.strike,
            cp.expiry,
            cp.delta,
            f"{cp.premium_usd:,.0f}",
            f"{cp.cash_reserved:,.0f}",
        )
    return plan, warnings


def csp_report_data(plan: RebalancePlan | None, *, cash_budget: float) -> dict[str, Any] | None:
    """Rows + totals for the report's cash-secured-put section; None when
    the plan sells no puts."""
    if plan is None or not plan.csp_writes:
        return None
    rows: list[dict[str, Any]] = []
    for cp in plan.csp_writes:
        days = (date.fromisoformat(cp.expiry) - date.today()).days
        yield_pct = (
            cp.est_premium_per_share / cp.strike * 365.0 / days * 100.0
            if days > 0 and cp.strike > 0
            else None
        )
        rows.append(
            {
                "ticker": cp.ticker,
                "account": cp.account,
                "contracts": cp.contracts,
                "strike": cp.strike,
                "expiry": cp.expiry,
                "delta": cp.delta,
                "premium_usd": cp.premium_usd,
                "cash_reserved": cp.cash_reserved,
                "annualized_yield_pct": yield_pct,
                "net_cost_if_assigned": cp.strike - cp.est_premium_per_share,
            }
        )
    reserved = sum(r["cash_reserved"] for r in rows)
    return {
        "rows": rows,
        "total_premium_usd": sum(r["premium_usd"] for r in rows),
        "total_cash_reserved": reserved,
        "cash_budget": cash_budget,
        "pct_of_budget": reserved / cash_budget * 100.0 if cash_budget > 0 else None,
    }
