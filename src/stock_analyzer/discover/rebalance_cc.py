"""Covered-call data pipeline and rebalancer plan validation for rebalance runs."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from ..config import Settings
from ..logging import get_logger
from ..models.portfolio import IvHvRegime
from ..models.rebalance import RebalancePlan

if TYPE_CHECKING:
    from ..data.options_chain import OptionChain

logger = get_logger(__name__)

# Cap CC eligible holdings sent to the rebalancer to bound prompt size.
_CC_MAX_ELIGIBLE_FOR_PROMPT = 25


def cc_empty_state() -> dict[str, Any]:
    return {
        "cc_context_block": "",
        "cc_eligibility": {},
        "cc_round_lot_coverage": {},
        "cc_stub_pool_total_usd": 0.0,
    }


def resolve_eligible_holdings(
    *,
    eligible: dict[str, list[Any]],
    holdings_technicals: dict[str, dict[str, Any]],
    cap: int = _CC_MAX_ELIGIBLE_FOR_PROMPT,
) -> dict[str, list[Any]]:
    """Bound eligible tickers by total dollar exposure."""
    if len(eligible) <= cap:
        return eligible

    def _exposure(t: str) -> float:
        spot = (holdings_technicals.get(t) or {}).get("price") or 0.0
        total = sum(eh.available_shares for eh in eligible[t])
        return float(total) * float(spot)

    kept = sorted(eligible, key=_exposure, reverse=True)[:cap]
    dropped = sorted(set(eligible) - set(kept))
    logger.warning(
        "CC: %d eligible tickers exceed prompt cap (%d); keeping top %d by exposure, dropping %s",
        len(eligible),
        cap,
        len(kept),
        dropped,
    )
    return {t: eligible[t] for t in kept}


def earnings_dates_from_signals(
    eligible: dict[str, list[Any]],
    finnhub_signals: dict[str, Any],
) -> dict[str, date]:
    earnings_map: dict[str, date] = {}
    for ticker in eligible:
        sig = finnhub_signals.get(ticker) or {}
        raw = sig.get("next_earnings_date") or sig.get("earnings_date")
        if isinstance(raw, str):
            with contextlib.suppress(ValueError):
                earnings_map[ticker] = date.fromisoformat(raw[:10])
        elif isinstance(raw, date):
            earnings_map[ticker] = raw
    return earnings_map


def filter_chains_by_earnings(
    chains: dict[str, Any],
    earnings_map: dict[str, date],
) -> dict[str, OptionChain]:
    from .cc_eligibility import apply_earnings_filter

    filtered: dict[str, OptionChain] = {}
    for ticker, chain in chains.items():
        filtered_chain, _ = apply_earnings_filter(
            chain,
            earnings_date=earnings_map.get(ticker),
        )
        filtered[ticker] = filtered_chain
    return filtered


def _best_in_band_premiums(
    coverage: dict[str, Any],
    spots: dict[str, float],
    settings: Any,
) -> dict[str, tuple[float, float, str]]:
    """{ticker: (premium, strike, expiry)} for the best policy-compliant
    call on each part-lot worth completing. Chains are free to fetch and
    the answer is what makes the trade-off legible, but a failure here
    must never cost the run its plan."""
    from ..data.options_chain import fetch_chains

    wanted = [
        t
        for t, rec in coverage.items()
        if rec.to_next_lot_shares and rec.stub_dollar_value >= settings.cc_min_stub_usd
    ]
    if not wanted:
        return {}
    try:
        chains = fetch_chains(
            wanted, dte_min=settings.cc_dte_min, dte_max=settings.cc_dte_max, kind="calls"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Stub premium lookup failed (%s) — cost shown without income", e)
        return {}
    out: dict[str, tuple[float, float, str]] = {}
    for ticker, chain in chains.items():
        spot = spots.get(ticker) or 0.0
        best = None
        for q in getattr(chain, "calls", []):
            delta = abs(q.delta) if q.delta is not None else None
            if delta is None or not (
                settings.cc_target_delta_min <= delta <= settings.cc_target_delta_max
            ):
                continue
            if not q.bid or q.bid <= 0 or not spot:
                continue
            if q.strike < spot * (1 + settings.cc_min_upside_pct / 100):
                continue
            if best is None or q.bid > best.bid:
                best = q
        if best is not None:
            out[ticker] = (best.bid * 100, best.strike, str(best.expiry))
    return out


def stub_income_block(
    coverage: dict[str, Any],
    premiums: dict[str, tuple[float, float, str]],
    *,
    min_stub_usd: float,
) -> str:
    """What completing each part-lot would cost, and what it would pay.

    The round-lot table already says a stub is 15 shares short of a
    writable lot. What it never said is that those 15 shares buy a
    contract worth $1,325 — so "sell the stub" was the only option ever
    priced, and on 2026-09-20 a plan trimmed the exact 61 AVGO shares
    that sat 38 short of a second lot. `premiums` maps ticker ->
    (premium_usd, strike, expiry) for the best in-band call.
    """
    rows = []
    for ticker in sorted(coverage):
        rec = coverage[ticker]
        shares = getattr(rec, "to_next_lot_shares", 0) or 0
        cost = getattr(rec, "to_next_lot_cost", 0.0) or 0.0
        if not shares or getattr(rec, "stub_dollar_value", 0.0) < min_stub_usd:
            continue
        line = (
            f"  {ticker}: {shares:,.0f} more share(s) (~${cost:,.0f}) completes a "
            f"writable lot from the {getattr(rec, 'stub_shares', 0):,.0f}-share stub"
        )
        if quote := premiums.get(ticker):
            premium, strike, expiry = quote
            line += f" — one call at ${strike:,.0f} expiring {expiry} pays ~${premium:,.0f}" + (
                f" ({premium / cost:.0%} of the cost)" if cost > 0 else ""
            )
        rows.append(line)
    if not rows:
        return ""
    return (
        "COMPLETING A PART-LOT (premium the stub cannot earn as it stands)\n"
        "A covered call needs 100 shares in ONE account. These holdings are\n"
        "close. Weigh buying the shortfall against selling the stub: selling it\n"
        "ends the premium permanently, and the figures below are what that\n"
        "premium is worth today.\n" + "\n".join(rows)
    )


def writable_positions(positions: dict[str, Any], spots: dict[str, float]) -> dict[str, Any]:
    """The holdings a covered call could actually be written on.

    Round-lot coverage answers "how close is this to a writable lot", so
    it may only contain tradable equity. Unfiltered, a money-market sweep
    reported 20,845 units as 208 round lots and a revoked CUSIP offered
    one more — `reporting.health.is_cash_like` was added for that exact
    symptom in the daily email and never reached this path. Positions
    dropped here are still held, valued and taxed everywhere else.
    """
    from ..data.brokerage import is_listed_symbol
    from ..reporting.health import is_cash_like

    return {
        t: p
        for t, p in positions.items()
        if is_listed_symbol(t)
        and not is_cash_like(t, spots.get(t))
        # A price is required, not incidental: this table exists to say
        # what the gap to the next lot COSTS. Without a quote it can only
        # print $0 stub and $0 to-next-lot, which is what a 401(k)
        # commingled pool and two revoked CUSIPs did on 2026-09-20.
        and float(spots.get(t) or 0) > 0
    }


def compute_iv_hv_regimes(
    eligible: dict[str, list[Any]],
    filtered_chains: dict[str, OptionChain],
) -> dict[str, IvHvRegime]:
    from ..data.historical_volatility import fetch_realized_volatility
    from .cc_eligibility import compute_iv_hv_regime

    hv_data = fetch_realized_volatility(list(eligible))
    iv_hv_regimes: dict[str, IvHvRegime] = {}
    for ticker in eligible:
        regime = compute_iv_hv_regime(
            chain=filtered_chains.get(ticker),
            hv=hv_data.get(ticker),
        )
        if regime is not None:
            iv_hv_regimes[ticker] = regime
    return iv_hv_regimes


@dataclass
class CcDataResult:
    context_block: str = ""
    eligibility: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    stub_pool: float = 0.0
    chains: dict[str, OptionChain] = field(default_factory=dict)
    iv_hv_regimes: dict[str, IvHvRegime] = field(default_factory=dict)
    # {ticker: why no call was offered on it today}
    cheap_premium: dict[str, str] = field(default_factory=dict)
    # What completing each part-lot costs and earns.
    stub_income_block: str = ""
    content: str = ""


def run_cc_data_pipeline(state: dict[str, Any], settings: Settings) -> CcDataResult:
    """Fetch chains, eligibility, and build the CC context block for Opus."""
    from .cc_eligibility import (
        build_cc_context_block,
    )

    logger.info(
        "CC pipeline starting: CC_ENABLED=%s, delta_band=[%.2f, %.2f], "
        "DTE_band=[%d, %d], min_premium=$%.0f, slippage_buffer=%.0f%%",
        settings.cc_enabled,
        settings.cc_target_delta_min,
        settings.cc_target_delta_max,
        settings.cc_dte_min,
        settings.cc_dte_max,
        settings.cc_min_premium_usd,
        settings.cc_slippage_buffer * 100,
    )

    positions = state.get("holdings_positions") or {}
    eligible = _eligible_for_calls(state, settings)
    coverage, stub_pool, stub_premiums = _stub_coverage(state, positions, settings)
    chains, chain_sources = _fetch_call_chains(eligible, settings)

    finnhub_signals = state.get("finnhub_signals") or {}
    earnings_map = earnings_dates_from_signals(eligible, finnhub_signals)
    logger.info(
        "CC earnings dates: %d/%d eligible tickers have known earnings date(s)",
        len(earnings_map),
        len(eligible),
    )

    filtered_chains = filter_chains_by_earnings(chains, earnings_map)
    iv_hv_regimes = compute_iv_hv_regimes(eligible, filtered_chains)
    logger.info(
        "CC IV/HV regimes: %s",
        {t: f"{r.iv_hv_ratio:.2f}x ({r.label})" for t, r in iv_hv_regimes.items()},
    )

    # Write when the market is paying up, not merely because shares are
    # free to cover. Below CC_MIN_IV_HV_RATIO the option market is asking
    # less for the upside than the stock's own realized movement says it
    # is worth, and waiting costs nothing but time.
    eligible, cheap = drop_cheap_premium(
        eligible, iv_hv_regimes, min_ratio=settings.cc_min_iv_hv_ratio
    )
    if cheap:
        logger.info("CC: holding off on %s — premium is cheap", ", ".join(sorted(cheap)))

    block = build_cc_context_block(
        eligible=eligible,
        chains=filtered_chains,
        coverage=coverage,
        reviews=state.get("holdings_reviews", {}),
        earnings=earnings_map,
        stub_pool_total_usd=stub_pool,
        iv_hv_regimes=iv_hv_regimes,
    )
    logger.info(
        "CC context block built: %d chars (will be fed to rebalancer Opus)",
        len(block),
    )

    return CcDataResult(
        context_block=block,
        eligibility=eligible,
        cheap_premium=cheap,
        coverage=coverage,
        stub_pool=stub_pool,
        stub_income_block=stub_income_block(
            coverage, stub_premiums, min_stub_usd=settings.cc_min_stub_usd
        ),
        chains=filtered_chains,
        iv_hv_regimes=iv_hv_regimes,
        content=(
            f"cc_data: {len(eligible)} eligible holding(s); "
            f"chain sources {sorted(chain_sources.keys())}; "
            f"stub pool ${stub_pool:,.0f}; "
            f"context block {len(block)} chars"
        ),
    )


def _eligible_for_calls(state: dict[str, Any], settings: Settings) -> dict[str, list[Any]]:
    """(ticker, account) pairs with 100+ shares not already backing a call."""
    from ..data.brokerage import fetch_open_option_positions
    from .cc_eligibility import eligible_holdings_per_account

    denylist = settings.options_denylist

    try:
        open_short_calls = fetch_open_option_positions()
    except Exception as e:
        logger.warning("open option position fetch failed: %s", e)
        open_short_calls = {}

    if open_short_calls:
        logger.info(
            "CC: %d ticker(s) already collateralizing short calls: %s",
            len(open_short_calls),
            dict(open_short_calls),
        )
    else:
        logger.info("CC: no existing short-call coverage detected")

    position_splits = state.get("position_splits") or {}
    eligible = eligible_holdings_per_account(
        position_splits,
        open_short_calls_by_account=open_short_calls,
        denylist=denylist,
        options_accounts=settings.options_accounts,
    )
    eligible = resolve_eligible_holdings(
        eligible=eligible,
        holdings_technicals=state.get("holdings_technicals") or {},
    )

    n_pairs = sum(len(v) for v in eligible.values())
    logger.info(
        "CC eligibility: %d ticker(s) / %d (ticker, account) pair(s) eligible. Pairs: %s",
        len(eligible),
        n_pairs,
        sorted((eh.ticker, eh.account) for v in eligible.values() for eh in v),
    )
    if not eligible:
        logger.warning(
            "CC: NO eligible holdings — rebalancer will produce NO WRITE_CALL "
            "recommendations. Reasons: positions < 100 shares OR all in denylist "
            "OR fully collateralized by existing short calls."
        )
    return eligible


def _stub_coverage(
    state: dict[str, Any], positions: dict[str, Any], settings: Settings
) -> tuple[dict[str, Any], float, dict[str, tuple[float, float, str]]]:
    """Round-lot coverage of writable positions, the part-lot dollar pool,
    and what the best call on each completed lot would pay."""
    from .cc_eligibility import round_lot_coverage

    spots = {
        t: (state.get("holdings_technicals", {}).get(t) or {}).get("price") or 0.0
        for t in positions
    }
    writable = writable_positions(positions, spots)
    if skipped := sorted(set(positions) - set(writable)):
        logger.info("Round-lot coverage skips %s — not writable", ", ".join(skipped))
    coverage = round_lot_coverage(writable, spots=spots)
    stub_pool = sum(rec.stub_dollar_value for rec in coverage.values() if rec.stub_shares)
    stub_premiums = _best_in_band_premiums(coverage, spots, settings)

    stub_eligible = sum(
        1 for rec in coverage.values() if rec.stub_dollar_value >= settings.cc_min_stub_usd
    )
    logger.info(
        "CC round-lot coverage: %d holding(s) have stubs, $%s total stub pool; "
        "%d stub(s) exceed CC_MIN_STUB_USD=$%s threshold",
        sum(1 for r in coverage.values() if r.stub_shares > 0),
        f"{stub_pool:,.0f}",
        stub_eligible,
        f"{settings.cc_min_stub_usd:,.0f}",
    )
    return coverage, stub_pool, stub_premiums


def _fetch_call_chains(
    eligible: dict[str, list[Any]], settings: Settings
) -> tuple[dict[str, Any], dict[str, int]]:
    """Call chains for the eligible tickers, and a count per provider."""
    from ..data.options_chain import fetch_chains

    chains = fetch_chains(
        list(eligible),
        dte_min=settings.cc_dte_min,
        dte_max=settings.cc_dte_max,
    )

    chain_sources: dict[str, int] = {}
    for c in chains.values():
        chain_sources[c.source] = chain_sources.get(c.source, 0) + 1
    logger.info(
        "CC chain fetch: %d eligible ticker(s); sources: %s",
        len(chains),
        dict(chain_sources),
    )
    if chains and all(c.source == "missing" for c in chains.values()):
        logger.error(
            "CC: ALL chain fetches failed (SnapTrade + yfinance both miss). "
            "Opus will see UNAVAILABLE for every ticker and won't emit "
            "WRITE_CALL. Check yfinance connectivity + SnapTrade tier."
        )
    return chains, chain_sources


def log_rebalancer_input_estimate(
    state: dict[str, Any],
    *,
    ranker_text: str,
    history_block: str,
) -> None:
    reviews_block_chars = sum(
        len(getattr(r, "full_text", str(r)) or "")
        for r in (state.get("holdings_reviews") or {}).values()
    )
    cc_block_chars = len(state.get("cc_context_block") or "")
    ranker_chars = len(ranker_text)
    history_chars = len(history_block)
    themes_chars = len(state.get("market_themes_block", "") or "")
    macro_chars = len(state.get("macro_summary", "") or "")
    total_input_chars = (
        reviews_block_chars
        + cc_block_chars
        + ranker_chars
        + history_chars
        + themes_chars
        + macro_chars
    )
    approx_input_tokens = total_input_chars // 4
    logger.info(
        "Rebalancer input estimate: %d total chars (~%d tokens). "
        "Breakdown: reviews=%d, cc_block=%d, ranker=%d, history=%d, "
        "themes=%d, macro=%d. (Opus 4.7 input limit: 200,000 tokens.)",
        total_input_chars,
        approx_input_tokens,
        reviews_block_chars,
        cc_block_chars,
        ranker_chars,
        history_chars,
        themes_chars,
        macro_chars,
    )
    if approx_input_tokens > 150_000:
        logger.warning(
            "Rebalancer input is approaching the 200K-token context "
            "limit (estimated %d tokens). Consider reducing the number "
            "of holdings reviewed, or shortening reviewer.full_text.",
            approx_input_tokens,
        )


def drop_cheap_premium(
    eligible: dict[str, Any],
    iv_hv_regimes: dict[str, IvHvRegime],
    *,
    min_ratio: float,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Keep only holdings whose options are paying above realized vol.

    Returns (kept, {ticker: why it was held back}). A ticker with no
    regime reading is kept: an unknown IV is not evidence of a cheap one,
    and dropping it would quietly stop writing calls whenever the vol
    data failed.
    """
    if min_ratio <= 0:
        return eligible, {}
    kept, cheap = {}, {}
    for ticker, entries in eligible.items():
        regime = iv_hv_regimes.get(ticker)
        if regime is None or regime.iv_hv_ratio >= min_ratio:
            kept[ticker] = entries
            continue
        cheap[ticker] = (
            f"{ticker}: IV {regime.current_iv:.0%} is only {regime.iv_hv_ratio:.2f}x its "
            f"realized {regime.hv_annualized:.0%} ({regime.label}) — below the "
            f"{min_ratio:.2f}x floor, so the upside is being underpaid. Wait for a "
            f"volatile session."
        )
    return kept, cheap


def apply_cc_plan_validation(
    plan: RebalancePlan,
    *,
    chains: dict[str, OptionChain],
    eligibility: dict[str, Any],
    cc_context_block: str,
    settings: Settings | None = None,
    spots: dict[str, float] | None = None,
) -> tuple[RebalancePlan, list[str]]:
    from .cc_backfill import backfill_option_writes
    from .cc_validation import validate_option_writes

    plan = backfill_option_writes(plan, chains=chains)
    plan, cc_warnings = validate_option_writes(
        plan,
        eligibility=eligibility,
        spots=spots,
        # The keep-the-shares rules are enforced, not suggested: a rich
        # premium is not a reason to accept a strike that gives the
        # position no room.
        delta_max=settings.cc_target_delta_max if settings else None,
        min_upside_pct=settings.cc_min_upside_pct if settings else None,
        dte_min=settings.cc_dte_min if settings else None,
        dte_max=settings.cc_dte_max if settings else None,
    )
    for w in cc_warnings:
        logger.warning("CC plan validation: %s", w)

    n_write_calls = sum(1 for a in plan.actions if a.action == "WRITE_CALL")
    if n_write_calls > 0:
        total_premium = sum(
            ow.contracts * ow.est_premium_per_share * 100.0 for ow in plan.option_writes
        )
        logger.info(
            "CC validation passed: %d WRITE_CALL action(s), $%s gross premium estimated. Details:",
            n_write_calls,
            f"{total_premium:,.0f}",
        )
        for ow in plan.option_writes:
            contract_premium = ow.contracts * ow.est_premium_per_share * 100.0
            logger.info(
                "  - %s: %d contracts @ $%.2f strike, expires %s, "
                "Δ=%.2f, ~$%s premium, assignment %.0f%%",
                ow.ticker,
                ow.contracts,
                ow.strike,
                ow.expiry,
                ow.delta,
                f"{contract_premium:,.0f}",
                ow.assignment_probability * 100,
            )
    elif not cc_context_block:
        logger.info(
            "CC: no WRITE_CALL recommendations — CC context was empty "
            "this run (no eligible holdings or CC_ENABLED=false)."
        )
    else:
        logger.warning(
            "CC: rebalancer received CC context (%d chars) but emitted "
            "0 WRITE_CALL actions. Possible reasons: every eligible chain "
            "failed the liquidity guard (bid<$0.20, OI<100, spread>15%%), "
            "every eligible position is a SELL verdict, or Opus declined.",
            len(cc_context_block),
        )
    return plan, cc_warnings
