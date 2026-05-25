"""Covered-call data pipeline and rebalancer plan validation for rebalance runs."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from ..config import Settings
from ..logging import get_logger
from ..models.portfolio import IvHvRegime
from ..models.rebalance import RebalancePlan

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
        "CC: %d eligible tickers exceed prompt cap (%d); "
        "keeping top %d by exposure, dropping %s",
        len(eligible), cap, len(kept), dropped,
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
) -> dict[str, object]:
    from .cc_eligibility import apply_earnings_filter

    filtered: dict[str, object] = {}
    for ticker, chain in chains.items():
        filtered_chain, _ = apply_earnings_filter(
            chain, earnings_date=earnings_map.get(ticker),
        )
        filtered[ticker] = filtered_chain
    return filtered


def compute_iv_hv_regimes(
    eligible: dict[str, list[Any]],
    filtered_chains: dict[str, object],
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
    chains: dict[str, object] = field(default_factory=dict)
    iv_hv_regimes: dict[str, IvHvRegime] = field(default_factory=dict)
    content: str = ""


def run_cc_data_pipeline(state: dict[str, Any], settings: Settings) -> CcDataResult:
    """Fetch chains, eligibility, and build the CC context block for Opus."""
    from ..data.brokerage import fetch_open_option_positions
    from ..data.options_chain import fetch_chains
    from .cc_eligibility import (
        build_cc_context_block,
        eligible_holdings_per_account,
        round_lot_coverage,
    )

    logger.info(
        "CC pipeline starting: CC_ENABLED=%s, delta_band=[%.2f, %.2f], "
        "DTE_band=[%d, %d], min_premium=$%.0f, slippage_buffer=%.0f%%",
        settings.cc_enabled,
        settings.cc_target_delta_min, settings.cc_target_delta_max,
        settings.cc_dte_min, settings.cc_dte_max,
        settings.cc_min_premium_usd,
        settings.cc_slippage_buffer * 100,
    )

    positions = state.get("holdings_positions") or {}
    denylist = settings.cc_denylist

    try:
        open_short_calls = fetch_open_option_positions()
    except Exception as e:
        logger.warning("open option position fetch failed: %s", e)
        open_short_calls = {}

    if open_short_calls:
        logger.info(
            "CC: %d ticker(s) already collateralizing short calls: %s",
            len(open_short_calls), dict(open_short_calls),
        )
    else:
        logger.info("CC: no existing short-call coverage detected")

    position_splits = state.get("position_splits") or {}
    eligible = eligible_holdings_per_account(
        position_splits,
        open_short_calls_by_account=open_short_calls,
        denylist=denylist,
    )
    eligible = resolve_eligible_holdings(
        eligible=eligible,
        holdings_technicals=state.get("holdings_technicals") or {},
    )

    n_pairs = sum(len(v) for v in eligible.values())
    logger.info(
        "CC eligibility: %d ticker(s) / %d (ticker, account) pair(s) eligible. "
        "Pairs: %s",
        len(eligible), n_pairs,
        sorted((eh.ticker, eh.account) for v in eligible.values() for eh in v),
    )
    if not eligible:
        logger.warning(
            "CC: NO eligible holdings — rebalancer will produce NO WRITE_CALL "
            "recommendations. Reasons: positions < 100 shares OR all in denylist "
            "OR fully collateralized by existing short calls."
        )

    spots = {
        t: (state.get("holdings_technicals", {}).get(t) or {}).get("price") or 0.0
        for t in positions
    }
    coverage = round_lot_coverage(positions, spots=spots)
    stub_pool = sum(
        rec.stub_dollar_value for rec in coverage.values() if rec.stub_shares
    )

    stub_eligible = sum(
        1 for rec in coverage.values()
        if rec.stub_dollar_value >= settings.cc_min_stub_usd
    )
    logger.info(
        "CC round-lot coverage: %d holding(s) have stubs, $%s total stub pool; "
        "%d stub(s) exceed CC_MIN_STUB_USD=$%s threshold",
        sum(1 for r in coverage.values() if r.stub_shares > 0),
        f"{stub_pool:,.0f}",
        stub_eligible,
        f"{settings.cc_min_stub_usd:,.0f}",
    )

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
        len(chains), dict(chain_sources),
    )
    if chains and all(c.source == "missing" for c in chains.values()):
        logger.error(
            "CC: ALL chain fetches failed (SnapTrade + yfinance both miss). "
            "Opus will see UNAVAILABLE for every ticker and won't emit "
            "WRITE_CALL. Check yfinance connectivity + SnapTrade tier."
        )

    finnhub_signals = state.get("finnhub_signals") or {}
    earnings_map = earnings_dates_from_signals(eligible, finnhub_signals)
    logger.info(
        "CC earnings dates: %d/%d eligible tickers have known earnings date(s)",
        len(earnings_map), len(eligible),
    )

    filtered_chains = filter_chains_by_earnings(chains, earnings_map)
    iv_hv_regimes = compute_iv_hv_regimes(eligible, filtered_chains)
    logger.info(
        "CC IV/HV regimes: %s",
        {t: f"{r.iv_hv_ratio:.2f}x ({r.label})" for t, r in iv_hv_regimes.items()},
    )

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
        coverage=coverage,
        stub_pool=stub_pool,
        chains=filtered_chains,
        iv_hv_regimes=iv_hv_regimes,
        content=(
            f"cc_data: {len(eligible)} eligible holding(s); "
            f"chain sources {sorted(chain_sources.keys())}; "
            f"stub pool ${stub_pool:,.0f}; "
            f"context block {len(block)} chars"
        ),
    )


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
        reviews_block_chars + cc_block_chars + ranker_chars
        + history_chars + themes_chars + macro_chars
    )
    approx_input_tokens = total_input_chars // 4
    logger.info(
        "Rebalancer input estimate: %d total chars (~%d tokens). "
        "Breakdown: reviews=%d, cc_block=%d, ranker=%d, history=%d, "
        "themes=%d, macro=%d. (Opus 4.7 input limit: 200,000 tokens.)",
        total_input_chars, approx_input_tokens,
        reviews_block_chars, cc_block_chars, ranker_chars,
        history_chars, themes_chars, macro_chars,
    )
    if approx_input_tokens > 150_000:
        logger.warning(
            "Rebalancer input is approaching the 200K-token context "
            "limit (estimated %d tokens). Consider reducing the number "
            "of holdings reviewed, or shortening reviewer.full_text.",
            approx_input_tokens,
        )


def apply_cc_plan_validation(
    plan: RebalancePlan,
    *,
    chains: dict[str, object],
    eligibility: dict[str, Any],
    cc_context_block: str,
) -> tuple[RebalancePlan, list[str]]:
    from .cc_backfill import backfill_option_writes
    from .cc_validation import validate_option_writes

    plan = backfill_option_writes(plan, chains=chains)
    plan, cc_warnings = validate_option_writes(plan, eligibility=eligibility)
    for w in cc_warnings:
        logger.warning("CC plan validation: %s", w)

    n_write_calls = sum(1 for a in plan.actions if a.action == "WRITE_CALL")
    if n_write_calls > 0:
        total_premium = sum(
            ow.contracts * ow.est_premium_per_share * 100.0
            for ow in plan.option_writes
        )
        logger.info(
            "CC validation passed: %d WRITE_CALL action(s), "
            "$%s gross premium estimated. Details:",
            n_write_calls, f"{total_premium:,.0f}",
        )
        for ow in plan.option_writes:
            contract_premium = ow.contracts * ow.est_premium_per_share * 100.0
            logger.info(
                "  - %s: %d contracts @ $%.2f strike, expires %s, "
                "Δ=%.2f, ~$%s premium, assignment %.0f%%",
                ow.ticker, ow.contracts, ow.strike, ow.expiry,
                ow.delta, f"{contract_premium:,.0f}",
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
