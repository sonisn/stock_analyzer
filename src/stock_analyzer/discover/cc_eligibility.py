"""Pure-Python eligibility, round-lot coverage, earnings filter, and
prompt-context assembly for the covered-call extension to the
rebalancer.

No I/O here — every function is testable as a pure transformation of
its inputs. CLI wiring (`cli/rebalance.py`) is responsible for fetching
holdings, chains, open short-call positions, and earnings dates, then
passing them in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from ..data.options_chain import OptionChain, OptionQuote
from .schemas import HoldingReview

if TYPE_CHECKING:
    from ..data.historical_volatility import RealizedVolatility


@dataclass(frozen=True)
class EligibleHolding:
    """A position that's eligible to write covered calls against."""
    ticker: str
    shares_held: int
    open_short_call_contracts: int
    available_shares: int   # shares_held - 100 × open_short_call_contracts
    max_contracts: int      # available_shares // 100


def eligible_holdings(
    positions: dict[str, dict[str, float]],
    *,
    open_short_calls: dict[str, int],
    denylist: tuple[str, ...],
) -> dict[str, EligibleHolding]:
    """Return {ticker: EligibleHolding} for every position that:
      - holds ≥ 100 shares
      - has ≥ 100 shares NOT already collateralizing an open short call
      - is not in `denylist`

    `positions` matches the shape produced by `_aggregate_positions`
    in `cli/rebalance.py` — {ticker: {"units": float, ...}}.
    """
    denyset = {t.upper() for t in denylist}
    out: dict[str, EligibleHolding] = {}
    for ticker, pos in positions.items():
        if ticker.upper() in denyset:
            continue
        shares = int(pos.get("units") or 0)
        if shares < 100:
            continue
        short_contracts = int(open_short_calls.get(ticker, 0))
        available = shares - 100 * short_contracts
        if available < 100:
            continue
        out[ticker] = EligibleHolding(
            ticker=ticker, shares_held=shares,
            open_short_call_contracts=short_contracts,
            available_shares=available,
            max_contracts=available // 100,
        )
    return out


@dataclass(frozen=True)
class RoundLotCoverage:
    """Round-lot decomposition of a single holding.

    Used by the stub-consolidation prompt rule and by the reporting
    layer's `RoundLotCoverage` section.
    """
    ticker: str
    shares: int
    round_lots: int
    stub_shares: int             # shares - round_lots × 100
    stub_dollar_value: float     # stub_shares × spot (0 when spot unknown)
    to_next_lot_shares: int      # (100 - stub_shares) if stub_shares else 0
    to_next_lot_cost: float      # to_next_lot_shares × spot


def round_lot_coverage(
    positions: dict[str, dict[str, float]],
    *,
    spots: dict[str, float],
) -> dict[str, RoundLotCoverage]:
    """Compute round-lot / stub decomposition for every held ticker.

    `spots` is the current price per ticker (from the technicals stage).
    Missing spots collapse dollar values to 0 — the report layer can
    still show share counts even when price data is stale.
    """
    out: dict[str, RoundLotCoverage] = {}
    for ticker, pos in positions.items():
        shares = int(pos.get("units") or 0)
        if shares <= 0:
            continue
        round_lots = shares // 100
        stub = shares - round_lots * 100
        spot = float(spots.get(ticker) or 0.0)
        to_next_shares = (100 - stub) if stub else 0
        out[ticker] = RoundLotCoverage(
            ticker=ticker, shares=shares,
            round_lots=round_lots, stub_shares=stub,
            stub_dollar_value=stub * spot,
            to_next_lot_shares=to_next_shares,
            to_next_lot_cost=to_next_shares * spot,
        )
    return out


EARNINGS_BLACKLIST_DAYS = 7


def apply_earnings_filter(
    chain: OptionChain,
    *,
    earnings_date: date | None,
) -> tuple[OptionChain, tuple[date, date] | None]:
    """Drop expiries that fall within ±EARNINGS_BLACKLIST_DAYS of
    earnings_date. Returns the filtered chain and the blacklist window
    (for prompt display) or None when no earnings date was provided.
    """
    if earnings_date is None:
        return chain, None
    lo = earnings_date - timedelta(days=EARNINGS_BLACKLIST_DAYS)
    hi = earnings_date + timedelta(days=EARNINGS_BLACKLIST_DAYS)
    survived = [q for q in chain.calls if q.expiry < lo or q.expiry > hi]
    return (
        OptionChain(
            ticker=chain.ticker, spot=chain.spot, asof=chain.asof,
            calls=survived, source=chain.source,
        ),
        (lo, hi),
    )


@dataclass(frozen=True)
class IvHvRegime:
    """IV-vs-realized-vol regime for one ticker (free IVR proxy)."""
    ticker: str
    current_iv: float         # representative chain IV, e.g. 0.32
    hv_annualized: float      # 252-day realized vol, e.g. 0.27
    iv_hv_ratio: float        # current_iv / hv
    label: str                # "elevated" | "average" | "depressed"


def _representative_iv_from_chain(chain: OptionChain | None) -> float | None:
    """Average `iv` across all OTM call rows in the chain (calls within
    our band, near-ATM-weighted by inclusion). Returns None when no
    rows have IV data."""
    if chain is None or not chain.calls:
        return None
    ivs = [q.iv for q in chain.calls if q.iv is not None and q.iv > 0]
    if not ivs:
        return None
    return sum(ivs) / len(ivs)


def _label_iv_hv_ratio(ratio: float) -> str:
    if ratio >= 1.20:
        return "elevated"
    if ratio >= 0.90:
        return "average"
    return "depressed"


def compute_iv_hv_regime(
    *,
    chain: OptionChain | None,
    hv: RealizedVolatility | None,
) -> IvHvRegime | None:
    """Pair a chain's representative IV with a realized-vol estimate
    to produce a regime label. Returns None when either input is
    missing or yields a degenerate ratio."""
    if chain is None or hv is None or hv.hv_annualized <= 0:
        return None
    iv = _representative_iv_from_chain(chain)
    if iv is None or iv <= 0:
        return None
    ratio = iv / hv.hv_annualized
    if math.isnan(ratio) or math.isinf(ratio) or ratio <= 0:
        return None
    return IvHvRegime(
        ticker=chain.ticker,
        current_iv=iv,
        hv_annualized=hv.hv_annualized,
        iv_hv_ratio=ratio,
        label=_label_iv_hv_ratio(ratio),
    )


_CHAIN_ROW_CAP_PER_TICKER = 8
_CC_CONTEXT_BLOCK_MAX_CHARS = 50_000  # ~12.5K tokens — safe margin under 200K context.


def _format_chain_row(q: OptionQuote) -> str:
    """Single-line chain row used inside the per-ticker context block.

    Coerces NaN/inf numeric fields to a "-" sentinel so the LLM sees
    clean text. yfinance occasionally returns NaN for low-volume strikes.
    """

    def _f(v: float | None, fmt: str) -> str:
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            return "—"
        return format(v, fmt)

    delta_str = f"Δ {_f(q.delta, '.2f')}"
    iv_str = f"IV {_f(q.iv, '.2f')}"
    oi_str = f"OI {q.open_interest}" if q.open_interest else "OI —"
    return (
        f"    {q.expiry.isoformat()} ${_f(q.strike, '>6.2f')} strike  "
        f"bid {_f(q.bid, '.2f')} / ask {_f(q.ask, '.2f')}  "
        f"{delta_str}  {iv_str}  {oi_str}"
    )


def _format_ticker_block(
    *, ticker: str,
    review: HoldingReview | str | None,
    eligible: EligibleHolding,
    chain: OptionChain | None,
    earnings_date: date | None,
    iv_hv: IvHvRegime | None = None,
) -> str:
    lines: list[str] = [f"TICKER: {ticker}"]
    if isinstance(review, HoldingReview):
        verdict_line = (
            f"  Reviewer verdict:        {review.verdict} "
            f"(confidence {review.confidence}/10)"
        )
    else:
        verdict_line = "  Reviewer verdict:        UNKNOWN"
    lines.append(verdict_line)
    lines.append(f"  Shares held:             {eligible.shares_held}")
    if eligible.open_short_call_contracts:
        plural = "s" if eligible.open_short_call_contracts > 1 else ""
        lines.append(
            f"  Available for CC:        {eligible.available_shares} "
            f"({100 * eligible.open_short_call_contracts} already "
            f"collateralizing open short call{plural})"
        )
    else:
        lines.append(f"  Available for CC:        {eligible.available_shares}")
    if earnings_date is not None:
        lo = earnings_date - timedelta(days=EARNINGS_BLACKLIST_DAYS)
        hi = earnings_date + timedelta(days=EARNINGS_BLACKLIST_DAYS)
        lines.append(
            f"  Earnings-blacklist:      {earnings_date.isoformat()} "
            f"(skip expiries {lo.isoformat()} .. {hi.isoformat()})"
        )
    else:
        lines.append(
            "  Earnings-blacklist:      earnings_unknown — be conservative on DTE"
        )
    if iv_hv is not None:
        lines.append(
            f"  IV/HV regime:            IV {iv_hv.current_iv * 100:.0f}%  "
            f"HV-252d {iv_hv.hv_annualized * 100:.0f}%  "
            f"ratio {iv_hv.iv_hv_ratio:.2f}x  ({iv_hv.label})"
        )
    else:
        lines.append(
            "  IV/HV regime:            unknown (insufficient data)"
        )
    if not isinstance(chain, OptionChain) or chain.source == "missing" or not chain.calls:
        lines.append("  Option chain: UNAVAILABLE")
    else:
        lines.append("  Option chain (OTM calls):")
        for q in chain.calls[:_CHAIN_ROW_CAP_PER_TICKER]:
            lines.append(_format_chain_row(q))
    return "\n".join(lines)


def build_cc_context_block(
    *,
    eligible: dict[str, EligibleHolding],
    chains: dict[str, OptionChain],
    coverage: dict[str, RoundLotCoverage],
    reviews: dict[str, HoldingReview | str],
    earnings: dict[str, date],
    stub_pool_total_usd: float,
    iv_hv_regimes: dict[str, IvHvRegime] | None = None,
) -> str:
    """Compose the COVERED-CALL CONTEXT block consumed by the rebalancer
    prompt. Returns the empty string when no positions are eligible
    (in which case the rebalancer prompt simply doesn't include a CC
    section)."""
    if not eligible:
        return ""

    per_ticker: list[str] = []
    for ticker in sorted(eligible):
        per_ticker.append(_format_ticker_block(
            ticker=ticker,
            review=reviews.get(ticker),
            eligible=eligible[ticker],
            chain=chains.get(ticker),
            earnings_date=earnings.get(ticker),
            iv_hv=(iv_hv_regimes or {}).get(ticker),
        ))

    rlc_lines: list[str] = [
        "",
        "ROUND-LOT COVERAGE (every holding, for stub-consolidation reasoning):",
        f"  {'Position':<8} {'Shares':>6} {'Round lots':>10} {'Stub':>5} "
        f"{'Stub $':>12} {'To-next-lot':>14}",
    ]
    for ticker in sorted(coverage):
        rec = coverage[ticker]
        rlc_lines.append(
            f"  {ticker:<8} {rec.shares:>6d} "
            f"{rec.round_lots:>4d} ({rec.round_lots * 100:>3d}) "
            f"{rec.stub_shares:>5d} "
            f"${rec.stub_dollar_value:>10,.0f} "
            f"${rec.to_next_lot_cost:>13,.0f}"
        )
    rlc_lines.append(f"  Stub pool total: ${stub_pool_total_usd:,.0f}")

    header = "=" * 70 + "\nCOVERED-CALL CONTEXT\n" + "=" * 70
    result = header + "\n\n" + "\n\n".join(per_ticker) + "\n" + "\n".join(rlc_lines)
    if len(result) > _CC_CONTEXT_BLOCK_MAX_CHARS:
        # Defensive: shouldn't happen if cli/rebalance trimmed eligible to
        # _CC_MAX_ELIGIBLE_FOR_PROMPT, but a single position with many earnings-
        # blacklisted strikes or unusually verbose reviewer text could still
        # push us over. Truncate with a visible marker so the LLM knows the
        # block was cut.
        truncated = result[:_CC_CONTEXT_BLOCK_MAX_CHARS]
        marker = f"\n\n[TRUNCATED — context exceeded {_CC_CONTEXT_BLOCK_MAX_CHARS:,}-char budget]"
        return truncated + marker
    return result
