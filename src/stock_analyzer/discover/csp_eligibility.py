"""Pure-Python eligibility and prompt-context assembly for cash-secured
puts (CSPs) — the front half of the wheel.

A CSP is sold on a stock you'd like to own but don't yet: you collect the
premium now, and only buy (at the strike, below today's price) if the put
is assigned. Candidates are recent discover picks that aren't already a
round-lot holding (those route to covered calls instead).

No I/O here. `discover/rebalance_csp.py` fetches picks, positions, open
short puts, chains and earnings dates, then passes them in.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

from ..models.market import OptionChain, OptionQuote
from ..models.portfolio import CspCandidate
from .cc_eligibility import EARNINGS_BLACKLIST_DAYS, _format_chain_row

__all__ = [
    "CspCandidate",
    "eligible_csp_tickers",
    "put_delta",
    "fill_put_deltas",
    "puts_near_band",
    "build_csp_context_block",
]

# Thesis-tracker statuses that disqualify a pick: the reason to own it is
# gone (BROKEN) or already paid out (TARGET HIT).
_EXCLUDED_THESIS = {"BROKEN", "TARGET HIT"}

# Risk-free rate for the Black-Scholes delta fallback. Delta is not very
# sensitive to it at 30-45 DTE; a rough current T-bill yield is enough.
_RISK_FREE_RATE = 0.04

# Puts shown per ticker, and per expiry within that (so a few expiries
# are visible rather than every strike of the nearest one).
_PUT_ROW_CAP_PER_TICKER = 8
_PUT_ROW_CAP_PER_EXPIRY = 3
# Same rounding slack on the |Δ| band as csp_validation applies.
_DELTA_TOLERANCE = 0.01


def eligible_csp_tickers(
    picks: list[tuple[str, int, str]],
    *,
    positions: dict[str, dict[str, float]],
    cash_budget: float,
    denylist: tuple[str, ...],
    open_short_puts: dict[str, dict[str, float]] | None = None,
    thesis_status: dict[str, str] | None = None,
    max_pct_per_put: float = 0.25,
    max_pct_total: float = 0.80,
    max_candidates: int = 8,
    covered_call_tickers: set[str] | None = None,
    max_account_room: float | None = None,
) -> dict[str, CspCandidate]:
    """Filter recent picks down to tickers you could sell a put on.

    `picks` is [(ticker, rank, run_at), ...] from any number of runs; a
    ticker picked more than once keeps its most recent run. Drops:
      - tickers already on the covered-call side: `covered_call_tickers`
        when given (100+ shares in ONE account), else >= 100 shares in total.
        Shares split across accounts (e.g. 60 + 50) can't back a call, so
        they stay put candidates when the covered-call set is known
      - tickers in `denylist`
      - tickers that already have a short put open (don't stack)
      - picks whose thesis is BROKEN or whose target is already hit
    Everything is dropped when `cash_budget` <= 0: a put that isn't fully
    cash-secured is a margin trade, not this strategy.

    At most `max_candidates` are kept, newest run first, then best rank.
    """
    if cash_budget <= 0:
        return {}
    denyset = {t.upper() for t in denylist}
    open_puts = open_short_puts or {}
    thesis = thesis_status or {}

    latest: dict[str, tuple[str, int]] = {}
    for ticker, rank, run_at in picks:
        t = ticker.upper()
        prev = latest.get(t)
        if prev is None or (run_at, -rank) > (prev[0], -prev[1]):
            latest[t] = (run_at, rank)

    per_put_cap = cash_budget * max_pct_per_put
    total_cap = cash_budget * max_pct_total
    # One put's collateral sits in one account: it can't exceed the
    # roomiest account's cash, however large the total.
    if max_account_room is not None:
        per_put_cap = min(per_put_cap, max_account_room)
    ordered = sorted(latest.items(), key=lambda kv: (kv[1][0], -kv[1][1]), reverse=True)
    out: dict[str, CspCandidate] = {}
    for ticker, (run_at, rank) in ordered:
        if ticker in denyset or ticker in open_puts:
            continue
        status = thesis.get(ticker)
        if status in _EXCLUDED_THESIS:
            continue
        shares = int((positions.get(ticker) or {}).get("units") or 0)
        on_cc_side = (
            ticker in covered_call_tickers if covered_call_tickers is not None else shares >= 100
        )
        if on_cc_side:
            continue
        out[ticker] = CspCandidate(
            ticker=ticker,
            last_pick_run_at=run_at[:10],
            last_pick_rank=rank,
            shares_held=max(shares, 0),
            thesis_status=status,
            max_csp_cash=round(min(per_put_cap, total_cap), 2),
        )
        if len(out) >= max_candidates:
            break
    return out


def put_delta(
    *, spot: float, strike: float, iv: float, days: int, rate: float = _RISK_FREE_RATE
) -> float | None:
    """Black-Scholes put delta, N(d1) - 1 (negative). None when an input
    makes it undefined. Used only when the chain carries no Greeks
    (yfinance); Tradier's own delta is always preferred."""
    if spot <= 0 or strike <= 0 or iv <= 0 or days <= 0:
        return None
    t = days / 365.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * iv * iv) * t) / (iv * math.sqrt(t))
    n_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    return n_d1 - 1.0


def fill_put_deltas(chain: OptionChain, *, today: date | None = None) -> OptionChain:
    """Estimate delta from IV for put rows that have none."""
    today = today or date.today()
    if not any(q.delta is None for q in chain.puts):
        return chain
    filled: list[OptionQuote] = []
    for q in chain.puts:
        if q.delta is None and q.iv:
            est = put_delta(spot=chain.spot, strike=q.strike, iv=q.iv, days=(q.expiry - today).days)
            if est is not None:
                q = q.model_copy(update={"delta": round(est, 3)})
        filled.append(q)
    return chain.model_copy(update={"puts": filled})


def puts_near_band(
    chain: OptionChain,
    *,
    delta_min: float,
    delta_max: float,
    max_collateral: float | None = None,
) -> list[OptionQuote]:
    """Put rows the plan could actually use: |Δ| inside the target band
    and, when `max_collateral` is given, one contract's collateral
    (strike × 100) within it. Nearest expiry first, highest strike first,
    at most a few rows per expiry. Rows without a delta are dropped —
    the band can't be checked."""
    lo, hi = delta_min - _DELTA_TOLERANCE, delta_max + _DELTA_TOLERANCE
    rows = [
        q
        for q in chain.puts
        if q.delta is not None
        and lo <= abs(q.delta) <= hi
        and (max_collateral is None or q.strike * 100.0 <= max_collateral)
    ]
    rows.sort(key=lambda q: (q.expiry, -q.strike))
    out: list[OptionQuote] = []
    per_expiry: dict[date, int] = {}
    for q in rows:
        if per_expiry.get(q.expiry, 0) >= _PUT_ROW_CAP_PER_EXPIRY:
            continue
        per_expiry[q.expiry] = per_expiry.get(q.expiry, 0) + 1
        out.append(q)
        if len(out) >= _PUT_ROW_CAP_PER_TICKER:
            break
    return out


def _format_candidate_block(
    c: CspCandidate,
    *,
    chain: OptionChain,
    rows: list[OptionQuote],
    earnings_date: date | None,
) -> str:
    held = f", you hold {c.shares_held} shares" if c.shares_held else ""
    thesis = f", thesis {c.thesis_status}" if c.thesis_status else ""
    lines = [
        f"TICKER: {c.ticker}  (picked rank {c.last_pick_rank} on "
        f"{c.last_pick_run_at}{thesis}{held})",
        f"  Spot:                    ${chain.spot:,.2f}",
        f"  Max collateral per put:  ${c.max_csp_cash:,.0f}",
    ]
    if earnings_date is not None:
        lo = earnings_date - timedelta(days=EARNINGS_BLACKLIST_DAYS)
        hi = earnings_date + timedelta(days=EARNINGS_BLACKLIST_DAYS)
        lines.append(
            f"  Earnings-blacklist:      {earnings_date.isoformat()} "
            f"(expiries {lo.isoformat()} .. {hi.isoformat()} already removed)"
        )
    else:
        lines.append("  Earnings-blacklist:      earnings_unknown — be conservative on DTE")
    lines.append("  Option chain (OTM puts, Δ is negative):")
    lines.extend(_format_chain_row(q) for q in rows)
    return "\n".join(lines)


def build_csp_context_block(
    *,
    candidates: dict[str, CspCandidate],
    chains: dict[str, OptionChain],
    earnings: dict[str, date],
    cash_budget: float,
    open_put_collateral: float,
    account_room: dict[str, float] | None = None,
    delta_min: float,
    delta_max: float,
    max_pct_per_put: float,
    max_pct_total: float,
) -> str:
    """Compose the CASH-SECURED PUT CONTEXT block for the rebalancer.
    Candidates with no in-band put that one contract of fits the per-put
    cap are left out; returns "" when none are left."""
    blocks: list[str] = []
    for ticker, c in candidates.items():
        chain = chains.get(ticker)
        if chain is None or chain.source == "missing" or chain.spot <= 0:
            continue
        rows = puts_near_band(
            chain, delta_min=delta_min, delta_max=delta_max, max_collateral=c.max_csp_cash
        )
        if not rows:
            continue
        blocks.append(
            _format_candidate_block(c, chain=chain, rows=rows, earnings_date=earnings.get(ticker))
        )
    if not blocks:
        return ""
    existing = (
        f"  Already reserved by open short puts: ${open_put_collateral:,.0f} "
        "(excluded from the budget below)\n"
        if open_put_collateral > 0
        else ""
    )
    header = (
        "=" * 70
        + "\nCASH-SECURED PUT CONTEXT\n"
        + "=" * 70
        + "\n"
        + existing
        + f"  Cash budget for puts:    ${cash_budget:,.0f}\n"
        + f"  Per-put collateral cap:  ${cash_budget * max_pct_per_put:,.0f} "
        f"({max_pct_per_put:.0%})\n"
        + f"  Total collateral cap:    ${cash_budget * max_pct_total:,.0f} "
        f"({max_pct_total:.0%})"
    )
    if account_room:
        header += "\n  Cash per account that can secure puts (collateral stays in ONE account):"
        for name, room in sorted(account_room.items(), key=lambda kv: -kv[1]):
            header += f"\n    {name}: ${room:,.0f}"
    return header + "\n\n" + "\n\n".join(blocks)
