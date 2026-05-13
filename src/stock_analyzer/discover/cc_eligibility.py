"""Pure-Python eligibility, round-lot coverage, earnings filter, and
prompt-context assembly for the covered-call extension to the
rebalancer.

No I/O here — every function is testable as a pure transformation of
its inputs. CLI wiring (`cli/rebalance.py`) is responsible for fetching
holdings, chains, open short-call positions, and earnings dates, then
passing them in.
"""
from __future__ import annotations

from dataclasses import dataclass


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
