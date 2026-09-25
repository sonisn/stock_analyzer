"""Rolling a covered call that has run at the shares.

A written call that the stock has caught up with poses one question:
give up the shares, or pay to keep them. Rolling is the third answer —
buy the near call back and sell a further one, ideally for a net credit,
which raises the cap and collects more premium in the same trade.

On 2026-09-20 TSLA sat 10% under a $400 strike expiring 2026-12-18 with
all 200 shares committed. "Roll it up and out" is the right instruction
and a useless one on its own: at which strike, into which expiry, and
does it still pay after buying the near call back? That arithmetic is
deterministic, so it is done here rather than asked of a model.

Two rules survive from the writing policy, because a roll is a new call:
the replacement strike must clear `min_upside_pct` above spot, and its
delta must sit under the same ceiling. A roll that only buys a little
room is not worth paying for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

from ..logging import get_logger
from ..models.market import OptionChain, OptionQuote

logger = get_logger(__name__)

# A roll has to buy meaningfully more room than it costs. Raising a cap
# by a couple of percent is churn dressed up as risk management.
MIN_STRIKE_GAIN_PCT = 5.0


@dataclass(frozen=True)
class RollCandidate:
    """One replacement call, priced against buying the current one back."""

    ticker: str
    account: str
    contracts: int
    from_strike: float
    from_expiry: date
    to_strike: float
    to_expiry: date
    buyback_per_share: float
    credit_per_share: float
    spot: float
    delta: float | None

    @property
    def net_per_share(self) -> float:
        """Positive = paid to roll; negative = it costs money."""
        return self.credit_per_share - self.buyback_per_share

    @property
    def net_usd(self) -> float:
        return self.net_per_share * 100 * self.contracts

    @property
    def strike_gain_pct(self) -> float:
        return (self.to_strike / self.from_strike - 1) * 100

    @property
    def upside_pct(self) -> float:
        return (self.to_strike / self.spot - 1) * 100

    @property
    def extra_days(self) -> int:
        return (self.to_expiry - self.from_expiry).days

    def describe(self) -> str:
        money = (
            f"a net credit of ${self.net_usd:,.0f}"
            if self.net_per_share >= 0
            else f"a net cost of ${-self.net_usd:,.0f}"
        )
        longer = f", {self.extra_days} days further out" if self.extra_days > 0 else ""
        return (
            f"Roll the {self.contracts} {self.ticker} ${self.from_strike:,.0f} call(s) "
            f"({self.from_expiry.isoformat()}) up to ${self.to_strike:,.0f} "
            f"{self.to_expiry.isoformat()}{longer} for {money}. That lifts the cap from "
            f"{(self.from_strike / self.spot - 1) * 100:+.0f}% to {self.upside_pct:+.0f}% "
            f"above today's ${self.spot:,.2f} and keeps the shares, capped until "
            f"{self.to_expiry.isoformat()}."
        )


def find_current_quote(chain: OptionChain, *, strike: float, expiry: date) -> OptionQuote | None:
    """The written call's own row, needed to price buying it back."""
    for quote in chain.calls:
        if abs(quote.strike - strike) < 0.01 and quote.expiry == expiry:
            return quote
    return None


def roll_candidates(
    *,
    ticker: str,
    account: str,
    contracts: int,
    from_strike: float,
    from_expiry: date,
    chain: OptionChain,
    min_upside_pct: float,
    delta_max: float,
    dte_max: int | None = None,
    today: date | None = None,
) -> list[RollCandidate]:
    """Replacement calls worth considering, best net credit first.

    `dte_max` bounds how far out a replacement may go. Rolling up for a
    credit gets easier the further out you sell, so without a bound the
    answer is always "sell a 2027 call" — which pays, and hands over the
    next year of decisions with it.

    Buying back is priced at the ASK and selling at the BID — the sides
    actually available — so a candidate that looks free at mid prices
    does not survive here. Returns [] when the current call has no quote
    to buy back at, since a roll cannot be costed without it.
    """
    today = today or date.today()
    spot = chain.spot
    if not spot or not chain.calls:
        return []
    current = find_current_quote(chain, strike=from_strike, expiry=from_expiry)
    if current is None or current.ask is None or current.ask <= 0:
        logger.info(
            "No quote for the open %s $%.0f %s call — cannot price a roll",
            ticker,
            from_strike,
            from_expiry,
        )
        return []
    buyback = current.ask

    floor = spot * (1 + min_upside_pct / 100)
    out: list[RollCandidate] = []
    for quote in chain.calls:
        if quote.expiry < from_expiry and quote.strike <= from_strike:
            continue  # neither further out nor higher up
        if dte_max is not None and (quote.expiry - today).days > dte_max:
            continue
        if quote.strike < floor:
            continue
        if quote.strike < from_strike * (1 + MIN_STRIKE_GAIN_PCT / 100):
            continue
        if delta_max is not None and quote.delta is not None and quote.delta > delta_max:
            continue
        credit = quote.bid
        if credit is None or credit <= 0:
            continue
        out.append(
            RollCandidate(
                ticker=ticker,
                account=account,
                contracts=contracts,
                from_strike=from_strike,
                from_expiry=from_expiry,
                to_strike=quote.strike,
                to_expiry=quote.expiry,
                buyback_per_share=buyback,
                credit_per_share=credit,
                spot=spot,
                delta=quote.delta,
            )
        )
    # Best paid first; among equals, the one that buys the most room.
    out.sort(key=lambda c: (-c.net_per_share, -c.strike_gain_pct))
    return out


def best_roll(candidates: list[RollCandidate]) -> RollCandidate | None:
    """The one worth naming: the SOONEST roll that still pays.

    Not the biggest credit. Credit rises with time to expiry, so ranking
    on it alone always answers "sell a 2028 call" — the most money and
    the longest surrender of decisions. The nearest expiry that covers
    the buy-back keeps the position free again sooner, and among equals
    the one that lifts the strike furthest wins.

    A roll that costs money is a real choice — it buys the upside back —
    but it is not "collect more premium", so it is only offered when
    nothing pays for itself.
    """
    if not candidates:
        return None
    paying = [c for c in candidates if c.net_per_share >= 0]
    if paying:
        return min(paying, key=lambda c: (c.to_expiry, -c.strike_gain_pct))
    return max(candidates, key=lambda c: c.net_per_share)


def roll_suggestion(
    *,
    ticker: str,
    obligation: dict[str, Any],
    chain: OptionChain,
    min_upside_pct: float,
    delta_max: float,
    dte_max: int | None = None,
    today: date | None = None,
) -> str:
    """One line of advice for a call the stock has run at, or "".

    Prefers a roll inside the normal writing window. Rolling up for a
    credit gets easier the further out you sell, so when nothing inside
    the window pays, the longer-dated one that does is offered with the
    lock-in stated rather than buried.

    `obligation` is one ticker's entry from
    `brokerage.fetch_covered_call_obligations`.
    """
    legs = (obligation or {}).get("legs") or []
    if not legs:
        return ""
    # The leg that decides the position is the nearest strike.
    leg = min(legs, key=lambda x: (x["strike"], x["expiry"]))
    try:
        expiry = date.fromisoformat(str(leg["expiry"]))
        args: dict[str, Any] = {
            "ticker": ticker,
            "account": str(leg.get("account") or ""),
            "contracts": int(leg.get("contracts") or 0),
            "from_strike": float(leg["strike"]),
            "from_expiry": expiry,
            "chain": chain,
            "min_upside_pct": min_upside_pct,
            "delta_max": delta_max,
            "today": today,
        }
    except KeyError, TypeError, ValueError:
        return ""

    in_window = best_roll(roll_candidates(**args, dte_max=dte_max))
    if in_window is not None and in_window.net_per_share >= 0:
        return in_window.describe()

    further = best_roll(roll_candidates(**args))
    if further is not None and further.net_per_share >= 0:
        window = f"the usual {dte_max}-day window" if dte_max else "a nearer expiry"
        return (
            f"Nothing in {window} pays for buying the {ticker} "
            f"${float(leg['strike']):,.0f} call back. {further.describe()}"
        )
    if in_window is not None:
        return (
            f"No roll on {ticker} pays for itself. The nearest that keeps the shares is "
            f"{in_window.describe()[0].lower() + in_window.describe()[1:]}"
        )
    return (
        f"No roll on {ticker} both clears the {min_upside_pct:.0f}% floor and pays for "
        f"the buy-back — holding the call to expiry or closing it are the choices."
    )


__all__ = [
    "MIN_STRIKE_GAIN_PCT",
    "RollCandidate",
    "best_roll",
    "find_current_quote",
    "roll_candidates",
    "roll_suggestion",
]
