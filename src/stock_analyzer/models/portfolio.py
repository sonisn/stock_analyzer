"""Pydantic models for portfolio holdings, tax lots, and covered-call
eligibility.

Includes the tax-lot domain (``Lot`` + ``TickerTaxSummary``) used by
the SnapTrade transaction-history pipeline, and the covered-call
eligibility decompositions (``EligibleHolding``, ``RoundLotCoverage``,
``IvHvRegime``) used by the rebalancer's CC extension.

The frozen ``TickerTaxSummary`` is the public type passed across the
codebase. The aggregation loop in ``data/transactions.py`` needs to
mutate counters as it walks activity rows; it does so through the
``TickerTaxSummaryMut`` subclass (``frozen=False``) and freezes back
into ``TickerTaxSummary`` once the per-ticker aggregate is complete.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

# 365 days = long-term holding period for US capital gains.
LONG_TERM_DAYS = 365


class _LoggerLike(Protocol):
    """Minimal logger protocol — just ``.debug(fmt, *args)``.

    Keeps this module free of any concrete logging import so it stays
    pure data and the I/O layer injects its own logger."""

    def debug(self, msg: str, *args: Any) -> None: ...


class Lot(BaseModel):
    """A single BUY transaction — i.e. one tax lot."""

    model_config = ConfigDict(frozen=True)

    date: str          # ISO date of purchase
    units: float       # shares acquired
    price: float       # per-share purchase price
    total_cost: float  # units * price + fee
    fee: float
    days_held: int
    is_long_term: bool
    account: str

    @classmethod
    def from_activity(
        cls,
        activity: dict[str, Any],
        account_name: str,
        today: date,
        *,
        coerce_date: Callable[[Any], date | None],
        logger: _LoggerLike,
    ) -> Lot | None:
        try:
            d = coerce_date(
                activity.get("trade_date") or activity.get("settlement_date")
            )
            if d is None:
                return None
            units = float(activity.get("units") or 0)
            price = float(activity.get("price") or 0)
            fee = float(activity.get("fee") or 0)
            if units <= 0 or price <= 0:
                return None
            days_held = (today - d).days
            return cls(
                date=d.isoformat(),
                units=units,
                price=price,
                total_cost=units * price + fee,
                fee=fee,
                days_held=days_held,
                is_long_term=days_held >= LONG_TERM_DAYS,
                account=account_name,
            )
        except (ValueError, TypeError) as e:
            logger.debug("Could not parse activity: %s", e)
            return None


class TickerTaxSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    ticker: str
    lots: list[Lot] = Field(default_factory=list)
    total_units_bought: float = 0.0
    total_units_sold: float = 0.0
    total_cost_basis: float = 0.0
    short_term_lot_count: int = 0
    long_term_lot_count: int = 0
    short_term_units: float = 0.0
    long_term_units: float = 0.0
    # SELL transactions within the last 60 days — used by the rebalancer
    # for wash-sale awareness (re-buying within 30 days of a loss-sale
    # disallows the loss for tax purposes).
    recent_sells_60d: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def current_units(self) -> float:
        return self.total_units_bought - self.total_units_sold

    @property
    def avg_cost(self) -> float:
        return (
            self.total_cost_basis / self.total_units_bought
            if self.total_units_bought
            else 0
        )

    def to_payload(self) -> dict[str, Any]:
        """Dict suitable for inclusion in the reviewer JSON payload."""
        # Sort lots newest-first so most-recent (shortest-held) appear at top.
        lots_sorted = sorted(self.lots, key=lambda x: x.date, reverse=True)
        return {
            "current_units_held": self.current_units,
            "total_units_bought": self.total_units_bought,
            "total_units_sold": self.total_units_sold,
            "average_cost_basis_per_share": round(self.avg_cost, 4),
            "lot_count": len(self.lots),
            "short_term_lots": self.short_term_lot_count,
            "long_term_lots": self.long_term_lot_count,
            "short_term_units": self.short_term_units,
            "long_term_units": self.long_term_units,
            "lots": [
                {
                    "date": lot.date,
                    "units": lot.units,
                    "price_per_share": round(lot.price, 4),
                    "total_cost": round(lot.total_cost, 2),
                    "days_held": lot.days_held,
                    "treatment": "long_term" if lot.is_long_term else "short_term",
                    "account": lot.account,
                }
                for lot in lots_sorted
            ],
            # Wash-sale flag data. Compare sale_price to avg_cost to estimate
            # whether the sell was at a loss; the LLM uses this to avoid
            # recommending re-purchase within 30 days.
            "recent_sells_60d": sorted(
                self.recent_sells_60d,
                key=lambda x: x.get("date", ""),
                reverse=True,
            ),
        }


class TickerTaxSummaryMut(TickerTaxSummary):
    """Mutable variant used inside the SnapTrade aggregation loop.

    The public type passed around the codebase stays frozen; this
    subclass only exists so ``data/transactions.py`` can increment
    counters as it walks activities, then convert to the immutable
    parent for downstream consumers."""

    model_config = ConfigDict(frozen=False)


# --- Covered-call eligibility ---------------------------------------------


class EligibleHolding(BaseModel):
    """A (ticker, account) pair eligible to write covered calls against.

    Each entry represents one specific brokerage account. The same
    ticker can appear in multiple EligibleHolding entries when round-lot
    shares sit in more than one account."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    account: str
    tax_status: Literal["taxable", "tax_advantaged"] = "taxable"
    shares_held: int
    open_short_call_contracts: int
    available_shares: int   # shares_held - 100 × open_short_call_contracts
    max_contracts: int      # available_shares // 100


class RoundLotCoverage(BaseModel):
    """Round-lot decomposition of a single holding.

    Used by the stub-consolidation prompt rule and by the reporting
    layer's ``RoundLotCoverage`` section.
    """

    model_config = ConfigDict(frozen=True)

    ticker: str
    shares: int
    round_lots: int
    stub_shares: int             # shares - round_lots × 100
    stub_dollar_value: float     # stub_shares × spot (0 when spot unknown)
    to_next_lot_shares: int      # (100 - stub_shares) if stub_shares else 0
    to_next_lot_cost: float      # to_next_lot_shares × spot


class IvHvRegime(BaseModel):
    """IV-vs-realized-vol regime for one ticker (free IVR proxy)."""

    model_config = ConfigDict(frozen=True)

    ticker: str
    current_iv: float         # representative chain IV, e.g. 0.32
    hv_annualized: float      # 252-day realized vol, e.g. 0.27
    iv_hv_ratio: float        # current_iv / hv
    label: str                # "elevated" | "average" | "depressed"


__all__ = [
    "LONG_TERM_DAYS",
    "Lot",
    "TickerTaxSummary",
    "TickerTaxSummaryMut",
    "EligibleHolding",
    "RoundLotCoverage",
    "IvHvRegime",
]
