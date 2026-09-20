"""Wholly worthless securities — the loss the harvester can never find.

`tax_harvest.find_harvest_candidates` skips any position whose price is
zero or missing, which is exactly what a dead listing looks like: the
larger the loss, the more certainly it is dropped. After the Schwab
reconnect two such positions surfaced, reported by CUSIP because there is
no ticker left to report — Taronis Technologies (SEC registration revoked
2023) and Taronis Fuels (bankrupt 2024), several thousand dollars of
basis between them.

The guard that separates a dead company from a data outage is the
symbol, not the price: a listed ticker quoting $0 is a feed that failed
today, and proposing a total loss on it would be wrong. A holding the
broker can only name by CUSIP, or that it marks as a non-market
instrument, has no listing to quote.

This is a NOTE, not an action. A worthless security usually cannot be
sold — there is no bid — and under IRC §165(g) it is treated as sold for
nothing on the last day of the tax year it became worthless, which is
generally an earlier year than the one the planner is running in. So
what the report owes the reader is the position, the basis, and the fact
that the year is theirs (and their preparer's) to establish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..data.brokerage import is_listed_symbol

# Below this, a position is worth nothing: brokers carry a dead holding at
# $0.00, and a few cents of rounding should not read as a live market.
RESIDUAL_VALUE_USD = 1.0
# Basis worth telling someone about. A $12 stub is not worth an amended
# return.
MIN_BASIS_USD = 100.0


@dataclass(frozen=True)
class WorthlessPosition:
    symbol: str  # as the broker reports it — usually a CUSIP, not a ticker
    account: str
    units: float
    cost_basis_usd: float
    value_usd: float
    reason: str

    @property
    def loss_usd(self) -> float:
        """Negative: the whole basis, less whatever residue it still carries."""
        return round(self.value_usd - self.cost_basis_usd, 2)


def find_worthless_positions(
    holdings: dict[str, list[dict[str, Any]]],
    account_meta: dict[str, dict[str, Any]],
    *,
    min_basis_usd: float = MIN_BASIS_USD,
) -> list[WorthlessPosition]:
    """Positions in TAXABLE accounts that appear to be wholly worthless.

    Tax-advantaged accounts are excluded for the same reason they are
    excluded from harvesting: a loss inside an IRA or HSA has no tax
    value. Biggest basis first.
    """
    out: list[WorthlessPosition] = []
    for account, items in holdings.items():
        meta = account_meta.get(account) or {}
        if (meta.get("tax_status") or "taxable") != "taxable":
            continue
        for h in items:
            symbol = str(h.get("ticker") or "").strip()
            units = float(h.get("units") or 0)
            if not symbol or units <= 0:
                continue
            if is_listed_symbol(symbol, h.get("kind")):
                continue  # a quotable ticker at $0 is a broken feed, not a dead company
            value = units * float(h.get("price") or 0)
            basis = units * float(h.get("average_purchase_price") or 0)
            if value >= RESIDUAL_VALUE_USD or basis < min_basis_usd:
                continue
            out.append(
                WorthlessPosition(
                    symbol=symbol,
                    account=account,
                    units=units,
                    cost_basis_usd=round(basis, 2),
                    value_usd=round(value, 2),
                    reason=_reason(symbol, h.get("kind")),
                )
            )
    return sorted(out, key=lambda p: -p.cost_basis_usd)


def _reason(symbol: str, kind: str | None) -> str:
    looks_like_cusip = len(symbol) == 9 and symbol[:8].isalnum() and not symbol.isalpha()
    if looks_like_cusip:
        return f"carried by CUSIP {symbol} — the listing it had is gone"
    if str(kind or "").strip().lower() in {"other", "crypto"}:
        return f"held as a {str(kind).strip().lower()} instrument, not a listed security"
    return "no market listing"


def worthless_report_data(positions: list[WorthlessPosition]) -> list[dict[str, Any]]:
    """Rows for the renderer."""
    return [
        {
            "symbol": p.symbol,
            "account": p.account,
            "units": p.units,
            "cost_basis_usd": p.cost_basis_usd,
            "value_usd": p.value_usd,
            "loss_usd": p.loss_usd,
            "reason": p.reason,
        }
        for p in positions
    ]
