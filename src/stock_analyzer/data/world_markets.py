"""World exchanges — the markets that trade before New York opens.

The macro context so far is US-only: FRED's yield curve, VIX and jobs
(`data/fred_macro.py`), plus relative strength against SPY. But a
portfolio this concentrated in semiconductors is priced overnight, in
other currencies, on other exchanges. Taiwan sets the tone for TSM and
for the foundry capacity behind NVDA and AVGO; Korea prices the memory
cycle; ASML trades in Amsterdam before the US open; a falling dollar
lifts the translated earnings of every US company selling abroad.

None of this is a reason to trade — these are 3-5 year holdings. It is
regime context: which regions are leading, which are breaking down, and
what that says about the demand behind a holding. So the trailing
windows matter more than the overnight move, and the overnight move is
reported last.

All free, from the same paced yfinance gateway everything else uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..logging import get_logger
from . import yf_gateway

logger = get_logger(__name__)


@dataclass(frozen=True)
class Market:
    symbol: str
    name: str
    region: str
    # Which holdings this market actually says something about. Keeping
    # it explicit is what makes the block worth reading: "Taiwan -2%" is
    # trivia until it is attached to TSM, AVGO and NVDA.
    bears_on: tuple[str, ...] = ()
    note: str = ""


# Ordered east to west, the way the trading day actually runs.
MARKETS: tuple[Market, ...] = (
    Market("^N225", "Nikkei 225", "Japan", ("AMAT", "TOELY"), "semicap equipment and yen carry"),
    Market("^KS11", "KOSPI", "South Korea", ("NVDA", "MRVL"), "memory/HBM cycle"),
    Market("^TWII", "Taiwan Weighted", "Taiwan", ("TSM", "NVDA", "AVGO"), "foundry capacity"),
    Market("^HSI", "Hang Seng", "Hong Kong", (), "China demand and export controls"),
    Market("000001.SS", "Shanghai Composite", "China", (), "China demand"),
    Market("^BSESN", "BSE Sensex", "India", (), ""),
    Market("^GDAXI", "DAX", "Germany", (), "European industrial demand"),
    Market("^FTSE", "FTSE 100", "UK", (), ""),
    Market("^STOXX50E", "Euro Stoxx 50", "Europe", ("ASML",), "ASML and EU industrials"),
    Market("^GSPC", "S&P 500", "US", (), ""),
    Market("^IXIC", "Nasdaq Composite", "US", (), ""),
    Market("^SOX", "Philadelphia Semiconductor", "US", ("NVDA", "AMD", "AVGO", "MRVL", "ARM")),
)

# Not exchanges, but they move the same holdings: the dollar translates
# foreign revenue, the yen funds the carry trade that unwinds into every
# risk asset, and copper is the cleanest read on industrial demand.
CROSS_ASSETS: tuple[Market, ...] = (
    Market("DX-Y.NYB", "US Dollar Index", "FX", (), "a stronger dollar shrinks foreign revenue"),
    Market("JPY=X", "USD/JPY", "FX", (), "carry-trade unwind risk"),
    Market("HG=F", "Copper", "Commodity", ("POWL",), "industrial and grid demand"),
    Market("CL=F", "Crude Oil (WTI)", "Commodity", (), ""),
)

# Trailing windows, in trading days. The 1-day column is context for what
# just happened; the rest is the regime.
_WINDOWS = {"1d": 1, "1mo": 21, "6mo": 126, "1y": 252}

# A move this big in one session is worth naming in the email; smaller
# ones are the noise a long-term holder should ignore.
NOTABLE_MOVE_PCT = 2.0
# A market this far below its own 1-year path has changed regime, not
# just had a bad week.
REGIME_BREAK_PCT = -10.0


def _pct_change(closes: Any, days: int) -> float | None:
    """Percent change over the last `days` sessions of a close series."""
    if closes is None or len(closes) <= days:
        return None
    last, prior = closes.iloc[-1], closes.iloc[-1 - days]
    if not prior:
        return None
    return float((last / prior - 1) * 100)


def _fetch_one(market: Market) -> dict[str, Any] | None:
    frame = yf_gateway.ticker_call(
        market.symbol,
        "world_markets.history",
        lambda t: t.history(period="2y", interval="1d"),
    )
    if frame is None or getattr(frame, "empty", True):
        logger.info("No history for %s (%s)", market.name, market.symbol)
        return None
    closes = frame["Close"].dropna()
    if closes.empty:
        return None
    row: dict[str, Any] = {
        "symbol": market.symbol,
        "name": market.name,
        "region": market.region,
        "bears_on": list(market.bears_on),
        "note": market.note,
        "last": float(closes.iloc[-1]),
        "as_of": str(closes.index[-1].date()) if len(closes.index) else None,
    }
    row.update({window: _pct_change(closes, days) for window, days in _WINDOWS.items()})
    return row


def fetch_world_markets(
    markets: tuple[Market, ...] = MARKETS + CROSS_ASSETS,
) -> list[dict[str, Any]]:
    """One row per market, in the order declared. Markets that fail are
    left out rather than faked — a missing index is not a flat one."""
    rows: list[dict[str, Any]] = []
    by_symbol = {m.symbol: m for m in markets}
    for _symbol, result in yf_gateway.map_symbols(
        lambda s: _fetch_one(by_symbol[s]), [m.symbol for m in markets]
    ):
        if result:
            rows.append(result)
    order = {m.symbol: i for i, m in enumerate(markets)}
    rows.sort(key=lambda r: order.get(r["symbol"], 999))
    return rows


@dataclass
class WorldSignals:
    """What the overnight tape says, for a holder who won't trade on it."""

    notable: list[str] = field(default_factory=list)
    regime_breaks: list[str] = field(default_factory=list)
    holdings_context: list[str] = field(default_factory=list)


def world_signals(rows: list[dict[str, Any]], held: set[str] | None = None) -> WorldSignals:
    """Turn the table into the few lines worth reading.

    `held` restricts the holdings context to tickers actually owned, so
    the block says something about this portfolio rather than reciting
    every mapping in the table.
    """
    held = {t.upper() for t in (held or set())}
    signals = WorldSignals()
    for row in rows:
        move, year = row.get("1d"), row.get("1y")
        if move is not None and abs(move) >= NOTABLE_MOVE_PCT:
            signals.notable.append(f"{row['name']} {move:+.1f}% ({row['region']})")
        if year is not None and year <= REGIME_BREAK_PCT:
            signals.regime_breaks.append(f"{row['name']} {year:+.0f}% over a year")
        overlap = sorted(held.intersection(row.get("bears_on") or []))
        if overlap and row.get("6mo") is not None:
            note = f" — {row['note']}" if row.get("note") else ""
            signals.holdings_context.append(
                f"{row['name']} {row['6mo']:+.0f}% over six months{note}: "
                f"reads across to {', '.join(overlap)}"
            )
    return signals


def world_markets_text(rows: list[dict[str, Any]], held: set[str] | None = None) -> str:
    """Plain-text block for the Ranker prompt, next to the FRED regime."""
    if not rows:
        return "World markets: data unavailable."
    lines = ["WORLD MARKETS (trailing % change; the US is one of these, not the whole picture):"]
    for row in rows:
        windows = " ".join(
            f"{window} {row[window]:+.1f}%" for window in _WINDOWS if row.get(window) is not None
        )
        lines.append(f"  {row['name']} ({row['region']}): {windows}")
    signals = world_signals(rows, held)
    if signals.regime_breaks:
        lines.append("Down more than 10% over the year: " + "; ".join(signals.regime_breaks))
    for line in signals.holdings_context:
        lines.append("Reads across to holdings: " + line)
    return "\n".join(lines)


__all__ = [
    "CROSS_ASSETS",
    "MARKETS",
    "WorldSignals",
    "fetch_world_markets",
    "world_markets_text",
    "world_signals",
]
