"""Peer comparison — for each ticker, identify 3-4 closest public-equity
peers and fetch their fundamentals subset.

Why this matters: a "forward P/E of 19" or "FCF yield of 4%" only means
something relative to peers. Without peer context, the LLM can't judge
whether a candidate is "cheap" or "expensive" — only "absolute valuation."

Peers are identified by a Haiku call (cheap, ~$0.001/call, trained-data
knowledge of well-known competitive sets). Then yfinance fetches a
minimal fundamentals snapshot for each peer.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from ..data.fundamentals import fetch_fundamentals
from ..data.sec_edgar import load_ticker_cik_map

logger = get_logger(__name__)

PEER_FINDER_INSTRUCTIONS = """\
You are identifying public-equity peers for one ticker. The user provides
a ticker symbol and (optionally) a company name and sector.

Output EXACTLY a JSON array of 3-4 peer ticker symbols (uppercase, no
exchange prefix, no quotes around tickers themselves — just the JSON
array literal). Peers must be:
  - Publicly traded on US exchanges (NYSE/NASDAQ/AMEX) — so yfinance
    can fetch them
  - Genuine business competitors, not just same-sector
  - Roughly comparable market cap (within 1 order of magnitude)
  - Not the target ticker itself

Output examples:
  Input: NVDA → ["AMD", "AVGO", "INTC", "TSM"]
  Input: COST → ["WMT", "BJ", "TGT"]
  Input: SBUX → ["MCD", "CMG", "YUM"]

CRITICAL:
- Output ONLY the JSON array. No explanation, no preamble.
- All tickers UPPERCASE.
- 3-4 tickers; never 0 or 5+.\
"""

_PEER_FIELDS = (
    "name",
    "market_cap",
    "forward_pe",
    "peg_ratio",
    "revenue_growth_yoy",
    "operating_margin",
    "fcf_yield",
    "analyst_target_upside_pct",
    "analyst_recommendation",
)


class PeerFinder:
    def __init__(self, provider: Provider = "claude", model: str = "claude-haiku-4-5"):
        self.agent = AgnoAgent(
            "PeerFinder",
            provider,
            model,
            model_kwargs={"temperature": 0},
            instructions=PEER_FINDER_INSTRUCTIONS,
        )

    def find(
        self, ticker: str, *, name: str | None = None, sector: str | None = None
    ) -> list[str]:
        prompt = f"Ticker: {ticker}"
        if name:
            prompt += f"\nName: {name}"
        if sector:
            prompt += f"\nSector: {sector}"
        try:
            raw = self.agent.run(prompt).content
        except Exception as e:
            logger.warning("PeerFinder failed for %s: %s", ticker, e)
            return []
        peers = _parse_peer_list(raw)
        # Validate against SEC ticker map (drops Haiku hallucinated symbols).
        sec_map = load_ticker_cik_map()
        if sec_map:
            valid = set(sec_map.keys())
            peers = [
                p for p in peers
                if p in valid or p.replace(".", "-") in valid
            ]
        # De-dupe and exclude the target itself.
        out: list[str] = []
        seen = {ticker.upper()}
        for p in peers:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out[:4]


_JSON_ARRAY_RE = re.compile(r"\[[^\]]*\]")


def _parse_peer_list(text: str) -> list[str]:
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return []
    return [
        str(t).strip().upper()
        for t in arr
        if isinstance(t, str) and re.fullmatch(r"[A-Z][A-Z.\-]{0,5}", t.upper())
    ]


def _peer_snapshot(ticker: str) -> dict[str, Any] | None:
    """Fetch the minimal subset of fundamentals that matters for comparison."""
    full = fetch_fundamentals(ticker)
    if not full:
        return None
    return {field: full.get(field) for field in _PEER_FIELDS}


def fetch_peer_comparison(
    finder: PeerFinder,
    ticker: str,
    *,
    name: str | None = None,
    sector: str | None = None,
) -> dict[str, Any] | None:
    """Return {peers: {peer_ticker: snapshot}} for one ticker, or None on
    failure / no peers identified."""
    peer_tickers = finder.find(ticker, name=name, sector=sector)
    if not peer_tickers:
        return None
    snapshots: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        for peer, snap in zip(peer_tickers, ex.map(_peer_snapshot, peer_tickers)):
            if snap:
                snapshots[peer] = snap
    if not snapshots:
        return None
    return {"target": ticker, "peers": snapshots}


def batch_peer_comparison(
    tickers: list[str],
    target_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Fetch peer comparisons for a list of tickers. `target_meta` optionally
    maps ticker → {"name", "sector"} for better Haiku context."""
    finder = PeerFinder()
    target_meta = target_meta or {}

    def _one(t: str) -> tuple[str, dict[str, Any] | None]:
        meta = target_meta.get(t, {})
        return (
            t,
            fetch_peer_comparison(
                finder, t, name=meta.get("name"), sector=meta.get("sector")
            ),
        )

    results: dict[str, dict[str, Any]] = {}
    # Sequential because of the LLM call per ticker (ratelimit-friendly).
    for ticker, r in map(_one, tickers):
        if r:
            results[ticker] = r
    return results
