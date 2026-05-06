"""Rank candidate news items by materiality to a stock using an LLM."""
from __future__ import annotations

import json
import re

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)

NEWS_RERANK_INSTRUCTIONS = """\
You are a financial analyst. Rank candidate news items by MATERIALITY to the
named stock — i.e., likely impact on the share price.

Materiality priorities (high to low):
1. Earnings results, guidance changes, revenue/EPS surprises
2. M&A, partnerships, major customer wins, large contracts
3. Regulatory: FDA approvals/rejections, lawsuits, fines, antitrust, SEC actions
4. Executive/leadership changes (CEO, CFO)
5. Product launches with revenue impact, supply chain shocks
6. Premium-firm analyst upgrades/downgrades with new thesis
7. Other items that mention the company by name

Strongly deprioritize but do NOT exclude:
- Generic market commentary
- Tangential sector news

Hard filter (exclude entirely): items where the company is not mentioned at all
in either title or snippet.

Return EXACTLY 5 indices (or fewer ONLY if the candidate list has fewer than 5
items). Order most-material first. Example: [3, 0, 7, 2, 5]

Return ONLY a JSON array of 0-based integer indices. No prose, no explanation.\
"""


class NewsReranker:
    def __init__(self, name: str, provider: Provider, model: str):
        self.agent = AgnoAgent(
            name,
            provider,
            model,
            instructions=NEWS_RERANK_INSTRUCTIONS,
        )

    def rerank(
        self,
        candidates: list[dict],
        symbol: str,
        name: str | None = None,
        *,
        top_n: int = 5,
    ) -> list[dict]:
        if len(candidates) <= top_n:
            return candidates

        listing = "\n".join(
            f"[{i}] {c['title']} — {(c.get('snippet') or '')[:160]}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"Stock: {name or symbol} ({symbol})\n\n"
            f"Candidate news (index, title, snippet):\n{listing}"
        )

        try:
            raw = self.agent.run(prompt).content
        except Exception as e:
            logger.warning("News rerank LLM call failed for %s: %s", symbol, e)
            return candidates[:top_n]

        match = re.search(r"\[[\d,\s]+\]", raw or "")
        if not match:
            logger.warning("Rerank response had no JSON array for %s", symbol)
            return candidates[:top_n]
        try:
            indices = json.loads(match.group())
        except json.JSONDecodeError:
            return candidates[:top_n]

        seen: set[int] = set()
        ordered: list[dict] = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                ordered.append(candidates[i])
            if len(ordered) == top_n:
                break
        return ordered or candidates[:top_n]
