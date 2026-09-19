"""Rank candidate news items by materiality to a stock using an LLM.

Every fallback here defers to `data/news_rank.py` rather than to feed
order: when the model is unreachable or answers with something that isn't
a JSON array, the deterministic ranking is still a materiality ranking.
"""

from __future__ import annotations

import json
import re

from ..data.news_rank import rank_news
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


BATCH_RERANK_INSTRUCTIONS = (
    NEWS_RERANK_INSTRUCTIONS.split("Return EXACTLY 5 indices")[0]
    + """You are given SEVERAL stocks at once, each with its own candidate list
and its own indices.

For each stock return the indices of up to 5 items that are ABOUT that
company and material to it. A stock whose candidates are all commentary
about other companies gets an empty array — that is a normal answer and
far more useful than five filler links.

Return ONLY a JSON object keyed by ticker, most-material index first:
{"AVGO": [3, 0, 7], "NVDA": []}
No prose, no explanation, no markdown fence."""
)


class NewsReranker:
    def __init__(self, name: str, provider: Provider, model: str):
        self.agent = AgnoAgent(
            name,
            provider,
            model,
            instructions=NEWS_RERANK_INSTRUCTIONS,
        )
        # One call for the whole portfolio (see `rerank_batch`).
        self.batch_agent = AgnoAgent(
            f"{name} (batch)",
            provider,
            model,
            instructions=BATCH_RERANK_INSTRUCTIONS,
        )

    def rerank_batch(
        self,
        candidates: dict[str, list[dict]],
        names: dict[str, str | None] | None = None,
        *,
        top_n: int = 5,
    ) -> dict[str, list[dict]]:
        """Rank every holding's news in ONE call.

        A call per holding was the last per-stock model call left in the
        daily email once long-term views started being reused, and it was
        being spent to sort a feed that is mostly syndicated commentary.
        Batching keeps the model's judgement — which beats keyword
        matching at "is this even about this company?" — for the price of
        a single request. Any ticker the reply doesn't cover falls back to
        `rank_news`, so a bad reply degrades per stock, not for everyone.
        """
        names = names or {}
        usable = {t: c for t, c in candidates.items() if c}
        fallback = {t: rank_news(c, t, names.get(t), top_n=top_n) for t, c in candidates.items()}
        if not usable:
            return fallback

        blocks = []
        for ticker, items in usable.items():
            listing = "\n".join(
                f"[{i}] {c['title']} — {(c.get('snippet') or '')[:160]}"
                for i, c in enumerate(items)
            )
            blocks.append(f"{names.get(ticker) or ticker} ({ticker}):\n{listing}")
        prompt = "Candidate news per stock:\n\n" + "\n\n".join(blocks)

        try:
            raw = self.batch_agent.run(prompt).content
        except Exception as e:
            logger.warning("Batched news rerank failed (%s) — ranking in code instead", e)
            return fallback

        match = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not match:
            logger.warning("Batched rerank reply had no JSON object — ranking in code instead")
            return fallback
        try:
            picked = json.loads(match.group())
        except json.JSONDecodeError:
            logger.warning("Batched rerank reply was not JSON — ranking in code instead")
            return fallback

        out = dict(fallback)
        covered = 0
        for ticker, items in usable.items():
            indices = picked.get(ticker)
            if not isinstance(indices, list):
                continue
            covered += 1
            chosen: list[dict] = []
            for i in indices:
                if isinstance(i, int) and 0 <= i < len(items) and items[i] not in chosen:
                    chosen.append(items[i])
                if len(chosen) == top_n:
                    break
            # An empty array is a real answer ("none of these are about it"),
            # so it is kept rather than falling back.
            out[ticker] = chosen
        logger.info("Batched news rerank: %d/%d stocks ranked by the model", covered, len(usable))
        return out

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

        fallback = rank_news(candidates, symbol, name, top_n=top_n)
        try:
            raw = self.agent.run(prompt).content
        except Exception as e:
            logger.warning("News rerank LLM call failed for %s: %s", symbol, e)
            return fallback

        match = re.search(r"\[[\d,\s]+\]", raw or "")
        if not match:
            logger.warning("Rerank response had no JSON array for %s", symbol)
            return fallback
        try:
            indices = json.loads(match.group())
        except json.JSONDecodeError:
            return fallback

        seen: set[int] = set()
        ordered: list[dict] = []
        for i in indices:
            if isinstance(i, int) and 0 <= i < len(candidates) and i not in seen:
                seen.add(i)
                ordered.append(candidates[i])
            if len(ordered) == top_n:
                break
        return ordered or fallback
