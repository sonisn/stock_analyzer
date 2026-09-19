"""Rank a stock's news by materiality without an LLM.

The daily email used to pay a rerank call per holding to put the most
material headline first. One batched call now ranks the whole portfolio
(agents/news_reranker.py), and this is what runs when that call fails or
can't be parsed — which is how a stock used to end up with whatever order
the feed happened to return.

It mirrors the priorities in the reranker's own prompt: earnings and
guidance first, then deals, regulation, leadership, products, and analyst
actions last, with listicles and syndicated commentary pushed down.

The harder job here is relevance, not order. Across four holdings' feeds
(2026-09-19) 35 of 40 items were syndication that never mentioned the
holding — NVDA's ten were about AbbVie, Boeing, Iamgold and CoreWeave —
so an item whose headline doesn't name the company is dropped outright
and the list is not padded back to five. When that leaves nothing, the
block says so and `discover/stock_facts.py` fills it with filings and
estimate revisions instead.
"""

from __future__ import annotations

import re
from typing import Any

# (points, markers). A headline scores the highest tier it matches, plus a
# small bonus per extra tier, so "Q3 earnings beat; CEO steps down" outranks
# a headline that is only about a CEO. Markers match whole words only —
# as substrings "ban" fires on "bank" and "order" on "borders".
_TIERS: tuple[tuple[int, tuple[str, ...]], ...] = (
    (
        100,
        (
            "earnings",
            "eps",
            "guidance",
            "quarterly results",
            "q1 ",
            "q2 ",
            "q3 ",
            "q4 ",
            "revenue",
            "beats",
            "misses",
            "outlook",
            "forecast",
            "raises guidance",
            "raises outlook",
            "raises forecast",
            "cuts guidance",
            "lowers guidance",
            "warns",
            "profit",
        ),
    ),
    (
        85,
        (
            "acquisition",
            "acquires",
            "merger",
            "to buy",
            "buyout",
            "takeover",
            "stake in",
            "partnership",
            "partners with",
            "contract",
            "deal",
            "order",
            "customer",
            "supply agreement",
            "joint venture",
        ),
    ),
    (
        75,
        (
            "fda",
            "lawsuit",
            "sues",
            "antitrust",
            "regulator",
            "investigation",
            "probe",
            "subpoena",
            "fine",
            "settlement",
            "sec filing",
            "doj",
            "ftc",
            "tariff",
            "export control",
            "sanction",
            "ban",
        ),
    ),
    (
        65,
        (
            "ceo",
            "cfo",
            "chief executive",
            "chief financial",
            "resigns",
            "steps down",
            "appoints",
            "names ",
            "succession",
            "layoff",
            "job cuts",
        ),
    ),
    (
        55,
        (
            "launch",
            "unveils",
            "announces",
            "chip",
            "product",
            "plant",
            "factory",
            "capacity",
            "supply",
            "shortage",
            "recall",
            "delay",
            "production",
        ),
    ),
    (
        40,
        (
            "upgrade",
            "downgrade",
            "price target",
            "initiated",
            "initiates",
            "rating",
            "analyst",
            "overweight",
            "underweight",
            "buy rating",
        ),
    ),
)

# Aggregator filler: it names the company but says nothing that moves it.
_NOISE: tuple[str, ...] = (
    "jim cramer",
    "motley fool",
    "zacks",
    "should you buy",
    "should you invest",
    "is it too late",
    "here's why",
    "here is why",
    "best stocks",
    "top stocks",
    "stocks to buy",
    "stocks to watch",
    "magnificent seven",
    "if you invested",
    "millionaire",
    "prediction:",
    "my top",
    "better buy",
    "vs.",
    "wall street thinks",
    "3 reasons",
    "5 reasons",
    "where will",
)

# Listicle shapes a fixed phrase can't catch: "3 AI Stocks ... to Buy",
# "The 5 Best Chip Stocks", "Here's the ETF I'd Buy".
_NOISE_RES: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern)
    for pattern in (
        r"\b\d+\s+(?:\w+\s+){0,2}stocks?\b",
        r"\bstocks?\s+(?:\w+\s+){0,3}to\s+buy\b",
        r"\bto\s+buy\s+and\s+hold\b",
        r"\bbest\s+(?:\w+\s+){0,2}stocks?\b",
        r"\betf\s+i['\u2019]?d\b",
        r"\bforget\s+buying\b",
        r"\bcould\s+turn\s+\$",
    )
)

# Syndication mills. Measured on four holdings' feeds (2026-09-19), 35 of
# 40 items came from these; they re-word the same market commentary for
# every ticker and almost never carry company news.
_SYNDICATION: tuple[str, ...] = (
    "motley fool",
    "fool.com",
    "insider monkey",
    "zacks",
    "stocktwits",
    "24/7 wall st",
    "simply wall st",
    "trefis",
    "investorplace",
    "tipranks",
    "gurufocus",
)

_PREMIUM: tuple[str, ...] = (
    "reuters",
    "bloomberg",
    "wall street journal",
    "wsj",
    "financial times",
    "cnbc",
    "barron",
    "marketwatch",
    "associated press",
    "investor's business daily",
)

_NOISE_PENALTY = 45
_SYNDICATION_PENALTY = 30
_PREMIUM_BONUS = 8
_NAMED_BONUS = 25
_EXTRA_TIER_BONUS = 6
_SUFFIXES = {
    "inc",
    "inc.",
    "corp",
    "corp.",
    "corporation",
    "co",
    "co.",
    "ltd",
    "plc",
    "sa",
    "nv",
    "holdings",
    "group",
    "technologies",
    "technology",
    "systems",
    "company",
    "the",
    "&",
    "limited",
    "ag",
}


_TIER_RES: tuple[tuple[int, re.Pattern[str]], ...] = tuple(
    (points, re.compile(r"\b(?:" + "|".join(re.escape(m) for m in markers) + r")"))
    for points, markers in _TIERS
)


def _text(item: dict[str, Any]) -> str:
    return f"{item.get('title') or ''} {item.get('snippet') or ''}".lower()


def company_terms(symbol: str, name: str | None) -> list[str]:
    """Words that mean "this item is about the company": the ticker, and
    the distinctive words of its name (never the legal suffix, which would
    match every filing headline)."""
    terms = [symbol.lower()] if symbol else []
    for word in re.split(r"[\s,]+", (name or "").strip()):
        clean = word.strip(".,").lower()
        if len(clean) >= 3 and clean not in _SUFFIXES:
            terms.append(clean)
    return terms


def mentions_company(text: str, symbol: str, name: str | None) -> bool:
    """Does this text actually name the company?

    Short terms — the ticker itself, and names like Arm — are matched only
    where they appear capitalized, because `\bbe\b` against lowercase text
    makes every headline "about" BE (Bloom Energy) and `\barm\b` matches
    prose. Longer terms are matched case-insensitively.
    """
    for term in company_terms(symbol, name):
        if len(term) <= 4:
            # Two-letter tickers only in full caps: "Could Be Bigger" is not
            # news about BE (Bloom Energy). Longer ones may be title-cased,
            # so "Arm Holdings" counts for ARM.
            variants = {term.upper()} if len(term) <= 2 else {term.upper(), term.capitalize()}
            if any(re.search(rf"\b{re.escape(v)}\b", text) for v in variants):
                return True
        elif re.search(rf"\b{re.escape(term)}\b", text.lower()):
            return True
    return False


def score_item(item: dict[str, Any], *, named: bool = True) -> int:
    text = _text(item)
    hits = [points for points, pattern in _TIER_RES if pattern.search(text)]
    score = (max(hits) + _EXTRA_TIER_BONUS * (len(hits) - 1)) if hits else 0
    if named:
        score += _NAMED_BONUS
    source = f"{item.get('publisher') or ''} {item.get('source') or ''}".lower()
    if any(p in source for p in _PREMIUM):
        score += _PREMIUM_BONUS
    if any(m in source for m in _SYNDICATION):
        score -= _SYNDICATION_PENALTY
    if any(n in text for n in _NOISE) or any(r.search(text) for r in _NOISE_RES):
        score -= _NOISE_PENALTY
    return score


def rank_news(
    candidates: list[dict[str, Any]],
    symbol: str,
    name: str | None = None,
    *,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """The items that are actually about this company, most material first.

    Anything whose HEADLINE never names the company is dropped, and the
    list is not padded back to `top_n` — an empty list is the honest
    answer and the caller says so. Measured on four holdings' feeds
    (2026-09-19), 35 of 40 items were syndicated commentary about other
    companies; ranking them only chose which filler went first.

    The body is deliberately not enough to qualify an item: every "10
    stocks to buy" piece mentions ten companies in its body. Naming the
    company is also not enough on its own — an item whose penalties sink
    it below zero ("Are Oils-Energy Stocks Lagging Bloom Energy?") is
    filler that happens to spell the name, and loses its slot too.
    """
    scored = [
        (i, c, score_item(c))
        for i, c in enumerate(candidates)
        if mentions_company(c.get("title") or "", symbol, name)
    ]
    # Feed order is roughly newest-first; it breaks ties and nothing more.
    ranked = sorted((row for row in scored if row[2] > 0), key=lambda row: (-row[2], row[0]))
    return [item for _, item, _ in ranked[:top_n]]


__all__ = ["rank_news", "score_item", "company_terms", "mentions_company"]
