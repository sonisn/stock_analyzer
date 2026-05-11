"""Comparative ranker (Opus + extended thinking, single call).

Takes all candidate analyses + user holdings, picks top N with comparative
theses. The single LLM call that does most of the work in this pipeline —
Opus's reasoning depth pays off here vs N isolated per-ticker calls.
"""
from __future__ import annotations

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)

RANKER_INSTRUCTIONS = """\
You are a portfolio manager picking 5 stocks for a 6-12 month hold from a
shortlist. The user provides one structured analysis per candidate plus a
summary of their current holdings, and optionally a macro regime block.

DO NOT make tool calls. Use ONLY the provided data. Reason comparatively —
the point of this stage is to pick BETWEEN candidates, not validate each
in isolation. If a macro regime block is provided, weight cyclicals vs
defensives appropriately and cite regime fit in your bull thesis where it
materially affects the call (e.g. inverted curve → underweight credit-sensitive
names; high VIX → favor balance-sheet quality).

For each of your 5 picks, output exactly this block:

---
PICK <n>: <TICKER> — <one-sentence thesis>

Why this over alternatives:
<2-3 sentences citing specific other candidates that lost out and why>

Conviction (1-10): <integer>
Time horizon: 6-12 months
Sector concentration check: <does this overlap with the user's current holdings? flag if so>

Bull thesis:
<3-4 sentences synthesizing fundamentals + trend + catalysts>

What you're betting on:
<1-2 sentences making the core assumption explicit>
---

End with a "Pairs not to hold together:" line listing any of your 5 picks
that are highly correlated (same sector + similar drivers).

CRITICAL:
- Plain text only. No markdown headings or bold.
- Pick exactly 5 unless fewer than 5 candidates were provided.
- Order picks by conviction descending.
- The "Why this over alternatives" section is non-optional — name the
  alternatives by ticker.\
"""


class Ranker:
    def __init__(
        self, provider: Provider, model: str, *, effort: str = "high"
    ):
        # Opus 4.7+ moved from `thinking.type=enabled` + budget_tokens to the
        # adaptive thinking API: Claude decides how much thinking to spend,
        # gated by `output_config.effort` (low | medium | high).
        self.agent = AgnoAgent(
            "Ranker",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "max_tokens": 8000,
            },
            instructions=RANKER_INSTRUCTIONS,
        )

    def rank(
        self,
        analyses: dict[str, str],
        holdings_summary: str,
        top_n: int = 5,
        macro_context: str = "",
    ) -> str:
        candidates_block = "\n\n".join(
            f"=== {ticker} ===\n{analysis}" for ticker, analysis in analyses.items()
        )
        macro_block = f"Macro regime:\n{macro_context}\n\n" if macro_context else ""
        prompt = (
            f"{macro_block}"
            f"You will pick the top {top_n} from {len(analyses)} candidates.\n\n"
            f"Current holdings summary:\n{holdings_summary or '(none)'}\n\n"
            f"Candidate analyses:\n\n{candidates_block}"
        )
        logger.info(
            "Ranking %d candidates with Opus (adaptive thinking, macro=%s)",
            len(analyses),
            bool(macro_context),
        )
        return self.agent.run(prompt).content
