"""Comparative ranker (Opus + extended thinking, single call).

Takes all candidate analyses + user holdings, picks top N with comparative
theses. The single LLM call that does most of the work in this pipeline —
Opus's reasoning depth pays off here vs N isolated per-ticker calls.
"""
from __future__ import annotations

from typing import Any

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from .schemas import RankerOutput

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
  alternatives by ticker.

CITATION RULE (anti-hallucination):
Every numerical claim you make (forward EPS, P/E, growth %, target
upside, margin, P/L) MUST appear in the analyst-reports input the user
provided. Do not invent or estimate. Every ticker you name in "Why
this over alternatives" MUST be one of the candidate tickers in the
analyses input. If you don't have a number to back a claim, drop the
claim — don't fabricate.

STRUCTURED OUTPUT:
Your response is validated against a Pydantic schema (RankerOutput).
Populate `picks` with one RankerPick per pick (rank, ticker, one_liner,
why_over_alternatives, conviction, time_horizon, sector_concentration_check,
bull_thesis, what_youre_betting_on). Populate `pairs_not_to_hold_together`
with any correlated pairs (empty list if none). Put the complete prose
rendering described above into `full_text` — the RedTeam, Sizer, and
Rebalancer all read full_text from their prompt input. The structured
fields must agree with `full_text` — same picks, same order, same
conviction numbers.\
"""


class Ranker:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        effort: str = "high",
        consensus_runs: int = 1,
    ):
        # Opus 4.7+ moved from `thinking.type=enabled` + budget_tokens to the
        # adaptive thinking API: Claude decides how much thinking to spend,
        # gated by `output_config.effort` (low | medium | high).
        # temperature=0 + consensus_runs > 1 → near-deterministic + variance check.
        self.consensus_runs = max(1, consensus_runs)
        self.agent = AgnoAgent(
            "Ranker",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "max_tokens": 8000,
                "temperature": 0,
            },
            instructions=RANKER_INSTRUCTIONS,
            output_schema=RankerOutput,
        )

    def _rank_once(
        self,
        analyses: dict[str, Any],
        holdings_summary: str,
        top_n: int,
        macro_context: str,
        track_record_block: str = "",
        market_themes_block: str = "",
    ) -> RankerOutput:
        # `analyses` is dict[ticker, AnalystReport] from Phase 4b; for
        # legacy callers it may be dict[ticker, str]. Unwrap to prose for
        # the prompt without depending on the type.
        candidates_block = "\n\n".join(
            f"=== {ticker} ===\n{getattr(analysis, 'full_text', analysis)}"
            for ticker, analysis in analyses.items()
        )
        macro_block = f"Macro regime:\n{macro_context}\n\n" if macro_context else ""
        themes_block = (
            f"Current dominant market themes (favor candidates that ride a "
            f"strong theme trending up; flag if a candidate is in a fading "
            f"or rolling-over theme):\n{market_themes_block}\n\n"
            if market_themes_block else ""
        )
        track_block = (
            f"Historical track record (your own past picks):\n{track_record_block}\n\n"
            if track_record_block else ""
        )
        prompt = (
            f"{macro_block}"
            f"{themes_block}"
            f"{track_block}"
            f"You will pick the top {top_n} from {len(analyses)} candidates.\n\n"
            f"Current holdings summary:\n{holdings_summary or '(none)'}\n\n"
            f"Candidate analyses:\n\n{candidates_block}"
        )
        result = self.agent.run(prompt).content
        if result is None:
            raise RuntimeError("Ranker returned no content.")
        if isinstance(result, RankerOutput):
            return result
        if isinstance(result, str):
            return RankerOutput.model_validate_json(result)
        raise RuntimeError(
            f"Ranker returned unexpected type {type(result).__name__}."
        )

    def rank(
        self,
        analyses: dict[str, Any],
        holdings_summary: str,
        top_n: int = 5,
        macro_context: str = "",
        track_record_block: str = "",
        market_themes_block: str = "",
    ) -> RankerOutput:
        """Single call when consensus_runs=1; otherwise run N times and
        return the run whose picks best overlap the majority-consensus set."""
        logger.info(
            "Ranking %d candidates with Opus (adaptive thinking, macro=%s, "
            "consensus_runs=%d)",
            len(analyses),
            bool(macro_context),
            self.consensus_runs,
        )
        if self.consensus_runs <= 1:
            return self._rank_once(
                analyses, holdings_summary, top_n, macro_context,
                track_record_block, market_themes_block,
            )

        outputs: list[RankerOutput] = []
        pick_sets: list[set[str]] = []
        for i in range(self.consensus_runs):
            output = self._rank_once(
                analyses, holdings_summary, top_n, macro_context,
                track_record_block, market_themes_block,
            )
            outputs.append(output)
            picks = {p.ticker for p in output.picks}
            pick_sets.append(picks)
            logger.info(
                "Ranker run %d/%d picked %s",
                i + 1, self.consensus_runs, sorted(picks),
            )

        # Majority threshold = ceil(N/2). With N=3 → 2 runs agreeing.
        threshold = (self.consensus_runs + 1) // 2
        all_tickers = set().union(*pick_sets)
        consensus = {
            t for t in all_tickers
            if sum(1 for s in pick_sets if t in s) >= threshold
        }
        logger.info(
            "Consensus: %d of %d distinct picks agreed in >=%d runs: %s",
            len(consensus), len(all_tickers), threshold, sorted(consensus),
        )

        if not consensus:
            logger.warning(
                "No consensus reached across %d ranker runs; "
                "returning first run's output verbatim",
                self.consensus_runs,
            )
            return outputs[0]

        best_idx = max(
            range(self.consensus_runs),
            key=lambda i: len(pick_sets[i] & consensus),
        )
        logger.info(
            "Using run %d's output (overlaps consensus by %d picks)",
            best_idx + 1, len(pick_sets[best_idx] & consensus),
        )
        return outputs[best_idx]
