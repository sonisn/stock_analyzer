"""Comparative ranker — one high-effort reasoning call per consensus round.

Takes all candidate analyses + user holdings, picks top N with comparative
theses. This is the highest-leverage stage in the pipeline — reasoning
depth pays off here vs N isolated per-ticker calls. By default it runs
one round per provider in `DISCOVER_RANKER_PROVIDERS` (claude, gemini,
openai) and majority-votes ticker membership across the rounds, so
disagreement reflects genuinely different models rather than one model's
own sampling stochasticity. A single-round config (one provider) still
works — `rank()` short-circuits to a plain single call.
"""

from __future__ import annotations

from typing import Any

from ..llm import AgnoAgent, Provider, reasoning_model_kwargs, run_with_fallback
from ..logging import get_logger
from ..models.llm import RankerOutput

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

PROBABILITY-WEIGHTED SCENARIOS:
For each pick, you MUST emit exactly 3 scenarios in the `scenarios` list:
  - bull: optimistic case (thesis fully plays out, multiple expands)
  - base: muted case (thesis half-plays-out, multiple roughly stable)
  - bear: thesis breaks (specific failure mode you can name)

Each scenario has a probability (in [0, 1]) and a target_return_pct
over the 6-12 month horizon. The 3 probabilities MUST sum to 1.0.

Probability discipline rules — these catch the common mistakes:
  - DO NOT default to 33/34/33. A pick at conviction 9 should look
    something like 50/35/15 (bull dominant); a conviction 5 should look
    more like 25/40/35 (base dominant); a conviction 7 might be 40/40/20.
  - Bear probability MUST be at least 10% on every pick — even your
    best ideas can fail. A bear<10% means you've miscalibrated.
  - target_return_pct must be conditional on the scenario playing out
    fully. Don't blend — bull is "if the bull scenario hits". Typical
    ranges over 6-12 months: bull +25% to +60% (rarely higher),
    base 0% to +15%, bear -15% to -35%.

A downstream Sizer + analytics layer computes expected return
deterministically as Σ(probability × target_return_pct). Calibrate
your numbers as if you'll be measured on the EV vs realized return.

You ARE measured on it. Every pick's conviction, EV and three scenario
probabilities are persisted, and a calibration pass grades them once the
horizon elapses. When a "Your forecast calibration" block appears in the
input, it is your own scorecard, and you must act on it:
  - A negative mean EV error means your past forecasts were too
    optimistic. Lower your target_return_pct values and/or shift
    probability mass from bull toward base and bear.
  - A positive mean EV error means you were too conservative.
  - If the block says your conviction scores are NOT ordered by realized
    alpha, your confidence has been carrying no information. Spread the
    conviction numbers only where the forward evidence actually differs
    between candidates, and say in "Why this over alternatives" what
    separates them.
  - If a scenario's observed frequency is far from the probability you
    stated (e.g. bear landed 35% of the time while you averaged 12%),
    move your probabilities toward the observed frequency.

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


def _build_agent(provider: Provider, model: str, effort: str) -> AgnoAgent:
    # Opus 4.7+ adaptive thinking spends part of `max_tokens` on the
    # thinking trace, so the JSON output competes with it. Our response is
    # rich (5 picks × 3 scenarios × bull/bear prose + pairs_not_to_hold_
    # together + full_text) and we hit truncation mid-string at ~4500
    # visible tokens when capped at 8000. Bumped to 16000 so thinking AND
    # output both fit comfortably — kept the same across providers even
    # though only Claude's adaptive thinking actually spends into it.
    return AgnoAgent(
        "Ranker",
        provider,
        model,
        model_kwargs=reasoning_model_kwargs(provider, effort, max_tokens=16000),
        instructions=RANKER_INSTRUCTIONS,
        output_schema=RankerOutput,
    )


class Ranker:
    def __init__(
        self,
        rounds: list[tuple[Provider, str]],
        *,
        effort: str = "high",
        fallback: tuple[Provider, str] | None = None,
    ):
        """`rounds` is one (provider, model) pair per consensus round.

        A single-entry list behaves exactly like the old single-provider,
        single-call Ranker. Multiple entries — typically one per provider
        (claude/gemini/openai) — each run a full independent ranking pass;
        `rank()` then majority-votes across them. `fallback`, if given, is
        the (provider, model) each round retries on if its primary call
        fails with an auth/rate-limit/provider error.
        """
        if not rounds:
            raise ValueError("Ranker needs at least one (provider, model) round.")
        self.rounds = rounds
        self.consensus_runs = len(rounds)
        self.effort = effort
        self.fallback = fallback
        self._agents = [_build_agent(provider, model, effort) for provider, model in rounds]

    def _run_round(self, agent: AgnoAgent, *args: Any, **kwargs: Any) -> Any:
        build_fallback = (
            (lambda: _build_agent(self.fallback[0], self.fallback[1], self.effort))
            if self.fallback and self.fallback[0] != agent.provider
            else None
        )
        return run_with_fallback(agent, build_fallback, *args, **kwargs)

    def _rank_once(
        self,
        agent: AgnoAgent,
        analyses: dict[str, Any],
        holdings_summary: str,
        top_n: int,
        macro_context: str,
        track_record_block: str = "",
        market_themes_block: str = "",
        calibration_block: str = "",
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
            if market_themes_block
            else ""
        )
        track_block = (
            f"Historical track record (your own past buy picks and sell calls, "
            f"with alpha vs SPY — positive alpha = call was right regardless "
            f"of direction; 'beta-adj' strips out market exposure, and is the "
            f"part attributable to picking):\n{track_record_block}\n\n"
            if track_record_block
            else ""
        )
        calib_block = (
            f"Your own forecast calibration — how your past conviction "
            f"scores, expected returns and scenario probabilities actually "
            f"held up. Adjust this run's numbers accordingly:\n"
            f"{calibration_block}\n\n"
            if calibration_block
            else ""
        )
        prompt = (
            f"{macro_block}"
            f"{themes_block}"
            f"{track_block}"
            f"{calib_block}"
            f"You will pick the top {top_n} from {len(analyses)} candidates.\n\n"
            f"Current holdings summary:\n{holdings_summary or '(none)'}\n\n"
            f"Candidate analyses:\n\n{candidates_block}"
        )
        result = self._run_round(agent, prompt).content
        if result is None:
            raise RuntimeError("Ranker returned no content.")
        if isinstance(result, RankerOutput):
            return result
        if isinstance(result, str):
            return RankerOutput.model_validate_json(result)
        raise RuntimeError(f"Ranker returned unexpected type {type(result).__name__}.")

    def rank(
        self,
        analyses: dict[str, Any],
        holdings_summary: str,
        top_n: int = 5,
        macro_context: str = "",
        track_record_block: str = "",
        market_themes_block: str = "",
        calibration_block: str = "",
    ) -> RankerOutput:
        """Single call when consensus_runs=1; otherwise run one pass per
        round (each on its own provider/model) and return the run whose
        picks best overlap the majority-consensus set, with agreement_ratio
        and voting_providers attached to each of its picks."""
        logger.info(
            "Ranking %d candidates (rounds=%s, macro=%s)",
            len(analyses),
            [f"{p}/{m}" for p, m in self.rounds],
            bool(macro_context),
        )
        if self.consensus_runs <= 1:
            return self._rank_once(
                self._agents[0],
                analyses,
                holdings_summary,
                top_n,
                macro_context,
                track_record_block,
                market_themes_block,
                calibration_block,
            )

        outputs: list[RankerOutput] = []
        pick_sets: list[set[str]] = []
        for i, agent in enumerate(self._agents):
            output = self._rank_once(
                agent,
                analyses,
                holdings_summary,
                top_n,
                macro_context,
                track_record_block,
                market_themes_block,
                calibration_block,
            )
            outputs.append(output)
            picks = {p.ticker for p in output.picks}
            pick_sets.append(picks)
            logger.info(
                "Ranker round %d/%d (%s/%s) picked %s",
                i + 1,
                self.consensus_runs,
                self.rounds[i][0],
                self.rounds[i][1],
                sorted(picks),
            )

        # Majority threshold = ceil(N/2). With N=3 → 2 runs agreeing.
        threshold = (self.consensus_runs + 1) // 2
        all_tickers = set().union(*pick_sets)
        consensus = {t for t in all_tickers if sum(1 for s in pick_sets if t in s) >= threshold}
        logger.info(
            "Consensus: %d of %d distinct picks agreed in >=%d runs: %s",
            len(consensus),
            len(all_tickers),
            threshold,
            sorted(consensus),
        )

        if not consensus:
            logger.warning(
                "No consensus reached across %d ranker rounds (%s) — the "
                "candidate set does not separate cleanly. Returning the "
                "first round's output verbatim; treat these picks as "
                "low-confidence.",
                self.consensus_runs,
                [p for p, _ in self.rounds],
            )
            return outputs[0]

        best_idx = max(
            range(self.consensus_runs),
            key=lambda i: len(pick_sets[i] & consensus),
        )
        logger.info(
            "Using round %d's output (overlaps consensus by %d picks)",
            best_idx + 1,
            len(pick_sets[best_idx] & consensus),
        )
        winner = outputs[best_idx]
        annotated_picks = []
        for pick in winner.picks:
            agreeing_rounds = [i for i, s in enumerate(pick_sets) if pick.ticker in s]
            annotated_picks.append(
                pick.model_copy(
                    update={
                        "agreement_ratio": len(agreeing_rounds) / self.consensus_runs,
                        "voting_providers": [self.rounds[i][0] for i in agreeing_rounds],
                    }
                )
            )
        return winner.model_copy(update={"picks": annotated_picks})
