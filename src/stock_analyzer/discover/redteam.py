"""Adversarial red-team (Opus, single call).

Forces an explicit bear case for each pick. Without this stage, users tend
to skim the bull thesis and ignore disconfirming evidence. The red-team
output is placed inline with each pick in the final report — not at the
end — so it's impossible to skip past.
"""
from __future__ import annotations

from ..llm import AgnoAgent, Provider
from ..logging import get_logger
from ..models.llm import RedTeamOutput

logger = get_logger(__name__)

REDTEAM_INSTRUCTIONS = """\
You are a professional short seller. The user provides stock picks with
bull theses. Your job: write the BEAR case against each. Be specific and
data-grounded; generic warnings ("market downturn", "valuation risk")
are useless.

For each pick, output:

---
TICKER: <symbol>

Bear case (what must go wrong for a 30%+ decline in 12 months):
<3-4 sentences naming concrete failure modes — earnings miss, margin
compression, competitor wins, valuation re-rating, regulatory action —
drawn from the data already on this ticker. Cite specific numbers
when possible>

Most fragile assumption in the bull thesis:
<single sentence identifying the load-bearing assumption that, if wrong, breaks the thesis>

Watch this number: <e.g. "Q2 revenue growth — if it slips below 15%, thesis is wrong">

Fragility rank: <integer 1-5; 1=most fragile pick, 5=most resilient>
---

End with a "Single most fragile pick:" line naming the ticker most likely
to disappoint, and one sentence on why.

CRITICAL:
- Plain text only. No markdown headings or bold.
- Argue against each pick — do not concede to the bull case.

CITATION RULE (anti-hallucination):
Every numerical claim in your bear case (revenue %, margin, debt level,
P/E, target price) MUST appear in the picks input the user provided.
Do not fabricate hypothetical scenarios with invented numbers ("if
revenue falls 30%..."). Frame fragility as conditional on stated facts:
"AT current forward P/E of 67x (from analyst report), a single missed
quarter takes the multiple to 30x." If the input doesn't give you a
number, do NOT use one — argue qualitatively. Bear cases without
factual grounding are easy to dismiss; bear cases tied to specific
real numbers force the bull to respond.

STRUCTURED OUTPUT:
Your response is validated against a Pydantic schema (RedTeamOutput).
Populate `bear_cases` with one BearCase per pick (ticker, bear_case,
most_fragile_assumption, watch_metric, fragility_rank 1-5). Set
`single_most_fragile_pick` to the ticker + reason. Put the complete
prose rendering into `full_text` so downstream stages (Sizer,
Rebalancer) can read it from their prompt input. Structured fields
must match the prose.\
"""


class RedTeam:
    def __init__(
        self, provider: Provider, model: str, *, effort: str = "high"
    ):
        # Opus 4.7+ adaptive thinking — Claude self-allocates thinking budget,
        # gated by output_config.effort. high = deep adversarial reasoning.
        self.agent = AgnoAgent(
            "RedTeam",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "max_tokens": 6000,
                "temperature": 0,
            },
            instructions=REDTEAM_INSTRUCTIONS,
            output_schema=RedTeamOutput,
        )

    def critique(self, picks_text: str) -> RedTeamOutput:
        prompt = f"Picks to critique:\n\n{picks_text}"
        logger.info("Red-team critique of picks")
        result = self.agent.run(prompt).content
        if result is None:
            raise RuntimeError("RedTeam returned no content.")
        if isinstance(result, RedTeamOutput):
            return result
        if isinstance(result, str):
            return RedTeamOutput.model_validate_json(result)
        raise RuntimeError(
            f"RedTeam returned unexpected type {type(result).__name__}."
        )
