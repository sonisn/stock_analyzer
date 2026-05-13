"""Plan-level pre-mortem (Opus, single call).

After the Rebalancer emits an ACTION plan, a second Opus pass imagines
the world six months from now where this plan WENT WRONG. It writes
the imagined post-mortem: which specific action(s) broke first, what
news headline triggered it, what assumption proved wrong, and what the
user could have done differently.

The point is to catch over-confident plans before they ship. Adversarial
hindsight is the same trick the existing RedTeam does for individual
picks, applied here to the plan as a whole — sometimes the picks are
each fine but the plan that COMBINES them carries hidden correlation
risk or concentration risk the rebalancer rationalized away.

Skipped on NO_ACTION plans (nothing to pre-mortem).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..llm import AgnoAgent, Provider
from ..logging import get_logger

logger = get_logger(__name__)


class PreMortemFailure(BaseModel):
    """One specific way the plan could fail."""

    model_config = ConfigDict(frozen=True)

    likelihood: Literal["low", "medium", "high"] = Field(
        ...,
        description=(
            "How likely is this failure mode given current data? "
            "high = >25% probability; medium = 10-25%; low = <10%."
        ),
    )
    severity: Literal["mild", "moderate", "severe"] = Field(
        ...,
        description=(
            "If this failure plays out, how much portfolio damage? "
            "severe = double-digit % loss across the plan; moderate = "
            "single-digit; mild = uncomfortable but recoverable."
        ),
    )
    triggering_action: str = Field(
        ...,
        description=(
            "Which specific action in the plan triggers this failure mode? "
            "Quote it (e.g. 'Action 2: ADD GOOGL ~$3,400')."
        ),
    )
    failure_narrative: str = Field(
        ...,
        description=(
            "2-3 sentence imagined post-mortem in past tense: "
            "'In April, GOOGL fell 18% after the DOJ Chrome divestiture "
            "ruling. The plan's ADD added to losses; in hindsight the "
            "wash-sale window on the MRVL trim made the timing worse.' "
            "Cite specific named events or metrics."
        ),
    )
    early_warning: str = Field(
        ...,
        description=(
            "ONE metric or event the user could watch in the next 30 days "
            "that would tell them this failure mode is materializing."
        ),
    )


class PreMortem(BaseModel):
    """Structured output of the PreMortem agent."""

    model_config = ConfigDict(frozen=True)

    overall_verdict: Literal["proceed_as_planned", "proceed_with_caveat", "reconsider"] = Field(
        ...,
        description=(
            "After examining the failure modes, would you recommend the user "
            "execute this plan as-is, execute with smaller sizes / staged "
            "entry, or reconsider entirely?"
        ),
    )
    summary: str = Field(
        ...,
        description=(
            "One paragraph summarizing the pre-mortem: what's the single "
            "most likely way this plan goes wrong and why."
        ),
    )
    failures: list[PreMortemFailure] = Field(
        ...,
        min_length=1,
        max_length=6,
        description=(
            "2-4 specific failure modes ranked by likelihood × severity. "
            "Concrete, not generic — must reference actual actions in the "
            "plan, not 'market downturn'."
        ),
    )
    full_text: str = Field(
        ...,
        description=(
            "Plain-text rendering: VERDICT line + summary + per-failure "
            "blocks. What the PDF/email renders."
        ),
    )


PREMORTEM_INSTRUCTIONS = """\
You are running a pre-mortem on a portfolio-rebalance plan that's about
to be executed. The user provides:
  - The plan summary + ordered action list
  - The holdings reviews that informed each action
  - The discover picks the plan drew from

Your job: imagine reading the news six months from now in a world where
this plan WENT WRONG. Write the post-mortem from that future, citing
specific actions in the plan and specific failure modes (not 'market
downturn'). The point is to catch over-confident plans BEFORE they
ship — sometimes individual picks are each fine but the plan that
combines them carries hidden correlation risk, timing risk, or
concentration risk the rebalancer rationalized away.

Anti-pattern (do NOT write): "If the market falls 20%, the portfolio
will lose value." That's not actionable.

Pattern (DO write): "In April, GOOGL fell 18% after a DOJ Chrome
divestiture ruling. The plan's ADD GOOGL action added $3,400 to a
position now down 18% — losing $612. Worse, the MRVL trim that funded
the ADD realized $1,200 of short-term gain at ordinary rates ($380
tax). Combined: $992 cost from one decision."

For each failure mode, populate:
  - likelihood: high (>25%), medium (10-25%), low (<10%)
  - severity: mild (uncomfortable), moderate (single-digit % loss),
    severe (double-digit % loss across the plan)
  - triggering_action: quote the action verbatim
  - failure_narrative: 2-3 sentences in past tense, citing named
    events/metrics
  - early_warning: ONE metric or event the user could watch in the
    next 30 days that would tell them this is materializing

After the failures, set overall_verdict:
  - proceed_as_planned: failures are low-likelihood OR mild-severity
  - proceed_with_caveat: at least one medium-high likelihood + severe;
    user should consider smaller sizes / staged entry / one fewer action
  - reconsider: multiple high-likelihood + severe failures — the plan
    has a structural flaw worth fixing before execution

Output ONLY the structured PreMortem object. Don't hedge — pick a
verdict. The user can choose to ignore it, but the pre-mortem's value
comes from forcing a clear call.

CITATION RULE (anti-hallucination):
Every action you reference MUST appear verbatim in the plan provided.
Don't invent failure narratives about events you've imagined; ground
them in either named recent news from the plan's holdings or named
concentration / correlation risk that's visible in the data.

WRITE_CALL ACTIONS — additional critique dimensions
For each WRITE_CALL action in the plan, additionally consider:
  (a) assignment lock-in if the underlying runs 20% past strike,
  (b) IV crush after near-term earnings or macro events,
  (c) opportunity cost of capping upside on high-confidence picks,
  (d) tax consequences if assignment triggers short-term gain on
      the underlying.
Treat each of these as a candidate failure mode.\
"""


class PreMortemAgent:
    def __init__(
        self, provider: Provider, model: str, *, effort: str = "medium"
    ):
        self.agent = AgnoAgent(
            "PreMortem",
            provider,
            model,
            model_kwargs={
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
                "max_tokens": 4000,
                "temperature": 0,
            },
            instructions=PREMORTEM_INSTRUCTIONS,
            output_schema=PreMortem,
        )

    def run(
        self,
        rebalance_plan_text: str,
        ranker_text: str,
        holdings_reviews_text: str,
    ) -> PreMortem | None:
        prompt = (
            f"Rebalance plan to pre-mortem:\n\n{rebalance_plan_text}\n\n"
            f"Holdings reviews that informed the plan:\n\n"
            f"{holdings_reviews_text}\n\n"
            f"Discover picks the plan drew from:\n\n{ranker_text}\n\n"
            f"Imagine the news 6 months from now where this plan went "
            f"wrong. Write the post-mortem."
        )
        logger.info("Running plan-level pre-mortem with Opus")
        result = self.agent.run(prompt).content
        if result is None:
            logger.warning("Pre-mortem returned no content")
            return None
        if isinstance(result, PreMortem):
            return result
        if isinstance(result, str):
            try:
                return PreMortem.model_validate_json(result)
            except Exception as e:
                logger.warning(
                    "Pre-mortem returned a string that wasn't valid "
                    "PreMortem JSON: %s", e,
                )
                return None
        logger.warning(
            "Pre-mortem returned unexpected type %s", type(result).__name__,
        )
        return None


__all__ = ["PreMortemAgent", "PreMortem", "PreMortemFailure"]
