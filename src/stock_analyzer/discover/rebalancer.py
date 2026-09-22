"""Portfolio rebalance decision (Opus + extended thinking, single call).

Synthesizes per-holding HOLD/TRIM/SELL verdicts from the Reviewer + new
discover picks from the Ranker + cash available + macro regime into a
coherent action list, ordered for execution: SELLs/TRIMs raise cash, BUYs
deploy it.

Long-term mandate: every holding is a 3-5 year investment. SELLs need a
broken thesis, a clearly better long-term use of the money, concentration
or a harvestable tax loss — never short-term price action.
"""

from __future__ import annotations

from typing import Any

from ..llm import (
    AgnoAgent,
    OutputTruncatedError,
    Provider,
    claude_thinking_kwargs,
)
from ..logging import get_logger
from ..models.llm import HoldingReview
from ..models.rebalance import RebalancePlan
from .rebalancer_prompt import _build_rebalancer_instructions

logger = get_logger(__name__)


REBALANCER_INSTRUCTIONS = _build_rebalancer_instructions()


def format_accounts_block(
    account_cash: dict[str, float], account_meta: dict[str, dict[str, Any]]
) -> str:
    """One line per account: tax status and the cash that can only be
    spent (or secure puts) inside that account."""
    names = sorted(set(account_cash) | set(account_meta))
    if not names:
        return ""
    lines = []
    for name in names:
        status = (account_meta.get(name) or {}).get("tax_status") or "unknown"
        cash = account_cash.get(name)
        cash_s = f"cash ${cash:,.0f}" if cash is not None else "cash unknown"
        lines.append(f"  {name} ({status.replace('_', '-')}): {cash_s}")
    return "\n".join(lines)


# What the rebalancer asks for. Three runs died at this step on
# 2026-09-20: 16,000 cut the JSON off mid-string, 32,000 was refused
# before it was sent (no timeout — see llm.MAX_NONSTREAMING_OUTPUT_TOKENS),
# and agno's stream flag did not reach the API. With an explicit timeout
# the budget can finally sit where truncation stops being the constraint.
REBALANCER_MAX_OUTPUT_TOKENS = 64_000


class RebalancePlanUnparseable(RuntimeError):
    """The model answered but the JSON did not survive.

    Carries the raw text so the caller can persist and show it: a plan
    that cost real money is worth more half-read than discarded, and a
    run that loses it must say so rather than render an empty action
    list.
    """

    def __init__(self, message: str, *, raw_text: str = "", truncated: bool = False) -> None:
        super().__init__(message)
        self.raw_text = raw_text
        self.truncated = truncated


class Rebalancer:
    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        effort: str = "high",
        cc_target_delta_min: float = 0.35,
        cc_target_delta_max: float = 0.45,
        cc_dte_min: int = 30,
        cc_dte_max: int = 45,
        cc_min_premium_usd: float = 500.0,
        cc_slippage_buffer: float = 0.10,
        cc_min_stub_usd: float = 1000.0,
        cc_stub_optimization: bool = True,
        csp_target_delta_min: float = 0.10,
        csp_target_delta_max: float = 0.25,
        csp_dte_min: int = 30,
        csp_dte_max: int = 45,
        csp_max_pct_per_put: float = 0.25,
        csp_max_pct_total: float = 0.80,
    ):
        instructions = _build_rebalancer_instructions(
            cc_target_delta_min=cc_target_delta_min,
            cc_target_delta_max=cc_target_delta_max,
            cc_dte_min=cc_dte_min,
            cc_dte_max=cc_dte_max,
            cc_min_premium_usd=cc_min_premium_usd,
            cc_slippage_buffer=cc_slippage_buffer,
            cc_min_stub_usd=cc_min_stub_usd,
            cc_stub_optimization=cc_stub_optimization,
            csp_target_delta_min=csp_target_delta_min,
            csp_target_delta_max=csp_target_delta_max,
            csp_dte_min=csp_dte_min,
            csp_dte_max=csp_dte_max,
            csp_max_pct_per_put=csp_max_pct_per_put,
            csp_max_pct_total=csp_max_pct_total,
        )
        # Opus 4.7+ adaptive thinking — high effort for the deepest synthesis
        # (combining holdings reviews + new picks + cash math + concentration).
        # output_schema=RebalancePlan gets agno to validate the model's
        # response against the Pydantic schema so downstream callers never
        # have to regex-parse plain text again.
        self.agent = AgnoAgent(
            "Rebalancer",
            provider,
            model,
            # Output budget. The plan must include: structured actions
            # list (incl. WRITE_CALL), option_writes list, AND the
            # full_text prose (cash math, tax-agnostic alternative,
            # wash-sale audit, per-holding reasoning, CC premium
            # reinvestment math, stub-consolidation narrative).
            # 8000 was the pre-CC value and caused mid-JSON truncation
            # on plans with WRITE_CALLs. 16000 then did the same on
            # 2026-09-20 — a 16-holding book with CC and CSP context
            # ran the JSON out at exactly 16,000 output tokens, and the
            # whole plan was lost. Past the SDK's non-streaming limit the
            # kwargs carry an explicit timeout, which is load-bearing.
            model_kwargs=claude_thinking_kwargs(effort, REBALANCER_MAX_OUTPUT_TOKENS),
            instructions=instructions,
            output_schema=RebalancePlan,
        )

    def decide(
        self,
        holdings_reviews: dict[str, HoldingReview] | dict[str, str],
        picks_text: str,
        cash_available: float | None,
        macro_summary: str = "",
        aggressiveness: str = "balanced",
        history_block: str = "",
        market_themes_block: str = "",
        cc_context_block: str = "",
        harvest_block: str = "",
        csp_context_block: str = "",
        accounts_block: str = "",
        add_on_block: str = "",
        obligations_block: str = "",
        backlog_block: str = "",
        stub_income_block: str = "",
    ) -> RebalancePlan:
        # Accept either the new structured form ({ticker: HoldingReview})
        # or the legacy free-text form ({ticker: str}). For the LLM prompt
        # we need prose, so unwrap HoldingReview.full_text.
        reviews_block = "\n\n".join(
            f"=== {ticker} ===\n{r.full_text if isinstance(r, HoldingReview) else r}"
            for ticker, r in holdings_reviews.items()
        )
        cash_line = (
            f"Available cash: ${cash_available:,.0f}"
            if cash_available is not None
            else "Available cash: unknown (size BUYs from SELL+TRIM proceeds only)"
        )
        if accounts_block:
            cash_line += (
                "\nCash by account (cash only funds BUYs, ADDs and put collateral "
                f"in its own account):\n{accounts_block}"
            )
        macro_block = f"Macro regime:\n{macro_summary}\n\n" if macro_summary else ""
        agg = aggressiveness.lower() if aggressiveness else "balanced"
        if agg not in ("conservative", "balanced", "aggressive"):
            logger.warning(
                "Unknown aggressiveness=%r — defaulting to 'balanced'",
                aggressiveness,
            )
            agg = "balanced"
        history_section = (
            f"Previous decisions (last 3 rebalance runs, oldest first):\n{history_block}\n\n"
            if history_block
            else ""
        )
        themes_section = (
            f"Current dominant market themes (use to validate continued "
            f"holding of theme members vs trimming positions in fading "
            f"themes):\n{market_themes_block}\n\n"
            if market_themes_block
            else ""
        )
        cc_section = f"{cc_context_block}\n\n" if cc_context_block else ""
        if add_on_block:
            cc_section = (
                "ADD-ON-WEAKNESS CANDIDATES (deterministic: HOLD >= 7, 15%+ below the "
                "52-week high, room under the weight limit — for a long-term holder a "
                "lower price on an intact case is a better entry; prefer these for ADDs "
                f"when the reviews still support them):\n{add_on_block}\n\n"
            ) + cc_section
        csp_section = f"{csp_context_block}\n\n" if csp_context_block else ""
        # Ahead of the other blocks: a sale that cannot be executed is
        # worse than no sale, and this is the only input that says which
        # shares are already spoken for.
        obligations_section = f"{obligations_block}\n\n" if obligations_block else ""
        # Signed orders behind a holding, and the premium a part-lot is
        # one purchase away from earning. Both bear on trims, and neither
        # was visible to this agent before 2026-09-20.
        backlog_section = f"{backlog_block}\n\n" if backlog_block else ""
        stub_income_section = f"{stub_income_block}\n\n" if stub_income_block else ""
        harvest_section = (
            f"TAX-LOSS HARVEST CANDIDATES (deterministic; see instructions):\n{harvest_block}\n\n"
            if harvest_block
            else ""
        )
        prompt = (
            f"AGGRESSIVENESS: {agg}\n"
            f"(Apply the {agg} rule set from your instructions. The "
            f"'Tax-agnostic alternative' section is MANDATORY in any "
            f"NO ACTION output.)\n\n"
            f"{obligations_section}"
            f"{backlog_section}"
            f"{stub_income_section}"
            f"{macro_block}"
            f"{themes_section}"
            f"{cc_section}"
            f"{csp_section}"
            f"{harvest_section}"
            f"{cash_line}\n\n"
            f"{history_section}"
            f"Current holdings reviews ({len(holdings_reviews)}):\n\n{reviews_block}\n\n"
            f"New discover picks:\n\n{picks_text}"
        )
        logger.info(
            "Generating rebalance plan with Opus (adaptive thinking, "
            "%d holdings, cash=%s, aggressiveness=%s)",
            len(holdings_reviews),
            f"${cash_available:,.0f}" if cash_available is not None else "unknown",
            agg,
        )
        try:
            raw = self.agent.run(prompt)
        except OutputTruncatedError as e:
            # The prose in a cut-off answer is still the only copy of
            # reasoning the run paid for, so it travels with the error
            # instead of dying in a log line.
            raise RebalancePlanUnparseable(str(e), raw_text=e.raw_text, truncated=True) from e
        result = raw.content
        if result is None:
            raise RuntimeError(
                "Rebalancer LLM returned no content — the rebalance plan "
                "cannot be rendered. Check provider rate limits and retry."
            )
        if not isinstance(result, RebalancePlan):
            # agno returns the parsed Pydantic instance when output_schema is set;
            # if for some reason we got a str, parse it.
            if isinstance(result, str):
                try:
                    result = RebalancePlan.model_validate_json(result)
                except Exception as e:
                    # Truncation is caught above; this is malformed JSON that
                    # finished inside the budget. Keep the text all the same.
                    raise RebalancePlanUnparseable(
                        f"Rebalancer returned a string that wasn't valid RebalancePlan JSON: {e}",
                        raw_text=result,
                    ) from e
            else:
                raise RuntimeError(
                    f"Rebalancer returned unexpected type {type(result).__name__}; "
                    "expected RebalancePlan."
                )
        if not result.full_text:
            raise RuntimeError(
                "Rebalancer returned a plan with empty full_text — nothing to render in the report."
            )
        return result
