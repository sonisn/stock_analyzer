"""Hard macro/regime veto on the Ranker's picks.

The Ranker gets the FRED macro regime as prompt context (`macro_context` in
`ranker.py`), but an LLM's reasoning tends to center on the single-name
thesis and can under-weight a regime shift even when the data is right
there in the prompt. This is a deterministic backstop: in a risk-off
regime, suppress picks whose own technicals mark them as high-momentum/
high-beta rather than trust the model to have discounted them correctly.

This does not replace the macro context in the prompt — it's a second,
rule-based check applied AFTER the Ranker has already reasoned about the
regime, catching the case where it didn't weight it enough.
"""

from __future__ import annotations

from typing import TypeIs

from ..logging import get_logger
from ..models.llm import RankerOutput

logger = get_logger(__name__)

# Same thresholds `data/fred_macro.py::regime_summary_text` uses to call a
# yield curve "INVERTED" and VIX "elevated" — kept identical so the veto
# fires exactly when the macro block Claude reads already says "risk-off",
# not on some separately-tuned definition.
_INVERTED_YIELD_SPREAD = -0.10
_ELEVATED_VIX = 30.0

# A pick counts as "high-momentum" (the style that historically gets hit
# hardest when a risk-off regime turns) when its 6-month relative strength
# vs SPY exceeds this — i.e. it has meaningfully outrun the market already.
_MOMENTUM_RS_6MO_THRESHOLD = 0.15


def _is_risk_off(macro_data: dict | None) -> TypeIs[dict]:
    if not macro_data:
        return False
    spread = macro_data.get("yield_spread_10y_2y")
    vix = macro_data.get("vix")
    return (
        spread is not None
        and vix is not None
        and spread < _INVERTED_YIELD_SPREAD
        and vix > _ELEVATED_VIX
    )


def apply_macro_veto(
    ranker_output: RankerOutput,
    macro_data: dict | None,
    technicals: dict[str, dict],
) -> tuple[RankerOutput, list[str]]:
    """Suppress high-momentum picks when the macro regime is risk-off.

    Returns `(possibly-trimmed output, suppression reasons)`. Returns the
    input unchanged with an empty reasons list when the regime isn't
    risk-off, or when no pick trips the momentum threshold — this never
    runs in a normal (non-inverted, non-elevated-VIX) regime.
    """
    if not _is_risk_off(macro_data):
        return ranker_output, []

    kept = []
    reasons: list[str] = []
    for pick in ranker_output.picks:
        rs6 = (technicals.get(pick.ticker) or {}).get("rs_6mo")
        if rs6 is not None and rs6 > _MOMENTUM_RS_6MO_THRESHOLD:
            reasons.append(
                f"{pick.ticker}: suppressed by macro veto — risk-off regime "
                f"(10Y-2Y spread {macro_data.get('yield_spread_10y_2y'):+.2f}%, "
                f"VIX {macro_data.get('vix'):.1f}) + high 6mo relative strength "
                f"({rs6:+.0%} vs SPY, above the {_MOMENTUM_RS_6MO_THRESHOLD:.0%} "
                f"momentum threshold)."
            )
            continue
        kept.append(pick)

    if not reasons:
        return ranker_output, []

    for r in reasons:
        logger.warning("Macro veto: %s", r)

    suppressed_block = "\n\nMACRO VETO — picks suppressed (risk-off regime):\n" + "\n".join(
        f"- {r}" for r in reasons
    )
    return (
        ranker_output.model_copy(
            update={"picks": kept, "full_text": ranker_output.full_text + suppressed_block}
        ),
        reasons,
    )
