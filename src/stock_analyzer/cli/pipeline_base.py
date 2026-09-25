"""What the step mixins read off the pipeline they are assembled into.

`DiscoverPipeline` and `RebalancePipeline` are built from mixins
(`DataSteps`, `AnalysisSteps`, `RebalancePlanSteps`, ...) that share
`self.state` and `self.settings` and call each other's helpers. Declaring
those here lets a type checker see them; nothing in this class runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import Settings


class PipelineBase:
    settings: Settings
    state: dict[str, Any]
    # Share of the run's LLM budget the per-candidate Analyst fan-out may use.
    ANALYST_BUDGET_SHARE: float

    if TYPE_CHECKING:
        # Defined on one mixin, called from another.
        def _apply_model_scores(
            self, candidates: list[dict[str, Any]], technicals: dict[str, dict[str, Any]]
        ) -> None: ...

        def _fetch_recent_news(self, tickers: list[str]) -> dict[str, list[dict[str, Any]]]: ...

        def _record_pick_suggestions(self, run_id: int) -> None: ...

        def _record_plan_suggestions(self, run_id: int) -> None: ...

        def _reinvest_for_unfunded_sales(self) -> dict[str, Any] | None: ...
