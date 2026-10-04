"""Step runner for the discover and rebalance pipelines.

A pipeline is a list of `Step`s and `Parallel` blocks of steps. Each step
is a zero-argument callable (a bound `step_*` method that reads and writes
its pipeline's `state`) returning a one-line summary.

  - A step that raises stops the run: later steps would read state it
    never wrote. `run_pipeline` raises `PipelineFailed` after logging.
  - A step inside a `Parallel` block that raises is logged and the block's
    other steps finish; the run goes on, and the steps after it degrade on
    the missing state (each one checks what it needs).
  - Nothing is retried here: a re-run would re-pay every model call the
    step already made. Provider errors fall back inside the step
    (llm.run_with_fallback); data fetches retry in the HTTP layer.

Each step's timing, status and summary (or error) is logged and kept in
the `pipeline_steps` table of the run's database.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime

from .logging import get_logger

logger = get_logger(__name__)

_DETAIL_CHARS = 500


@dataclass(frozen=True)
class Step:
    name: str
    run: Callable[[], str | None]


@dataclass(frozen=True)
class Parallel:
    name: str
    steps: tuple[Step, ...]

    def __init__(self, *steps: Step, name: str) -> None:
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "steps", steps)


@dataclass
class StepRecord:
    step: str
    block: str | None
    started_at: str
    seconds: float
    status: str
    detail: str | None


@dataclass
class PipelineResult:
    run_key: str
    records: list[StepRecord] = field(default_factory=list)

    @property
    def failed(self) -> list[StepRecord]:
        return [r for r in self.records if r.status == "failed"]


class PipelineFailed(RuntimeError):
    def __init__(self, pipeline: str, step: str, cause: BaseException) -> None:
        super().__init__(f"{pipeline}: step {step!r} failed: {cause}")
        self.step = step


def _run_step(step: Step, block: str | None) -> tuple[StepRecord, BaseException | None]:
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()
    try:
        summary = step.run()
    except Exception as e:
        seconds = time.monotonic() - t0
        logger.exception("Step %s failed after %.1fs", step.name, seconds)
        detail = f"{type(e).__name__}: {e}"[:_DETAIL_CHARS]
        return StepRecord(step.name, block, started, seconds, "failed", detail), e
    seconds = time.monotonic() - t0
    logger.info("Step %s done in %.1fs: %s", step.name, seconds, summary or "")
    detail = summary[:_DETAIL_CHARS] if summary else None
    return StepRecord(step.name, block, started, seconds, "ok", detail), None


def run_pipeline(
    name: str,
    steps: Sequence[Step | Parallel],
    *,
    db_path: str | None = None,
) -> PipelineResult:
    """Run `steps` in order; see the module docstring for failure handling."""
    result = PipelineResult(run_key=uuid.uuid4().hex)
    logger.info("=== %s: %d stages ===", name, len(steps))
    try:
        for item in steps:
            if isinstance(item, Parallel):
                with ThreadPoolExecutor(
                    max_workers=len(item.steps), thread_name_prefix=item.name
                ) as ex:
                    outcomes = list(ex.map(lambda s, b=item.name: _run_step(s, b), item.steps))
                result.records.extend(rec for rec, _ in outcomes)
                continue
            rec, error = _run_step(item, None)
            result.records.append(rec)
            if error is not None:
                raise PipelineFailed(name, item.name, error) from error
    finally:
        total = sum(r.seconds for r in result.records if r.block is None) + sum(
            max((r.seconds for r in result.records if r.block == b), default=0.0)
            for b in {r.block for r in result.records if r.block}
        )
        failed = [r.step for r in result.failed]
        logger.info(
            "=== %s: %d steps in %.0fs%s ===",
            name,
            len(result.records),
            total,
            f", failed: {', '.join(failed)}" if failed else "",
        )
        if db_path:
            _store(db_path, name, result)
    return result


def _store(db_path: str, pipeline: str, result: PipelineResult) -> None:
    """Keep the run's step log; a failure to write it never fails the run."""
    try:
        from .db.session import get_session
        from .db.tables import PipelineStep

        with get_session(db_path) as session:
            for r in result.records:
                session.add(
                    PipelineStep(
                        run_key=result.run_key,
                        pipeline=pipeline,
                        step=r.step,
                        block=r.block,
                        started_at=r.started_at,
                        seconds=round(r.seconds, 2),
                        status=r.status,
                        detail=r.detail,
                    )
                )
    except Exception as e:
        logger.warning("Could not store the %s step log: %s", pipeline, e)
