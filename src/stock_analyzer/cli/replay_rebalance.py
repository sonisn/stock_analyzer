"""`replay-rebalance` — re-run only the rebalancer, on a past run's inputs.

The rebalance plan is one LLM call at the end of a forty-minute, ~$2.40
pipeline. That made every change to it — a token budget, a timeout, a
prompt tweak — cost a full run to test, and on 2026-09-20 three runs in a
row were spent discovering that the call had been refused before it was
sent. Testing the last step should not require repeating the first
fifteen.

Everything that call needs is already persisted: the holdings reviews and
the ranker's picks are the bulk of its prompt. This loads them back and
calls `Rebalancer.decide` on its own, for roughly a fifth of the cost and
a tenth of the wait.

It is a diagnostic, not a substitute for a run: the covered-call and
cash-secured-put context blocks are built live and are not stored, so the
replayed prompt is a little smaller than the real one and its plan must
not be traded on. What it answers is whether the call completes and
parses.

    uv run replay-rebalance              # the most recent rebalance run
    uv run replay-rebalance --run 34
    uv run replay-rebalance --out plan.txt
"""

from __future__ import annotations

import argparse
from typing import Any

from dotenv import load_dotenv
from sqlmodel import select

from ..config import Settings
from ..db.session import get_session
from ..db.tables import HoldingReviewRow, Run, RunOutput
from ..logging import get_logger
from ..usage import TRACKER, log_usage_summary

logger = get_logger(__name__)


def load_inputs(db_path: str, run_id: int | None) -> tuple[int, dict[str, str], str]:
    """(run_id, {ticker: review text}, ranker text) for a stored rebalance."""
    with get_session(db_path) as session:
        if run_id is None:
            row = session.exec(
                select(Run).where(Run.kind == "rebalance").order_by(Run.id.desc())  # type: ignore[attr-defined]
            ).first()
            if row is None:
                raise SystemExit("No rebalance run is stored yet — run rebalance-portfolio once.")
            run_id = row.id
        reviews = {
            r.ticker: r.review_text or ""
            for r in session.exec(
                select(HoldingReviewRow).where(HoldingReviewRow.run_id == run_id)
            ).all()
        }
        out = session.exec(select(RunOutput).where(RunOutput.run_id == run_id)).first()
        ranker_text = (out.ranker_full if out else "") or ""
    if not reviews:
        raise SystemExit(f"Run {run_id} stored no holdings reviews — nothing to replay.")
    return run_id, reviews, ranker_text


def main(argv: list[str] | None = None) -> None:
    from ..discover.rebalancer import Rebalancer

    parser = argparse.ArgumentParser(prog="replay-rebalance", description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=int, default=None, help="run id (default: latest rebalance)")
    parser.add_argument("--cash", type=float, default=None, help="override the cash balance")
    parser.add_argument("--out", default=None, help="write the plan text here")
    args = parser.parse_args(argv)

    load_dotenv()
    settings = Settings.from_env()
    run_id, reviews, ranker_text = load_inputs(settings.discover_db_path, args.run)
    prompt_chars = sum(len(v) for v in reviews.values()) + len(ranker_text)
    logger.info(
        "Replaying run %d: %d holding review(s), ranker %d chars (~%d prompt chars). "
        "CC/CSP context is NOT included — this is a diagnostic, not tradeable advice.",
        run_id,
        len(reviews),
        len(ranker_text),
        prompt_chars,
    )

    # The deterministic blocks are cheap to rebuild live, so the replay
    # carries them: they are exactly the facts a sale decision needs, and
    # replaying without them reproduces the gap instead of the fix.
    from ..data.backlog import backlog_block, batch_rpo
    from ..data.brokerage import (
        fetch_account_cash,
        fetch_covered_call_obligations,
        fetch_portfolio_holdings,
    )
    from ..discover.sale_validation import covered_call_block, validate_sales

    holdings = fetch_portfolio_holdings()
    positions: dict[str, dict[str, Any]] = {}
    for items in holdings.values():
        for h in items:
            if t := h.get("ticker"):
                positions.setdefault(t, {"units": 0.0})["units"] += float(h.get("units") or 0)
    obligations = fetch_covered_call_obligations()
    books = batch_rpo([t for t in reviews if t in positions])
    cash = args.cash if args.cash is not None else sum(fetch_account_cash().values())
    logger.info(
        "Live context: %d position(s), %d with written calls, %d with a contracted book, cash $%s",
        len(positions),
        len(obligations),
        len(books),
        f"{cash:,.0f}",
    )

    rebalancer = Rebalancer("claude", settings.discover_opus_model)
    plan: Any = rebalancer.decide(
        reviews,
        ranker_text,
        cash,
        aggressiveness=settings.discover_rebalance_aggressiveness,
        obligations_block=covered_call_block(positions, obligations),
        backlog_block=backlog_block(books),
    )
    plan, sale_warnings = validate_sales(plan, positions=positions, obligations=obligations)

    print(f"\nstatus      : {plan.status}")
    print(f"actions     : {len(plan.actions)}")
    for a in plan.actions:
        print(f"  {a.action:<12} {a.ticker:<8} {a.sizing}")
    print(f"summary     : {plan.summary}")
    for w in sale_warnings:
        print(f"  ! {w}")
    print(f"full_text   : {len(plan.full_text):,} chars")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(plan.full_text)
        print(f"written to  : {args.out}")
    log_usage_summary()
    total, _ = TRACKER.total_cost()
    print(f"cost        : ${total:.2f}")


if __name__ == "__main__":
    main()
