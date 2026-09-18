"""`model-review` — the monthly check-up for the forward-return model and
the screen score, emailed. No LLM calls; yfinance prices only.

  1. Label every candidate outcome whose window has closed.
  2. Retrain the shadow model (21-day, beta-neutral, 15 years). If it ever
     passes the walk-forward gate, the screen starts using it — the email
     says so in its subject.
  3. Grade the percentiles earlier runs recorded in shadow against what
     those names then did: the only test on data the model never saw.
  4. `validate-screen`: IC of every screen sub-score (fundamentals included,
     as run history accumulates) and the Ranker's forecast calibration.

Cron (scripts/run_model_review.sh) runs it on the 1st of each month.
"""

from __future__ import annotations

import io
import traceback
from contextlib import redirect_stdout
from datetime import date

from dotenv import load_dotenv

from ..config import Settings
from ..data.universe_base import load_base_universe
from ..logging import get_logger

logger = get_logger(__name__)


def _section(title: str, fn) -> str:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn()
    except Exception:
        buf.write(f"FAILED:\n{traceback.format_exc()}")
    return f"== {title} ==\n{buf.getvalue().rstrip()}\n"


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="model-review", description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--print", dest="print_only", action="store_true", help="print instead of emailing"
    )
    args = parser.parse_args(argv)
    load_dotenv()
    settings = Settings()
    db = settings.discover_db_path
    verdict = {"accepted": None}

    def labels() -> None:
        from ..model.labels import label_candidates

        print(f"New candidate outcomes labeled: {label_candidates(db)}")

    def train() -> None:
        from ..model.dataset import build_dataset, load_panel
        from ..model.ranker_model import format_model_report, save_model, walk_forward

        panel = load_panel(list(load_base_universe()), settings.model_cache_dir, years=15)
        result = walk_forward(
            build_dataset(panel), panel.spy.dropna().index, horizon=21, label_kind="beta_adj"
        )
        verdict["accepted"] = result.accepted
        print(format_model_report(result))
        print(f"Saved as model version {save_model(db, result)}")

    def shadow() -> None:
        from ..model.labels import grade_shadow_scores

        g = grade_shadow_scores(db, horizon=21)
        if not g["runs"]:
            print("No run has both shadow percentiles and a closed 21-day window yet.")
            return
        print(
            f"{g['runs']} runs, {g['names']} names: mean per-run IC "
            f"{g['mean_ic']:+.3f}, positive in {g['hit_rate']:.0%} of runs "
            f"(needs several months of runs before it means much)"
        )

    def screen() -> None:
        from .validate_screen import main as validate_main

        validate_main(["--horizon", "63"])

    body = "\n".join(
        [
            _section("1. Candidate outcomes", labels),
            _section("2. Shadow model retrain (21d, beta-neutral, 15y)", train),
            _section("3. Shadow model on live runs", shadow),
            _section("4. Screen score and forecast calibration (validate-screen)", screen),
        ]
    )
    status = {True: "MODEL ACCEPTED — screen now uses it", False: "model still in shadow"}.get(
        verdict["accepted"], "retrain failed"
    )
    subject = f"Model review {date.today():%b-%d}: {status}"
    if args.print_only or not settings.email_to:
        print(subject + "\n\n" + body)
        return
    from ..reporting.smtp import SmtpServer

    SmtpServer().send_email(settings.email_to, subject, body)
    logger.info("Model review emailed to %s", settings.email_to)


if __name__ == "__main__":
    main()
