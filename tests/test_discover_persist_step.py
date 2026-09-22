"""The discover run's last real step: persist, render, deliver.

A delivery failure must never cost the run — the rows and the PDF are
written before the email is attempted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from stock_analyzer.cli.discover import DiscoverPipeline
from stock_analyzer.config import Settings

STEPS = "stock_analyzer.cli.discover_steps.report_steps"


def _cand(ticker, ok, score):
    return {
        "ticker": ticker,
        "passed_filter": ok,
        "fail_reasons": [] if ok else ["below 200d"],
        "score": score,
        "score_components": {"fundamentals": 20, "trend": 30},
        "score_breakdown": {"fundamentals": {"roe": 5}},
        "sources": ["sp500"],
        "conviction": 3,
        "sector": "Tech",
        "price": 100.0,
    }


def _pipeline(tmp_path, email_to):
    settings = Settings(discover_db_path=str(tmp_path / "d.db"), email_to=email_to)
    p = DiscoverPipeline(settings)
    p.state.update(
        {
            "candidates": [
                _cand("NVDA", True, 80),
                _cand("AMD", True, 70),
                _cand("XOM", False, None),
            ],
            "survivors": [_cand("NVDA", True, 80), _cand("AMD", True, 70)],
            "picks": [(1, "NVDA", "#1 NVDA"), (2, "AMD", "#2 AMD")],
            "analyses": {"NVDA": SimpleNamespace(full_text="analyst text", upcoming_catalysts=[])},
            "ranker_text": "#1 NVDA — strong\n\n#2 AMD — ok\n",
            "redteam_text": "NVDA: bear\n---\nsummary",
            "sizer_text": "NVDA: 60%\nAMD: 40%\n---\nwarn",
            "holdings_summary": "(none)",
        }
    )
    return p


def _run(p, smtp):
    with (
        patch(f"{STEPS}.fetch_charts", return_value={"NVDA": b"png"}),
        patch(f"{STEPS}.SmtpServer", return_value=smtp),
        patch(f"{STEPS}._save_local_pdf", return_value=Path("/x/discover.pdf")),
        patch(f"{STEPS}.print_terminal_summary"),
        patch("builtins.print"),
    ):
        return p.step_persist_and_report(None)


def _count(db, table):
    with sqlite3.connect(db) as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_the_run_is_stored_and_emailed_with_its_pdf(tmp_path):
    smtp_calls: list = []
    p = _pipeline(tmp_path, "me@example.com")
    out = _run(p, SimpleNamespace(send_email=lambda *a, **k: smtp_calls.append((a, k))))

    db = tmp_path / "d.db"
    assert (_count(db, "candidates"), _count(db, "picks"), _count(db, "suggestions")) == (3, 2, 2)
    assert out.content.startswith("Run #1 emailed")
    (to, subject, _html), kw = smtp_calls[0]
    assert subject.endswith("NVDA, AMD")
    assert kw["attachments"][0][0].endswith(".pdf")
    assert set(kw["inline_images"]) == {"chart-NVDA"}
    assert p.state["run_id"] == 1


def test_no_recipient_still_persists(tmp_path):
    p = _pipeline(tmp_path, None)
    out = _run(p, SimpleNamespace(send_email=lambda *a, **k: pytest.fail("no email expected")))
    assert out.content.startswith("Run #1 persisted (no email)")
    assert _count(tmp_path / "d.db", "picks") == 2


def test_a_failed_delivery_does_not_lose_the_run(tmp_path):
    def down(*a, **k):
        raise OSError("smtp down")

    p = _pipeline(tmp_path, "me@example.com")
    out = _run(p, SimpleNamespace(send_email=down))
    assert out.content.startswith("Run #1 persisted (no email)")
    assert _count(tmp_path / "d.db", "picks") == 2
    assert p.state["pdf_bytes"]
