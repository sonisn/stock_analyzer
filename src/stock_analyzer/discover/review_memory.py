"""The Reviewer's memory: what it last said about each holding.

The holdings are kept for years, so a verdict that flips SELL -> HOLD ->
SELL from one run to the next on the same facts is noise the user has to
see through. Each holding's latest stored review (`holdings_reviews`) goes
back into its payload as `previous_review`, and the Reviewer is asked to
name what changed whenever its verdict does.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from ..db.session import exec_sql, get_session
from ..logging import get_logger

logger = get_logger(__name__)

EXCERPT_CHARS = 400
_REASONING_START = "Forward outlook:"


def previous_reviews(db: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """{ticker: {date, verdict, confidence, excerpt}} from the latest run that
    reviewed it. Never raises: a missing table just means no memory."""
    wanted = sorted({t.upper() for t in tickers})
    if not wanted:
        return {}
    params = {f"t{i}": t for i, t in enumerate(wanted)}
    try:
        with get_session(db) as session:
            rows = (
                exec_sql(
                    session,
                    text(
                        "SELECT h.ticker, r.run_at, h.verdict, h.confidence, h.review_text "
                        "FROM holdings_reviews h JOIN runs r ON r.id = h.run_id "
                        f"WHERE h.ticker IN ({', '.join(f':{k}' for k in params)}) "
                        "AND h.verdict IS NOT NULL ORDER BY r.run_at DESC"
                    ),
                    params,
                )
                .mappings()
                .all()
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Previous holding reviews unavailable (%s)", e)
        return {}
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        ticker = row["ticker"]
        if ticker in out:
            continue
        excerpt = " ".join((row["review_text"] or "").split())
        # The stored text opens with the ticker, verdict and position facts
        # the payload already carries; the reasoning starts at the outlook.
        _, found, outlook = excerpt.partition(_REASONING_START)
        if found:
            excerpt = outlook.strip()
        if len(excerpt) > EXCERPT_CHARS:
            excerpt = excerpt[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."
        out[ticker] = {
            "date": str(row["run_at"])[:10],
            "verdict": row["verdict"],
            "confidence": row["confidence"],
            "excerpt": excerpt,
        }
    return out
