"""The evidence score: a ranking built only from signals that held up when
tested, recorded beside the screen and the model's picks so each can be
graded against the others.

Nothing the LLM stages produce is in it, and they never see it (the
analysis payload drops it): it is the control group for the question the
pick scorecard asks — does the model step add anything to what plain,
measured evidence would have chosen?

The ingredients and why each is here:

  - contracted book (SEC remaining performance obligations) year on year:
    126-day excess-return IC +0.12, S&P 500 2016-2026, surviving beta,
    sector, 12-1 momentum and revenue growth (screen.py, 2026-09-20);
  - gross profitability (gross profit over total assets, as filed): among
    companies already $20B+, beta-adjusted IC +0.058 / +0.083 at 126 / 252
    days (t 2.5 / 2.6), positive in both halves but fading
    (`factor-study`, 2026-10-06);
  - an insider-buying cluster in the last 90 days (2+ insiders, $10k each):
    12-month excess +8.9% vs +3.2% without, t 1.84, both eras positive
    (factor study 2026-09-26).

Each becomes a 0-1 percentile within the run's screen survivors (a name
that doesn't report one gets the middle, 0.5: absence is not evidence), and
the score is their weighted sum on 0-100. The weights follow the strength
of each test and are fixed in advance — they are NOT to be tuned on the
live scorecard, which would turn the control into one more fitted model.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

WEIGHTS = {"contracted_book": 0.45, "gross_profitability": 0.35, "insider_cluster": 0.20}
NEUTRAL = 0.5


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    """{ticker: 0-1 rank percentile}; ties share their average rank."""
    if not values:
        return {}
    order = sorted(values.items(), key=lambda kv: kv[1])
    n = len(order)
    out: dict[str, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and order[j + 1][1] == order[i][1]:
            j += 1
        pct = ((i + j) / 2) / (n - 1) if n > 1 else NEUTRAL
        for k in range(i, j + 1):
            out[order[k][0]] = pct
        i = j + 1
    return out


def evidence_scores(
    tickers: Iterable[str],
    *,
    book_yoy: dict[str, float],
    gross_profitability: dict[str, float],
    insider_clusters: set[str],
) -> dict[str, dict[str, Any]]:
    """{ticker: {"score": 0-100, "contracted_book", "gross_profitability",
    "insider_cluster"}} for `tickers` (the run's screen survivors); each
    part is the 0-1 value that entered the score."""
    names = [t.upper() for t in tickers]
    book = _percentiles({t: v for t, v in book_yoy.items() if t in names})
    gp = _percentiles({t: v for t, v in gross_profitability.items() if t in names})
    out = {}
    for t in names:
        parts = {
            "contracted_book": round(book.get(t, NEUTRAL), 3),
            "gross_profitability": round(gp.get(t, NEUTRAL), 3),
            "insider_cluster": 1.0 if t in insider_clusters else 0.0,
        }
        score = 100 * sum(WEIGHTS[k] * v for k, v in parts.items())
        out[t] = {"score": round(score, 1), **parts}
    return out


def hidden_from_model(breakdown: dict[str, Any] | None) -> dict[str, Any]:
    """`breakdown` as the LLM stages may see it: without the evidence score."""
    return {k: v for k, v in (breakdown or {}).items() if k != "evidence"}


__all__ = ["NEUTRAL", "WEIGHTS", "evidence_scores", "hidden_from_model"]
