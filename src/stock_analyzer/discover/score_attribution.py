"""Which parts of the screen's score actually predict anything?

The screen awards points across ten components — revenue growth, FCF
yield, operating margin, debt health, 6-month relative strength, entry
zone, volume trend, weekly RSI, mentions, source diversity — and until
now not one of them had been measured against a realized return. The
weights were chosen by judgement and never audited. That matters: the
trend group alone can be forty percent of a candidate's score, and the
price trend score's information coefficient has already been measured at
roughly zero.

This joins each candidate's stored `score_breakdown` to its realized
forward excess return and computes, per component, the rank correlation
with that outcome — the same measure used on the contracted book.

It refuses to report on too little data. With a handful of run dates a
cross-sectional IC is noise wearing a number, and a confident-looking
table is worse than an empty one: the whole point is to stop weighting
things on judgement alone.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any

from ..logging import get_logger
from ..serialization import loads

logger = get_logger(__name__)

# Below these a result is not reported. Cross-sectional IC is averaged
# over dates, so it is the number of DATES that carries the power, not
# the number of rows.
MIN_DATES = 12
MIN_ROWS_PER_DATE = 20
# A t-stat is capped rather than infinite so the report stays readable;
# anything at the cap is "as consistent as this sample can show".
T_STAT_CAP = 99.9


@dataclass(frozen=True)
class ComponentResult:
    component: str
    ic: float
    t_stat: float
    dates: int
    rows: int
    coverage: float  # fraction of rows where the component was non-zero

    @property
    def verdict(self) -> str:
        if abs(self.t_stat) < 2:
            return "no evidence"
        return "predicts" if self.ic > 0 else "predicts inversely"


def load_rows(db_path: str, horizon: int) -> list[dict[str, Any]]:
    """One row per scored candidate that has a realized outcome: its
    component points, the run date, and the excess return."""
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    query = """
        select substr(r.run_at, 1, 10) as day, c.ticker, c.score_breakdown, o.excess_pct
        from candidates c
        join candidate_outcomes o on o.run_id = c.run_id and o.ticker = c.ticker
        join runs r on r.id = c.run_id
        where c.score_breakdown is not null and o.horizon_days = ?
    """
    rows = []
    for rec in con.execute(query, (horizon,)):
        try:
            breakdown = loads(rec["score_breakdown"])
        except TypeError, ValueError:
            continue
        flat: dict[str, float] = {}
        for group, parts in breakdown.items():
            if isinstance(parts, dict):
                for name, value in parts.items():
                    if isinstance(value, (int, float)):
                        flat[f"{group}.{name}"] = float(value)
        if flat:
            rows.append(
                {"day": rec["day"], "ticker": rec["ticker"], "excess": rec["excess_pct"], **flat}
            )
    con.close()
    return rows


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation, without scipy: rank both, then Pearson."""
    n = len(xs)
    if n < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return None

    def rank(vals: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            shared = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return None if den == 0 else num / den


def attribute(rows: list[dict[str, Any]]) -> list[ComponentResult]:
    """Per-component IC against forward excess return, averaged across run
    dates. Sorted by strength of evidence."""
    if not rows:
        return []
    components = sorted({k for r in rows for k in r if k not in {"day", "ticker", "excess"}})
    by_day: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_day.setdefault(r["day"], []).append(r)
    usable = {d: rs for d, rs in by_day.items() if len(rs) >= MIN_ROWS_PER_DATE}

    out: list[ComponentResult] = []
    for comp in components:
        ics: list[float] = []
        used_rows = 0
        nonzero = 0
        for day_rows in usable.values():
            pairs = [
                (r[comp], r["excess"]) for r in day_rows if comp in r and r["excess"] is not None
            ]
            if len(pairs) < MIN_ROWS_PER_DATE:
                continue
            ic = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if ic is not None:
                ics.append(ic)
                used_rows += len(pairs)
                nonzero += sum(1 for p in pairs if p[0])
        if len(ics) < 2:
            continue
        mean = sum(ics) / len(ics)
        var = sum((x - mean) ** 2 for x in ics) / (len(ics) - 1)
        se = math.sqrt(var / len(ics)) if var > 0 else 0.0
        if se > 0:
            t_stat = mean / se
        elif mean:
            # Identical IC on every date means no sampling variation at
            # all — the strongest evidence the sample can carry, not the
            # weakest. Returning 0 here read a perfect signal as noise.
            t_stat = math.copysign(T_STAT_CAP, mean)
        else:
            t_stat = 0.0
        out.append(
            ComponentResult(
                component=comp,
                ic=mean,
                t_stat=max(-T_STAT_CAP, min(T_STAT_CAP, t_stat)),
                dates=len(ics),
                rows=used_rows,
                coverage=(nonzero / used_rows) if used_rows else 0.0,
            )
        )
    return sorted(out, key=lambda r: -abs(r.t_stat))


def enough_data(results: list[ComponentResult]) -> bool:
    """Whether any component cleared the date threshold. A cross-sectional
    IC over a handful of dates is noise wearing a number."""
    return bool(results) and max(r.dates for r in results) >= MIN_DATES


def format_report(results: list[ComponentResult], horizon: int, total_rows: int) -> str:
    if not results:
        return (
            f"No component could be measured at {horizon}d: no run date has "
            f"{MIN_ROWS_PER_DATE}+ scored candidates with a realized outcome "
            f"({total_rows} joinable row(s) in total)."
        )
    dates = max(r.dates for r in results)
    head = (
        f"Screen component attribution — {horizon}-day forward excess return\n"
        f"{len(results)} component(s) over {dates} run date(s), {results[0].rows} observation(s)\n"
    )
    if not enough_data(results):
        head += (
            f"\n*** NOT ENOUGH DATA — reported for inspection only. ***\n"
            f"An IC averaged over {dates} date(s) has no power; {MIN_DATES} is the\n"
            f"minimum before any of this should change a weight.\n"
        )
    lines = [
        f"\n{'component':<32}{'IC':>8}{'t':>7}{'dates':>7}{'cover':>7}  verdict",
        "-" * 78,
    ]
    for r in results:
        lines.append(
            f"{r.component:<32}{r.ic:>+8.3f}{r.t_stat:>+7.2f}{r.dates:>7}"
            f"{r.coverage:>7.0%}  {r.verdict}"
        )
    return head + "\n".join(lines)
