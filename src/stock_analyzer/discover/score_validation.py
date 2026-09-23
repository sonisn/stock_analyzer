"""Does the screen score predict anything? Retrospective validation.

The 0-100 composite in `screen.py` decides which candidates reach the
expensive Sonnet and Opus stages, and its weights were asserted rather than
fitted ("tune the thresholds here based on what the pipeline surfaces over
a few months"). This module is that tuning loop: it grades the score
against what the names actually did.

Two outputs, both of which can change how you weight things:

  1. MEAN FORWARD ALPHA BY SCORE QUINTILE. If the curve rises from Q1 to
     Q5, the composite is carrying signal and a higher cutoff is
     justified. If it is flat, the score is noise dressed up as a number
     and the ranking it imposes is arbitrary.

  2. INFORMATION COEFFICIENT PER SUB-COMPONENT. Spearman rank correlation
     between each sub-score and forward alpha. A component
     near zero is dead weight; a component with the WRONG SIGN is actively
     costing accuracy, which is the thing most worth knowing and the thing
     nothing in this repo previously measured.

POINT-IN-TIME DISCIPLINE. This grades the `score` values already stored in
the `candidates` table — the ones computed from the data available on the
run date. It deliberately does NOT recompute a score from today's
fundamentals: yfinance's `info` is a live snapshot with no history, so
today's `revenueGrowth` and `targetMeanPrice` are not what the screen saw,
and rescoring with them would leak the outcome into the feature. Prices are
the one thing safe to fetch retroactively, because a historical close is
the same number today as it was then.

No LLM calls. Run it whenever the DB has a few months of runs in it:

    uv run validate-screen
    uv run validate-screen --horizon 30 --lookback 365
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..db.session import get_session
from ..logging import get_logger
from .track_record import _close_on_or_after, _close_on_or_before, _fetch_history

logger = get_logger(__name__)

_DEFAULT_HORIZON_DAYS = 90
_DEFAULT_LOOKBACK_DAYS = 540
# Below this the quintile curve is meaningless — 5 buckets need bodies in them.
_MIN_SAMPLE_FOR_QUINTILES = 20
# An IC on fewer observations than this is noise; reported but flagged.
_MIN_SAMPLE_FOR_IC = 12
_MAX_WORKERS = 6


@dataclass(frozen=True)
class ScoredCandidate:
    """One historical candidate, its stored score, and what it then did."""

    ticker: str
    run_date: str
    score: float
    components: dict[str, float]
    forward_return_pct: float
    spy_return_pct: float

    @property
    def alpha_pct(self) -> float:
        return self.forward_return_pct - self.spy_return_pct


@dataclass(frozen=True)
class Bucket:
    label: str
    n: int
    mean_score: float
    mean_alpha_pct: float
    median_alpha_pct: float
    hit_rate: float  # share with positive alpha


@dataclass(frozen=True)
class ComponentIC:
    component: str
    n: int
    ic: float  # Spearman rank correlation vs forward alpha
    mean_value: float

    @property
    def noise_bar(self) -> float:
        """|IC| a component must clear to be distinguishable from chance.

        The standard error of a Spearman IC is about 1/sqrt(n-1), so this is
        the 95% bound. It matters a lot here: on 30 observations the bar is
        ~0.36, which means a -0.06 IC is nothing at all. Calling that "wrong
        sign" would send someone off to delete a component over noise — the
        mirror image of the false comfort this module exists to prevent.
        """
        return 1.96 / math.sqrt(max(self.n - 1, 1))

    @property
    def is_significant(self) -> bool:
        return self.n >= _MIN_SAMPLE_FOR_IC and abs(self.ic) >= self.noise_bar

    @property
    def verdict(self) -> str:
        """Plain-language read on one component's usefulness."""
        if self.n < _MIN_SAMPLE_FOR_IC:
            return f"too few observations (n={self.n})"
        if not self.is_significant:
            return f"inconclusive (need |IC| > {self.noise_bar:.2f} at n={self.n})"
        if self.ic < 0:
            return "WRONG SIGN — actively hurting"
        if self.ic < 0.15:
            return "weak but real"
        return "useful"


@dataclass(frozen=True)
class ValidationReport:
    horizon_days: int
    n_candidates: int
    n_unmeasurable: int
    buckets: list[Bucket]
    component_ics: list[ComponentIC]
    score_ic: float | None
    mean_alpha_pct: float | None


# --- statistics (no scipy dependency) --------------------------------------


def _ranks(values: list[float]) -> list[float]:
    """Fractional ranks with ties averaged — the basis of Spearman."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation. Used instead of Pearson because the score is an
    ordinal ranking device, and forward returns have fat tails that a
    linear correlation would let a single outlier dominate."""
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(_ranks(xs), _ranks(ys))


# --- DB read ---------------------------------------------------------------


def _flatten_components(
    components_json: str | None, breakdown_json: str | None
) -> dict[str, float]:
    """Every numeric sub-score as a flat {name: value} map.

    Group totals come from `score_components`, individual sub-scores from
    `score_breakdown`. Non-numeric leaves (the theme metadata) are skipped
    rather than coerced.
    """
    out: dict[str, float] = {}
    try:
        components = json.loads(components_json) if components_json else {}
    except json.JSONDecodeError, TypeError:
        components = {}
    try:
        breakdown = json.loads(breakdown_json) if breakdown_json else {}
    except json.JSONDecodeError, TypeError:
        breakdown = {}

    if isinstance(components, dict):
        for key, value in components.items():
            if isinstance(value, int | float) and not isinstance(value, bool):
                out[f"total.{key}"] = float(value)
    if isinstance(breakdown, dict):
        for group, leaves in breakdown.items():
            if not isinstance(leaves, dict):
                continue
            for key, value in leaves.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    out[f"{group}.{key}"] = float(value)
    return out


def _load_candidates(
    db_path: str, lookback_days: int, horizon_days: int
) -> list[tuple[str, str, float, dict[str, float]]]:
    """Stored scores old enough to have a finished forward window.

    Deduplicated to the OLDEST appearance per ticker so a name that keeps
    resurfacing across runs contributes one observation, not one per run
    (which would weight frequently-surfaced names more heavily and
    correlate the sample with itself).
    """
    from sqlalchemy import text

    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    try:
        with get_session(db_path) as session:
            rows = list(
                session.exec(
                    text(
                        "SELECT r.run_at, c.ticker, c.score, c.score_components, "
                        "       c.score_breakdown "
                        "FROM candidates c JOIN runs r ON r.id = c.run_id "
                        "WHERE r.run_at >= :cutoff AND c.score IS NOT NULL "
                        "ORDER BY r.run_at ASC"
                    ),
                    params={"cutoff": cutoff},
                )
            )
    except Exception as e:
        logger.warning("score-validation fetch failed (%s)", e)
        return []

    today = date.today()
    seen: set[str] = set()
    out: list[tuple[str, str, float, dict[str, float]]] = []
    for run_at, ticker, score, components_json, breakdown_json in rows:
        if ticker in seen:
            continue
        try:
            run_date = datetime.fromisoformat(str(run_at)).date()
        except ValueError:
            continue
        if (today - run_date).days < horizon_days:
            continue  # forward window hasn't finished
        seen.add(str(ticker))
        out.append(
            (
                str(ticker),
                run_date.isoformat(),
                float(score),
                _flatten_components(components_json, breakdown_json),
            )
        )
    return out


# --- forward returns -------------------------------------------------------


def _forward_return(ticker: str, run_date: str, horizon_days: int) -> float | None:
    start = date.fromisoformat(run_date)
    frame = _fetch_history(ticker, start - timedelta(days=5), start + timedelta(days=horizon_days))
    if frame is None or frame.empty:
        return None
    closes = frame["Close"].dropna()
    entry = _close_on_or_after(closes, start)
    exit_ = _close_on_or_before(closes, start + timedelta(days=horizon_days))
    if entry is None or exit_ is None or entry[0] <= 0:
        return None
    if exit_[1] <= entry[1]:
        return None
    return (exit_[0] / entry[0] - 1) * 100


# --- validation ------------------------------------------------------------


def _quintiles(scored: list[ScoredCandidate]) -> list[Bucket]:
    """Five equal-count buckets by score, lowest first."""
    if len(scored) < _MIN_SAMPLE_FOR_QUINTILES:
        return []
    ordered = sorted(scored, key=lambda c: c.score)
    size = len(ordered) // 5
    buckets: list[Bucket] = []
    for i in range(5):
        start = i * size
        end = (i + 1) * size if i < 4 else len(ordered)
        chunk = ordered[start:end]
        if not chunk:
            continue
        alphas = [c.alpha_pct for c in chunk]
        buckets.append(
            Bucket(
                label=f"Q{i + 1}",
                n=len(chunk),
                mean_score=statistics.mean([c.score for c in chunk]),
                mean_alpha_pct=statistics.mean(alphas),
                median_alpha_pct=statistics.median(alphas),
                hit_rate=sum(1 for a in alphas if a > 0) / len(alphas),
            )
        )
    return buckets


def validate_score(
    db_path: str,
    *,
    horizon_days: int = _DEFAULT_HORIZON_DAYS,
    lookback_days: int = _DEFAULT_LOOKBACK_DAYS,
) -> ValidationReport:
    """Grade every stored candidate score against its forward alpha."""
    candidates = _load_candidates(db_path, lookback_days, horizon_days)
    if not candidates:
        return ValidationReport(
            horizon_days=horizon_days,
            n_candidates=0,
            n_unmeasurable=0,
            buckets=[],
            component_ics=[],
            score_ic=None,
            mean_alpha_pct=None,
        )

    distinct_dates = sorted({run_date for _t, run_date, _s, _c in candidates})
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        spy_by_date = dict(
            zip(
                distinct_dates,
                ex.map(lambda d: _forward_return("SPY", d, horizon_days), distinct_dates),
                strict=False,
            )
        )
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        returns = list(ex.map(lambda c: _forward_return(c[0], c[1], horizon_days), candidates))

    scored: list[ScoredCandidate] = []
    unmeasurable = 0
    for (ticker, run_date, score, components), forward in zip(candidates, returns, strict=True):
        spy = spy_by_date.get(run_date)
        if forward is None or spy is None:
            unmeasurable += 1
            continue
        scored.append(
            ScoredCandidate(
                ticker=ticker,
                run_date=run_date,
                score=score,
                components=components,
                forward_return_pct=forward,
                spy_return_pct=spy,
            )
        )

    if not scored:
        return ValidationReport(
            horizon_days=horizon_days,
            n_candidates=0,
            n_unmeasurable=unmeasurable,
            buckets=[],
            component_ics=[],
            score_ic=None,
            mean_alpha_pct=None,
        )

    alphas = [c.alpha_pct for c in scored]
    by_component: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for candidate in scored:
        for name, value in candidate.components.items():
            by_component[name].append((value, candidate.alpha_pct))

    component_ics: list[ComponentIC] = []
    for name, pairs in by_component.items():
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ic = spearman(xs, ys)
        if ic is None:
            continue
        component_ics.append(
            ComponentIC(
                component=name,
                n=len(pairs),
                ic=ic,
                mean_value=statistics.mean(xs),
            )
        )

    return ValidationReport(
        horizon_days=horizon_days,
        n_candidates=len(scored),
        n_unmeasurable=unmeasurable,
        buckets=_quintiles(scored),
        component_ics=sorted(component_ics, key=lambda c: c.ic, reverse=True),
        score_ic=spearman([c.score for c in scored], alphas),
        mean_alpha_pct=statistics.mean(alphas),
    )


# --- rendering -------------------------------------------------------------


def format_validation_report(report: ValidationReport) -> str:
    """Human-readable report for the terminal."""
    if report.n_candidates == 0:
        return (
            "Score validation: nothing to grade yet. Need candidates whose "
            f"{report.horizon_days}-day forward window has finished "
            f"({report.n_unmeasurable} had no usable price data).\n"
            "Run the discover pipeline for a few months first — every run "
            "stores its scores, so this becomes answerable over time."
        )

    lines: list[str] = [
        "=" * 72,
        f"SCREEN SCORE VALIDATION — {report.horizon_days}-day forward window",
        "=" * 72,
        "",
        f"Candidates graded : {report.n_candidates}",
        f"Unmeasurable      : {report.n_unmeasurable} (no forward price data)",
        f"Mean alpha vs SPY : {report.mean_alpha_pct:+.2f}%",
        f"Score IC (Spearman, score vs forward alpha): {report.score_ic:+.3f}"
        if report.score_ic is not None
        else "Score IC          : n/a",
        "",
    ]

    if report.buckets:
        lines += _quintile_lines(report)
    else:
        lines.append(
            f"Not enough observations for quintiles "
            f"(need {_MIN_SAMPLE_FOR_QUINTILES}, have {report.n_candidates})."
        )

    if report.component_ics:
        lines += _component_lines(report)

    lines += [
        "",
        "Reminder: these scores were computed from point-in-time data at each",
        "run. Do not 'improve' this by rescoring with today's fundamentals —",
        "yfinance info has no history, so that would leak the outcome.",
        "=" * 72,
    ]
    return "\n".join(lines)


def _quintile_lines(report: ValidationReport) -> list[str]:
    """Mean forward alpha by score quintile, the spread, and the verdict."""
    lines: list[str] = [
        "MEAN FORWARD ALPHA BY SCORE QUINTILE",
        "(a rising curve means the composite ranks correctly; a flat one",
        " means the score is not separating winners from losers)",
        "",
        f"  {'':4s}  {'n':>4s}  {'mean score':>10s}  {'mean alpha':>10s}  "
        f"{'median':>8s}  {'hit rate':>8s}",
    ]
    for bucket in report.buckets:
        lines.append(
            f"  {bucket.label:4s}  {bucket.n:4d}  {bucket.mean_score:10.1f}  "
            f"{bucket.mean_alpha_pct:+9.2f}%  "
            f"{bucket.median_alpha_pct:+7.2f}%  {bucket.hit_rate:7.0%}"
        )
    top = report.buckets[-1].mean_alpha_pct
    bottom = report.buckets[0].mean_alpha_pct
    means = [b.mean_alpha_pct for b in report.buckets]
    monotone = all(b >= a for a, b in zip(means, means[1:], strict=False))
    lines += [
        "",
        f"  Q5 - Q1 spread: {top - bottom:+.2f}%",
        f"  Curve is {'monotone' if monotone else 'NOT monotone'} across Q1-Q5.",
    ]
    # The spread alone is a two-point comparison on small buckets, so a
    # pure-noise score can show a large one. The verdict leans on the
    # rank correlation AND on whether that correlation is resolvable at
    # this sample size: the standard error of a Spearman IC is roughly
    # 1/sqrt(n-1), so on 30 names anything under ~0.36 is
    # indistinguishable from chance. Saying so is the point — this tool
    # exists to withhold false comfort, and "not enough data yet" is a
    # legitimate answer that a bare spread would hide.
    ic = report.score_ic
    n = report.n_candidates
    se = 1.0 / math.sqrt(max(n - 1, 1))
    threshold = 1.96 * se
    lines.append(
        f"  IC needed to clear noise at n={n}: |IC| > {threshold:.3f} (1.96 x standard error)."
    )
    lines += ["", _separation_verdict(top - bottom, ic, threshold, n, monotone=monotone)]
    return lines


def _separation_verdict(
    spread: float, ic: float | None, threshold: float, n: int, *, monotone: bool
) -> str:
    if abs(spread) <= 1.0:
        return (
            f"  VERDICT: NO separation (spread {spread:+.2f}%) — the "
            f"composite is not earning its weight in deciding which "
            f"candidates reach the LLM stages."
        )
    if ic is None:
        return (
            "  VERDICT: forward returns show no variation, so no rank "
            "correlation is computable. Treat the spread as an artifact."
        )
    if ic >= threshold:
        return (
            f"  VERDICT: the score is separating (IC {ic:+.3f} clears the "
            f"{threshold:.3f} noise bar, spread {spread:+.2f}%)."
            + (
                ""
                if monotone
                else " The non-monotone middle says the "
                "ordering is noisy inside the range — trust the extremes, "
                "not the individual buckets."
            )
        )
    if ic <= -threshold:
        return (
            f"  VERDICT: the score is INVERTED (IC {ic:+.3f}) — "
            f"high-scoring candidates underperformed low-scoring ones. "
            f"Stop raising the cutoff and re-examine the components below."
        )
    return (
        f"  VERDICT: NO reliable separation. The Q5-Q1 gap "
        f"({spread:+.2f}%) is not backed by the overall ranking "
        f"(IC {ic:+.3f}, inside the {threshold:.3f} noise bar at "
        f"n={n}), so it is two small buckets differing by chance "
        f"rather than the score working. Collect more runs before "
        f"acting on it."
    )


def _component_lines(report: ValidationReport) -> list[str]:
    """Information coefficient per sub-component, and what to do about it."""
    lines = [
        "",
        "INFORMATION COEFFICIENT BY SUB-COMPONENT",
        "(Spearman rank correlation with forward alpha. Negative = the",
        " component is pointing the wrong way and costing you accuracy.)",
        "",
        f"  {'component':28s}  {'n':>4s}  {'IC':>7s}  {'mean':>6s}  verdict",
    ]
    for component in report.component_ics:
        lines.append(
            f"  {component.component:28s}  {component.n:4d}  "
            f"{component.ic:+7.3f}  {component.mean_value:6.1f}  "
            f"{component.verdict}"
        )
    bad = [c for c in report.component_ics if c.is_significant and c.ic < 0]
    if bad:
        lines += [
            "",
            "ACTION: these components have a wrong sign that clears the "
            "noise bar — drop or invert them in screen.py before tuning "
            "anything else:",
        ]
        lines += [
            f"  - {c.component} (IC {c.ic:+.3f} vs bar {c.noise_bar:.3f}, n={c.n})" for c in bad
        ]
    elif report.component_ics:
        lines += [
            "",
            "No component's IC clears the noise bar yet, in either "
            "direction. That is a sample-size result, not a clean bill of "
            "health — keep running the pipeline and re-check.",
        ]
    return lines
