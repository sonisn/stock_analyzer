"""Forecast calibration — score the ranker's own probabilities.

The ranker prompt ends with "Calibrate your numbers as if you'll be
measured on the EV vs realized return." This module is that measurement.
It answers three questions, each of which used to be unanswerable because
the forecast was never written down:

  1. IS EV BIASED?  mean(realized - EV) over picks whose horizon has
     elapsed. A persistently positive number means the ranker is
     under-promising; negative (the common failure) means it is talking
     itself into picks.

  2. DOES CONVICTION MEAN ANYTHING?  mean forward alpha bucketed by the
     stated conviction score. If a conviction-9 bucket does not beat a
     conviction-6 bucket, the number is decoration and the Sizer should
     stop weighting it.

  3. ARE THE SCENARIO PROBABILITIES HONEST?  for each of bull/base/bear,
     the mean stated probability against the share of picks where that
     scenario actually landed. The prompt forces bear >= 10% on every
     pick; this is the check on whether 10% was the right floor.

Two horizons are in play and they are deliberately kept apart. EV targets
are quoted over the ranker's own 6-12 month horizon, so EV error is only
scored once `_EV_HORIZON_DAYS` have elapsed — comparing a 9-month forecast
to a 90-day outcome would manufacture a bias that isn't there. Conviction
ordering, by contrast, should be visible at any horizon, so it is measured
on the same 90-day window the track record uses and labeled as such.

Picks written before the forecast columns existed carry NULLs and are
excluded rather than guessed at.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ..db.session import get_session
from ..logging import get_logger
from ..models.calibration import (
    CalibrationRecord,
    ConvictionBucket,
    EVError,
    ScenarioReliability,
)
from .track_record import _close_on_or_after, _close_on_or_before, _fetch_history

logger = get_logger(__name__)

# The midpoint of the ranker's stated "6-12 months" horizon. EV error is
# only computed for picks at least this old.
_EV_HORIZON_DAYS = 270
# Conviction ordering is measured on the same window the track record
# headlines, so the two numbers in the prompt are comparable.
_CONVICTION_HORIZON_DAYS = 90
# Conviction buckets. Kept coarse on purpose: per-integer buckets would be
# n=1 or 2 for a long time and read as noise.
_CONVICTION_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("low 1-5", 1, 5),
    ("mid 6-7", 6, 7),
    ("high 8-10", 8, 10),
)
_MIN_BUCKET_N = 3
_MAX_WORKERS = 6


@dataclass(frozen=True)
class _Forecast:
    """One persisted pick's forecast, plus the date it was made."""

    ticker: str
    pick_date: str
    age_days: int
    conviction: int | None
    ev_pct: float | None
    entry_price: float | None
    scenarios: dict[str, tuple[float, float]]  # label -> (probability, target)


# --- DB read ---------------------------------------------------------------


def _load_forecasts(db_path: str, lookback_days: int) -> list[_Forecast]:
    """Every pick with a recorded forecast, newest runs included.

    Deduplicated to the OLDEST decision per ticker, matching the
    track-record convention: the first time the system told you to buy
    something is the call that gets graded.
    """
    from sqlalchemy import text

    cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
    rows: list[tuple] = []
    try:
        with get_session(db_path) as session:
            rows = list(
                session.exec(
                    text(
                        "SELECT r.run_at, p.run_id, p.rank, p.ticker, p.conviction, "
                        "       p.ev_pct, p.entry_price "
                        "FROM picks p JOIN runs r ON r.id = p.run_id "
                        "WHERE r.run_at >= :cutoff "
                        "ORDER BY r.run_at ASC"
                    ),
                    params={"cutoff": cutoff},
                )
            )
            scenario_rows = list(
                session.exec(
                    text(
                        "SELECT run_id, rank, label, probability, target_return_pct "
                        "FROM pick_scenarios"
                    ),
                )
            )
    except Exception as e:
        logger.warning("calibration fetch failed (%s) — returning empty", e)
        return []

    by_pick: dict[tuple[int, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    for run_id, rank, label, probability, target in scenario_rows:
        by_pick[(run_id, rank)][str(label)] = (float(probability), float(target))

    today = date.today()
    seen: set[str] = set()
    out: list[_Forecast] = []
    for run_at, run_id, rank, ticker, conviction, ev_pct, entry_price in rows:
        if ticker in seen:
            continue
        seen.add(ticker)
        try:
            pick_date = datetime.fromisoformat(str(run_at)).date()
        except ValueError:
            continue
        out.append(
            _Forecast(
                ticker=str(ticker),
                pick_date=pick_date.isoformat(),
                age_days=(today - pick_date).days,
                conviction=int(conviction) if conviction is not None else None,
                ev_pct=float(ev_pct) if ev_pct is not None else None,
                entry_price=float(entry_price) if entry_price is not None else None,
                scenarios=dict(by_pick.get((run_id, rank), {})),
            )
        )
    return out


# --- realized returns -----------------------------------------------------


def _realized_return_pct(forecast: _Forecast, horizon_days: int) -> float | None:
    """Total return from the decision date to the horizon anniversary.

    Entry uses the stored `entry_price` when present — the price the screen
    actually saw — falling back to the first close at/after the decision
    date. Either way the entry is point-in-time; nothing here reprices a
    historical pick with today's fundamentals.
    """
    if forecast.age_days < horizon_days:
        return None
    pick_date = date.fromisoformat(forecast.pick_date)
    frame = _fetch_history(
        forecast.ticker,
        pick_date - timedelta(days=5),
        pick_date + timedelta(days=horizon_days),
    )
    if frame is None or frame.empty:
        return None
    closes = frame["Close"].dropna()
    exit_ = _close_on_or_before(closes, pick_date + timedelta(days=horizon_days))
    if exit_ is None:
        return None
    entry_price = forecast.entry_price
    if entry_price is None or entry_price <= 0:
        entry = _close_on_or_after(closes, pick_date)
        if entry is None:
            return None
        entry_price = entry[0]
    if entry_price <= 0:
        return None
    return (exit_[0] / entry_price - 1) * 100


def _spy_return_pct(pick_date: str, horizon_days: int) -> float | None:
    start = date.fromisoformat(pick_date)
    frame = _fetch_history("SPY", start - timedelta(days=5), start + timedelta(days=horizon_days))
    if frame is None or frame.empty:
        return None
    closes = frame["Close"].dropna()
    entry = _close_on_or_after(closes, start)
    exit_ = _close_on_or_before(closes, start + timedelta(days=horizon_days))
    if entry is None or exit_ is None or entry[0] <= 0:
        return None
    return (exit_[0] / entry[0] - 1) * 100


# --- scoring --------------------------------------------------------------


def _which_scenario_landed(
    realized_pct: float, scenarios: dict[str, tuple[float, float]]
) -> str | None:
    """The scenario whose target return is closest to what happened.

    Nearest-target attribution, not a threshold rule, because the targets
    themselves are what the ranker committed to — it keeps the check
    honest even when a pick's bull/base targets are unusually close
    together.
    """
    if not scenarios:
        return None
    return min(scenarios.items(), key=lambda kv: abs(realized_pct - kv[1][1]))[0]


def measure_calibration(db_path: str, *, lookback_days: int = 540) -> CalibrationRecord:
    """Score every persisted forecast whose horizon has elapsed."""
    forecasts = _load_forecasts(db_path, lookback_days)
    if not forecasts:
        return CalibrationRecord()

    ev_candidates = [
        f for f in forecasts if f.ev_pct is not None and f.age_days >= _EV_HORIZON_DAYS
    ]
    conviction_candidates = [
        f for f in forecasts if f.conviction is not None and f.age_days >= _CONVICTION_HORIZON_DAYS
    ]
    n_pending = sum(1 for f in forecasts if f.ev_pct is not None and f.age_days < _EV_HORIZON_DAYS)
    n_no_forecast = sum(1 for f in forecasts if f.ev_pct is None)

    # --- EV error + scenario reliability (EV horizon) ---
    ev_errors: list[EVError] = []
    landed: dict[str, int] = defaultdict(int)
    stated: dict[str, list[float]] = defaultdict(list)
    # Results are zipped positionally: _Forecast carries a dict field, so it
    # is not hashable and cannot key a lookup.
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        realized = list(ex.map(lambda f: _realized_return_pct(f, _EV_HORIZON_DAYS), ev_candidates))
    for forecast, got in zip(ev_candidates, realized, strict=True):
        if got is None or forecast.ev_pct is None:
            continue
        ev_errors.append(
            EVError(
                ticker=forecast.ticker,
                pick_date=forecast.pick_date,
                conviction=forecast.conviction,
                ev_pct=forecast.ev_pct,
                realized_pct=got,
                error_pct=got - forecast.ev_pct,
            )
        )
        winner = _which_scenario_landed(got, forecast.scenarios)
        if winner is not None:
            landed[winner] += 1
        for label, (probability, _target) in forecast.scenarios.items():
            stated[label].append(probability)

    reliability: list[ScenarioReliability] = []
    total_landed = sum(landed.values())
    for label in ("bull", "base", "bear"):
        probabilities = stated.get(label) or []
        if not probabilities:
            continue
        reliability.append(
            ScenarioReliability(
                label=label,
                n=len(probabilities),
                mean_stated_probability=sum(probabilities) / len(probabilities),
                observed_frequency=(landed.get(label, 0) / total_landed if total_landed else None),
                n_landed=landed.get(label, 0),
            )
        )

    # --- conviction ordering (track-record horizon) ---
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        conv_returns = list(
            ex.map(
                lambda f: _realized_return_pct(f, _CONVICTION_HORIZON_DAYS),
                conviction_candidates,
            )
        )
    spy_cache: dict[str, float | None] = {}
    buckets: list[ConvictionBucket] = []
    grouped: dict[str, list[float]] = defaultdict(list)
    for forecast, got in zip(conviction_candidates, conv_returns, strict=True):
        if got is None or forecast.conviction is None:
            continue
        if forecast.pick_date not in spy_cache:
            spy_cache[forecast.pick_date] = _spy_return_pct(
                forecast.pick_date, _CONVICTION_HORIZON_DAYS
            )
        spy = spy_cache[forecast.pick_date]
        if spy is None:
            continue
        for label, lo, hi in _CONVICTION_BUCKETS:
            if lo <= forecast.conviction <= hi:
                grouped[label].append(got - spy)
                break
    for label, _lo, _hi in _CONVICTION_BUCKETS:
        alphas = grouped.get(label) or []
        if len(alphas) < _MIN_BUCKET_N:
            continue
        buckets.append(
            ConvictionBucket(
                label=label,
                n=len(alphas),
                mean_alpha_pct=sum(alphas) / len(alphas),
            )
        )

    record = CalibrationRecord(
        ev_horizon_days=_EV_HORIZON_DAYS,
        conviction_horizon_days=_CONVICTION_HORIZON_DAYS,
        n_scored=len(ev_errors),
        n_pending=n_pending,
        n_without_forecast=n_no_forecast,
        mean_ev_error_pct=(
            statistics.mean([e.error_pct for e in ev_errors]) if ev_errors else None
        ),
        median_ev_error_pct=(
            statistics.median([e.error_pct for e in ev_errors]) if ev_errors else None
        ),
        ev_errors=sorted(ev_errors, key=lambda e: e.error_pct),
        conviction_buckets=buckets,
        scenario_reliability=reliability,
    )
    logger.info(
        "Calibration: %d forecast(s) scored at %dd, %d pending, "
        "%d without a recorded forecast; mean EV error=%s",
        record.n_scored,
        _EV_HORIZON_DAYS,
        record.n_pending,
        record.n_without_forecast,
        f"{record.mean_ev_error_pct:+.1f}%" if record.mean_ev_error_pct is not None else "n/a",
    )
    return record


# --- formatter ------------------------------------------------------------


def format_calibration_block(record: CalibrationRecord) -> str:
    """Prompt-ready block: how well this system's own forecasts have held up.

    Returns "" when there is nothing measured yet, so the ranker prompt
    simply omits the section rather than carrying an empty header.
    """
    lines: list[str] = []

    if record.n_scored:
        lines.append(
            f"=== Your forecast calibration "
            f"({record.n_scored} picks scored at {record.ev_horizon_days}d) ==="
        )
        mean_err = record.mean_ev_error_pct
        median_err = record.median_ev_error_pct
        if mean_err is not None and median_err is not None:
            direction = (
                "you OVERSHOT (realized came in below EV)"
                if mean_err < 0
                else "you UNDERSHOT (realized came in above EV)"
            )
            lines.append(
                f"EV error (realized - EV): mean {mean_err:+.1f}%, "
                f"median {median_err:+.1f}% — {direction}."
            )
        if record.scenario_reliability:
            parts = []
            for s in record.scenario_reliability:
                observed = (
                    f"{s.observed_frequency:.0%}" if s.observed_frequency is not None else "n/a"
                )
                parts.append(
                    f"{s.label}: said {s.mean_stated_probability:.0%} avg, "
                    f"landed {observed} ({s.n_landed}/{s.n})"
                )
            lines.append("Scenario reliability — " + "; ".join(parts) + ".")
    elif record.n_pending or record.n_without_forecast:
        bits = []
        if record.n_pending:
            bits.append(
                f"{record.n_pending} forecast(s) not yet at the {record.ev_horizon_days}d horizon"
            )
        if record.n_without_forecast:
            bits.append(f"{record.n_without_forecast} older pick(s) with no recorded forecast")
        lines.append("=== Your forecast calibration ===")
        lines.append("Not scorable yet: " + "; ".join(bits) + ".")

    if record.conviction_buckets:
        lines.append(
            f"Conviction vs realized alpha at {record.conviction_horizon_days}d: "
            + " | ".join(
                f"{b.label}: {b.mean_alpha_pct:+.1f}% (n={b.n})" for b in record.conviction_buckets
            )
        )
        if not record.is_conviction_monotone:
            lines.append(
                "WARNING: your conviction scores are NOT ordered by realized "
                "alpha — higher conviction did not produce better outcomes. "
                "Treat your own conviction number with suspicion and justify "
                "it from forward evidence rather than confidence."
            )

    return "\n".join(lines)
