"""Cross-sectional ranking model for forward excess return.

Each week, every feature and the label are converted to cross-sectional
percentile ranks centred on zero, and a ridge regression learns one weight
per feature. Ranks make the model robust to outliers and regime-level
shifts (only the ordering within a week matters), and a linear model with
eleven weights is hard to overfit and easy to read: the coefficients say
which signals the history rewarded and with what sign.

Validation is walk-forward. Each 6-month test block is predicted by a
model trained only on earlier weeks whose labels had fully resolved
before the block starts (a purge of `horizon + 1` trading days), because
labels on consecutive weeks overlap and would otherwise leak the test
period into training. The out-of-sample information coefficient is
compared with the screen's existing price-based trend score on exactly the
same rows; the model is only marked `accepted` — and only then used by the
screen — if it beats that baseline with a t-statistic above 2 after
correcting for the label overlap.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ..db.session import get_session
from ..db.tables import ModelVersion
from ..discover.screen import _score_trend
from .features import FEATURES

RIDGE_ALPHA = 0.1  # shrinkage relative to a rank feature's variance (1/12)
TEST_MONTHS = 6
MIN_TRAIN_YEARS = 2
MIN_T_STAT = 2.0
MIN_NAMES_TO_SCORE = 5
MAX_SCREEN_POINTS = 5.0


@dataclass
class ModelResult:
    horizon: int
    population: str
    coefficients: dict[str, float]
    metrics: dict[str, Any]
    train_start: str
    train_end: str
    accepted: bool
    fold_coefficients: list[dict[str, float]] = field(default_factory=list)


def rank_center(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Per-date percentile rank minus 0.5; missing values sit at 0."""
    ranked = frame[cols].groupby(level="date").rank(pct=True) - 0.5
    return ranked.fillna(0.0)


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    n, k = x.shape
    lam = alpha * n / 12.0
    return np.linalg.solve(x.T @ x + lam * np.eye(k), x.T @ y)


def baseline_trend_score(frame: pd.DataFrame) -> pd.Series:
    """The screen's price-only trend points (screen._score_trend without the
    EPS-revision term, which has no history) for every row."""
    records = frame[["rs_6mo", "dist_from_52w_high", "volume_trend_20_60", "weekly_rsi"]]
    return pd.Series(
        [_score_trend(r)[0] for r in records.to_dict("records")],
        index=frame.index,
        dtype=float,
    )


def _daily_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    ranks = df.groupby(level="date").rank()
    return (
        ranks.groupby(level="date")
        .apply(lambda g: g["p"].corr(g["y"]) if len(g) >= MIN_NAMES_TO_SCORE else np.nan)
        .dropna()
    )


def ic_stats(ic: pd.Series, horizon: int) -> dict[str, float | int | None]:
    """Mean weekly IC, its IR, and a t-statistic that counts overlapping
    weekly labels as ~horizon/5 times fewer independent observations."""
    n = len(ic)
    if n < 3 or ic.std() == 0:
        return {"ic_mean": None, "ic_ir": None, "t_stat": None, "hit_rate": None, "weeks": n}
    overlap = max(1.0, horizon / 5.0)
    n_eff = n / overlap
    return {
        "ic_mean": float(ic.mean()),
        "ic_ir": float(ic.mean() / ic.std()),
        "t_stat": float(ic.mean() / ic.std() * math.sqrt(n_eff)),
        "hit_rate": float((ic > 0).mean()),
        "weeks": n,
    }


def quintile_spread(pred: pd.Series, label: pd.Series) -> dict[str, float | None]:
    """Mean forward excess return of each score quintile (Q5 = best), and
    the top-minus-bottom spread, averaged over weeks."""
    df = pd.DataFrame({"p": pred, "y": label}).dropna()
    if df.empty:
        return {"spread": None}
    df["q"] = df.groupby(level="date")["p"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 5, labels=False) + 1 if len(s) >= 5 else np.nan
    )
    by_q = df.dropna(subset=["q"]).groupby(["date", "q"])["y"].mean().unstack()
    means = by_q.mean()
    out: dict[str, float | None] = {f"q{int(q)}": float(v) for q, v in means.items()}
    out["spread"] = float((by_q[5] - by_q[1]).mean()) if 5 in by_q and 1 in by_q else None
    return out


def walk_forward(
    data: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    *,
    horizon: int,
    population: str = "gated",
    label_kind: str = "excess",
) -> ModelResult:
    label = f"fwd_{horizon}" if label_kind == "excess" else f"fwd_{horizon}_badj"
    frame = data[data["gated"]] if population == "gated" else data
    features = list(FEATURES)
    x_all = rank_center(frame, features)
    labeled = frame[label].notna()
    y_all = (frame[label].groupby(level="date").rank(pct=True) - 0.5).where(labeled)

    dates = frame.index.get_level_values("date")
    # A training row is usable for a test block only once its label window
    # (next bar + horizon bars) has closed before the block begins.
    pos = calendar.searchsorted(dates)
    label_end = calendar[np.minimum(pos + 1 + horizon, len(calendar) - 1)]

    first = dates.min()
    start = first + pd.DateOffset(years=MIN_TRAIN_YEARS)
    last_labeled = dates[labeled.to_numpy()].max()
    preds: list[pd.Series] = []
    fold_coefs: list[dict[str, float]] = []
    while start <= last_labeled:
        end = start + pd.DateOffset(months=TEST_MONTHS)
        train = (label_end < start) & labeled.to_numpy()
        test = (dates >= start) & (dates < end)
        if train.sum() > 1000 and test.any():
            w = fit_ridge(x_all[train].to_numpy(), y_all[train].to_numpy())
            fold_coefs.append(dict(zip(features, map(float, w), strict=True)))
            preds.append(pd.Series(x_all[test].to_numpy() @ w, index=x_all[test].index))
        start = end
    if not preds:
        raise RuntimeError("Not enough history for a walk-forward test")
    oos = pd.concat(preds)
    oos_label = frame.loc[oos.index, label]
    base = baseline_trend_score(frame.loc[oos.index])

    model_ic = _daily_ic(oos, oos_label)
    base_ic = _daily_ic(base, oos_label)
    per_feature = {
        name: ic_stats(_daily_ic(frame.loc[oos.index, name], oos_label), horizon)["ic_mean"]
        for name in features
    }
    m, b = ic_stats(model_ic, horizon), ic_stats(base_ic, horizon)
    diff = ic_stats((model_ic - base_ic).dropna(), horizon)
    metrics = {
        "horizon_days": horizon,
        "population": population,
        "label": label_kind,
        "rows": int(len(frame)),
        "oos_rows": int(oos_label.notna().sum()),
        "oos_start": str(oos.index.get_level_values("date").min().date()),
        "oos_end": str(oos.index.get_level_values("date").max().date()),
        "model": {**m, **quintile_spread(oos, oos_label)},
        "baseline_trend_score": {**b, **quintile_spread(base, oos_label)},
        "model_minus_baseline": diff,
        "feature_ic": per_feature,
    }
    accepted = bool(
        m["ic_mean"] is not None
        and b["ic_mean"] is not None
        and m["ic_mean"] > max(0.0, b["ic_mean"])
        and (m["t_stat"] or 0) >= MIN_T_STAT
        and (diff["t_stat"] or 0) > 0
    )

    # Final weights: every labeled row, for use from here on.
    w = fit_ridge(x_all[labeled].to_numpy(), y_all[labeled].to_numpy())
    labeled_dates = dates[labeled.to_numpy()]
    return ModelResult(
        horizon=horizon,
        population=population,
        coefficients=dict(zip(features, map(float, w), strict=True)),
        metrics=metrics,
        train_start=str(labeled_dates.min().date()),
        train_end=str(labeled_dates.max().date()),
        accepted=accepted,
        fold_coefficients=fold_coefs,
    )


# --- persistence + live scoring -----------------------------------------------


def save_model(db_path: str, result: ModelResult) -> int:
    with get_session(db_path) as session:
        row = ModelVersion(
            created_at=datetime.now().isoformat(timespec="seconds"),
            horizon_days=result.horizon,
            population=result.population,
            features=json.dumps(list(result.coefficients)),
            coefficients=json.dumps(result.coefficients),
            metrics=json.dumps({**result.metrics, "fold_coefficients": result.fold_coefficients}),
            train_start=result.train_start,
            train_end=result.train_end,
            accepted=int(result.accepted),
        )
        session.add(row)
        session.flush()
        return int(row.id or 0)


@dataclass(frozen=True)
class ActiveModel:
    version: int
    horizon: int
    coefficients: dict[str, float]
    accepted: bool = True


def load_active_model(db_path: str) -> ActiveModel | None:
    """Latest ACCEPTED model version, or None (the screen then ignores it)."""
    from sqlmodel import select

    with get_session(db_path) as session:
        row = session.exec(
            select(ModelVersion).where(ModelVersion.accepted == 1).order_by(ModelVersion.id.desc())  # type: ignore[union-attr]
        ).first()
        if row is None:
            return None
        return ActiveModel(int(row.id or 0), row.horizon_days, json.loads(row.coefficients))


def load_latest_model(db_path: str) -> ActiveModel | None:
    """The accepted model if there is one, otherwise the newest model of
    any kind (the screen then records its scores in shadow only)."""
    active = load_active_model(db_path)
    if active is not None:
        return active
    from sqlmodel import select

    with get_session(db_path) as session:
        row = session.exec(
            select(ModelVersion).order_by(ModelVersion.id.desc())  # type: ignore[union-attr]
        ).first()
        if row is None:
            return None
        return ActiveModel(
            int(row.id or 0), row.horizon_days, json.loads(row.coefficients), accepted=False
        )


def score_percentiles(
    model: ActiveModel, features_by_ticker: dict[str, dict[str, float | None]]
) -> dict[str, float]:
    """Model percentile (0-100) for each ticker, ranked within this set —
    the same within-week ranking the model was trained on. Empty when there
    are too few names for a ranking to mean anything."""
    if len(features_by_ticker) < MIN_NAMES_TO_SCORE:
        return {}
    names = list(model.coefficients)
    frame = pd.DataFrame.from_dict(features_by_ticker, orient="index").reindex(columns=names)
    frame.index = pd.MultiIndex.from_product([["today"], frame.index], names=["date", "ticker"])
    x = rank_center(frame.astype(float), names)
    raw = x.to_numpy() @ np.array([model.coefficients[n] for n in names])
    pct = pd.Series(raw, index=frame.index.get_level_values("ticker")).rank(pct=True) * 100
    return {t: float(v) for t, v in pct.items()}


def screen_points(percentile: float) -> float:
    """Composite-score adjustment: ±MAX_SCREEN_POINTS from the best to the
    worst model percentile, 0 at the median."""
    return round(MAX_SCREEN_POINTS * (percentile / 50.0 - 1.0), 2)


def format_model_report(result: ModelResult) -> str:
    m = result.metrics

    def f(v: Any, fmt: str) -> str:
        return format(v, fmt) if isinstance(v, (int, float)) else "n/a"

    def row(label: str, s: dict[str, Any]) -> str:
        return (
            f"  {label:<22} IC {f(s.get('ic_mean'), '+.4f')}  IR {f(s.get('ic_ir'), '+.2f')}  "
            f"t {f(s.get('t_stat'), '+.2f')}  hit {f(s.get('hit_rate'), '.0%')}  "
            f"Q5-Q1 {f(s.get('spread'), '+.2%')}"
        )

    lines = [
        f"Forward-return model — {result.horizon}-day horizon, "
        f"population={result.population}, label={m.get('label', 'excess')}",
        f"  trained {result.train_start} .. {result.train_end}; "
        f"out-of-sample {m['oos_start']} .. {m['oos_end']} ({m['oos_rows']:,} rows)",
        row("model", m["model"]),
        row("screen trend score", m["baseline_trend_score"]),
        row("model - baseline", m["model_minus_baseline"]),
        "  quintile mean excess return (model, Q1 worst .. Q5 best): "
        + "  ".join(f"{q}: {f(m['model'].get(q), '+.2%')}" for q in ("q1", "q2", "q3", "q4", "q5")),
        "  per-feature out-of-sample IC and final weight:",
    ]
    for name, ic in sorted(m["feature_ic"].items(), key=lambda kv: -abs(kv[1] or 0)):
        lines.append(f"    {name:<20} IC {f(ic, '+.4f')}   weight {result.coefficients[name]:+.4f}")
    lines.append(
        f"  verdict: {'ACCEPTED — the screen will use it' if result.accepted else 'NOT accepted — the screen ignores it'}"
        f" (needs OOS IC > max(0, baseline), t >= {MIN_T_STAT:.0f}, and beating the baseline on average)"
    )
    return "\n".join(lines)
