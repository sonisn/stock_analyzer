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
import polars as pl
from dateutil.relativedelta import relativedelta

from ..data.frames import DATE
from ..db.session import get_session
from ..db.tables import ModelVersion
from ..discover.screen import _score_trend
from .features import FEATURES
from .fundamental_features import FUNDAMENTAL_FEATURES

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


def _rank_pct(col: pl.Expr) -> pl.Expr:
    """pandas' groupby-rank(pct=True) per date: average rank over the count
    of present values; missing stays missing."""
    c = col.fill_nan(None)
    return c.rank("average").over(DATE) / c.count().over(DATE)


def rank_center(frame: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Per-date percentile rank minus 0.5; missing values sit at 0."""
    return frame.select([(_rank_pct(pl.col(c)) - 0.5).fill_null(0.0).alias(c) for c in cols])


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float = RIDGE_ALPHA) -> np.ndarray:
    n, k = x.shape
    lam = alpha * n / 12.0
    return np.linalg.solve(x.T @ x + lam * np.eye(k), x.T @ y)


def baseline_trend_score(frame: pl.DataFrame) -> np.ndarray:
    """The screen's price-only trend points (screen._score_trend without the
    EPS-revision term, which has no history) for every row."""
    records = frame.select("rs_6mo", "dist_from_52w_high", "volume_trend_20_60", "weekly_rsi")
    return np.array([_score_trend(r)[0] for r in records.to_dicts()], dtype=float)


def _daily_ic(dates: np.ndarray, pred: np.ndarray, label: np.ndarray) -> pl.DataFrame:
    """(date, ic): per date with 5+ names, the rank correlation of `pred`
    with `label`. Rows missing either are left out first."""
    df = pl.DataFrame({DATE: dates, "p": pred, "y": label}).filter(
        pl.col("p").is_not_nan() & pl.col("y").is_not_nan()
    )
    ranked = df.with_columns(
        pl.col("p").rank("average").over(DATE).alias("rp"),
        pl.col("y").rank("average").over(DATE).alias("ry"),
    )
    out = (
        ranked.group_by(DATE)
        .agg(pl.len().alias("n"), pl.corr("rp", "ry").alias("ic"))
        .filter((pl.col("n") >= MIN_NAMES_TO_SCORE) & pl.col("ic").is_not_null())
        .filter(pl.col("ic").is_not_nan())
        .sort(DATE)
    )
    return out.select(DATE, "ic")


def ic_stats(ic: pl.DataFrame | np.ndarray, horizon: int) -> dict[str, float | int | None]:
    """Mean weekly IC, its IR, and a t-statistic that counts overlapping
    weekly labels as ~horizon/5 times fewer independent observations."""
    values = ic["ic"].to_numpy() if isinstance(ic, pl.DataFrame) else np.asarray(ic, float)
    n = len(values)
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    if n < 3 or sd == 0:
        return {"ic_mean": None, "ic_ir": None, "t_stat": None, "hit_rate": None, "weeks": n}
    overlap = max(1.0, horizon / 5.0)
    n_eff = n / overlap
    mean = float(values.mean())
    return {
        "ic_mean": mean,
        "ic_ir": float(mean / sd),
        "t_stat": float(mean / sd * math.sqrt(n_eff)),
        "hit_rate": float((values > 0).mean()),
        "weeks": n,
    }


def _quintiles(ranks: np.ndarray) -> np.ndarray:
    """pd.qcut(ranks, 5, labels=False) + 1: equal-count bins, right-closed,
    the lowest edge included."""
    edges = np.quantile(ranks, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    return np.searchsorted(edges[1:-1], ranks, side="left") + 1


def quintile_spread(
    dates: np.ndarray, pred: np.ndarray, label: np.ndarray
) -> dict[str, float | None]:
    """Mean forward excess return of each score quintile (Q5 = best), and
    the top-minus-bottom spread, averaged over weeks."""
    df = pl.DataFrame({DATE: dates, "p": pred, "y": label}).filter(
        pl.col("p").is_not_nan() & pl.col("y").is_not_nan()
    )
    if df.is_empty():
        return {"spread": None}
    parts = []
    for (day,), g in df.group_by(DATE, maintain_order=True):
        if g.height < 5:
            continue
        ranks = g["p"].rank("ordinal").to_numpy().astype(float)
        parts.append(g.with_columns(pl.Series("q", _quintiles(ranks)), pl.lit(day).alias(DATE)))
    if not parts:
        return {"spread": None}
    by_q = pl.concat(parts).group_by(DATE, "q").agg(pl.col("y").mean())
    wide = by_q.pivot(on="q", index=DATE, values="y")
    qs = sorted(int(c) for c in wide.columns if c != DATE)
    out: dict[str, float | None] = {
        f"q{q}": float(wide[str(q)].drop_nulls().to_numpy().mean()) for q in qs
    }
    if 5 in qs and 1 in qs:
        diff = (wide["5"] - wide["1"]).drop_nulls().to_numpy()
        out["spread"] = float(diff.mean()) if len(diff) else None
    else:
        out["spread"] = None
    return out


def _as_days(values) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        values = values[DATE]
    if isinstance(values, pl.Series):
        values = values.to_numpy()
    return np.asarray(values).astype("datetime64[D]")


def walk_forward(
    data: pl.DataFrame,
    calendar,
    *,
    horizon: int,
    population: str = "gated",
    label_kind: str = "excess",
) -> ModelResult:
    """`calendar` is the trading days (SPY's), as dates."""
    label = f"fwd_{horizon}" if label_kind == "excess" else f"fwd_{horizon}_badj"
    frame = data.filter(pl.col("gated")) if population == "gated" else data
    # Fundamentals join the feature set only when the dataset carries
    # them, so a price-only run behaves exactly as it did before.
    features = [*FEATURES, *(c for c in FUNDAMENTAL_FEATURES if c in data.columns)]
    x_all = rank_center(frame, features).to_numpy()
    label_values = frame[label].to_numpy().astype(float)
    labeled = ~np.isnan(label_values)
    y_all = frame.select((_rank_pct(pl.col(label)) - 0.5).alias("y"))["y"].to_numpy()
    y_all = np.where(labeled, y_all.astype(float), np.nan)

    dates = _as_days(frame)
    cal = _as_days(calendar)
    # A training row is usable for a test block only once its label window
    # (next bar + horizon bars) has closed before the block begins.
    pos = np.searchsorted(cal, dates)
    label_end = cal[np.minimum(pos + 1 + horizon, len(cal) - 1)]

    first = dates.min().astype(object)
    start = first + relativedelta(years=MIN_TRAIN_YEARS)
    last_labeled = dates[labeled].max().astype(object)
    pred_rows: list[np.ndarray] = []
    pred_vals: list[np.ndarray] = []
    fold_coefs: list[dict[str, float]] = []
    while start <= last_labeled:
        end = start + relativedelta(months=TEST_MONTHS)
        s64, e64 = np.datetime64(start), np.datetime64(end)
        train = (label_end < s64) & labeled
        test = (dates >= s64) & (dates < e64)
        if train.sum() > 1000 and test.any():
            w = fit_ridge(x_all[train], y_all[train])
            fold_coefs.append(dict(zip(features, map(float, w), strict=True)))
            rows = np.flatnonzero(test)
            pred_rows.append(rows)
            pred_vals.append(x_all[rows] @ w)
        start = end
    if not pred_rows:
        raise RuntimeError("Not enough history for a walk-forward test")
    rows = np.concatenate(pred_rows)
    oos = np.concatenate(pred_vals)
    oos_dates = dates[rows]
    oos_label = label_values[rows]
    oos_frame = frame[rows.tolist()]
    base = baseline_trend_score(oos_frame)

    model_ic = _daily_ic(oos_dates, oos, oos_label)
    base_ic = _daily_ic(oos_dates, base, oos_label)
    per_feature = {
        name: ic_stats(
            _daily_ic(oos_dates, oos_frame[name].to_numpy().astype(float), oos_label), horizon
        )["ic_mean"]
        for name in features
    }
    m, b = ic_stats(model_ic, horizon), ic_stats(base_ic, horizon)
    both = model_ic.join(base_ic, on=DATE, how="inner", suffix="_b")
    diff = ic_stats((both["ic"] - both["ic_b"]).to_numpy(), horizon)
    metrics = {
        "horizon_days": horizon,
        "population": population,
        "label": label_kind,
        "rows": int(frame.height),
        "oos_rows": int((~np.isnan(oos_label)).sum()),
        "oos_start": str(oos_dates.min()),
        "oos_end": str(oos_dates.max()),
        "model": {**m, **quintile_spread(oos_dates, oos, oos_label)},
        "baseline_trend_score": {**b, **quintile_spread(oos_dates, base, oos_label)},
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
    w = fit_ridge(x_all[labeled], y_all[labeled])
    labeled_dates = dates[labeled]
    return ModelResult(
        horizon=horizon,
        population=population,
        coefficients=dict(zip(features, map(float, w), strict=True)),
        metrics=metrics,
        train_start=str(labeled_dates.min()),
        train_end=str(labeled_dates.max()),
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
    from sqlmodel import col, select

    with get_session(db_path) as session:
        row = session.exec(
            select(ModelVersion)
            .where(ModelVersion.accepted == 1)
            .order_by(col(ModelVersion.id).desc())
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
    from sqlmodel import col, select

    with get_session(db_path) as session:
        row = session.exec(select(ModelVersion).order_by(col(ModelVersion.id).desc())).first()
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
    tickers = list(features_by_ticker)
    frame = pl.DataFrame(
        {
            DATE: ["today"] * len(tickers),
            **{
                n: [
                    float("nan") if (v := features_by_ticker[t].get(n)) is None else float(v)
                    for t in tickers
                ]
                for n in names
            },
        }
    )
    x = rank_center(frame, names).to_numpy()
    raw = x @ np.array([model.coefficients[n] for n in names])
    pct = pl.Series(raw).rank("average").to_numpy() / len(raw) * 100
    return {t: float(v) for t, v in zip(tickers, pct, strict=True)}


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
