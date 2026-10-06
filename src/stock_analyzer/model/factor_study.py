"""Offline studies over fifteen years of month-ends: does a measure rank
next six to twelve months' returns, and does sizing by risk help?

`fundamentals_study`: each of the screen's fundamental measures, plus two
it does not use (gross profitability, the earnings surprise), read
point-in-time from SEC filings (`sec_history`), against the excess return
over SPY from the next session's close 126 and 252 sessions on — raw and
beta-adjusted, since an "edge" that is only market exposure vanishes under
the second (the price model's did, 2026-09-18). Also the screen's own
0-45 fundamentals points and its hard rules, computed with screen.py's
functions on the same figures.

`risk_study`: whether volatility is predictable (it is the input to any
risk-based sizing), the low-volatility quintile spread, and monthly
portfolios — equal weight against inverse-volatility weight, over all
names and over the screen's trend-gated names — against SPY.

Both use today's S&P 500 members, so names that fell out of the index are
missing and absolute returns look better than they were. Rankings within a
month are less affected, which is what the information coefficient (IC:
per-month Spearman correlation of measure and outcome) measures — but not
immune: today's list also holds small companies that later grew into it,
the winners of their day. So every fundamental measure is also graded
among companies already worth LARGE_THEN_CAP at the time, and a verdict
that only holds across all names is reported as "survivorship?". The
first run showed why: operating margin "pointed the wrong way" (252-day IC
-0.067, t -3.2) and among companies already $20B+ it was noise (-0.026,
t -1.3) — small low-margin names that later made the index had won.
Months overlap at these horizons, so t-statistics use Newey-West errors
with a lag of one horizon. No LLM calls.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import polars as pl

from ..data.frames import DATE
from .dataset import PricePanel, forward_returns
from .features import panel_features

HORIZONS = (126, 252)
LARGE_THEN_CAP = 20e9  # the survivorship check: already this big on the date
MIN_NAMES = 30  # a month with fewer names carrying the measure is skipped
QUINTILES = 5


# --- statistics -----------------------------------------------------------------


def newey_west_t(x: np.ndarray, lags: int) -> float | None:
    """t-statistic of the mean of `x` with Newey-West (Bartlett) errors."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return None
    e = x - x.mean()
    var = float(e @ e) / n
    for lag in range(1, min(lags, n - 1) + 1):
        w = 1 - lag / (lags + 1)
        var += 2 * w * float(e[lag:] @ e[:-lag]) / n
    if var <= 0:
        return None
    return float(x.mean() / math.sqrt(var / n))


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float | None:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < MIN_NAMES:
        return None
    ra = pl.Series(a[ok]).rank().to_numpy()
    rb = pl.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


@dataclass
class FactorResult:
    factor: str
    label: str
    months: int
    names: float  # average names per month
    mean_ic: float | None
    t: float | None
    positive: float | None  # share of months with IC > 0
    first_half: float | None
    second_half: float | None
    spread: float | None  # top-quintile minus bottom-quintile mean outcome, in %
    large: FactorResult | None = None  # the same, among companies already large

    @property
    def own_verdict(self) -> str:
        if self.mean_ic is None or self.t is None:
            return "no data"
        halves = (self.first_half or 0) * (self.second_half or 0) > 0
        if abs(self.t) >= 2 and halves:
            return "works" if self.mean_ic > 0 else "WRONG WAY"
        if abs(self.t) >= 2:
            return "one era only"
        return "no edge"

    @property
    def verdict(self) -> str:
        """`own_verdict`, unless a result across all names does not hold
        among the companies that were already large."""
        own = self.own_verdict
        if (
            own in ("works", "WRONG WAY")
            and self.large is not None
            and self.large.own_verdict != own
        ):
            return "survivorship?"
        return own


def evaluate(frame: pl.DataFrame, factor: str, label: str, horizon: int) -> FactorResult:
    """IC of `factor` against `label` per month, across `frame` (date,
    ticker, factor, label)."""
    ics: list[tuple[date, float]] = []
    spreads: list[float] = []
    names: list[int] = []
    for (day,), g in frame.select(DATE, factor, label).group_by(DATE, maintain_order=True):
        a = g[factor].cast(pl.Float64).fill_null(np.nan).to_numpy()
        b = g[label].cast(pl.Float64).fill_null(np.nan).to_numpy()
        ic = _rank_ic(a, b)
        if ic is None:
            continue
        ics.append((day, ic))
        ok = np.isfinite(a) & np.isfinite(b)
        names.append(int(ok.sum()))
        ranks = pl.Series(a[ok]).rank(method="ordinal").to_numpy()
        bucket = np.ceil(ranks / len(ranks) * QUINTILES)
        spreads.append(float(b[ok][bucket == QUINTILES].mean() - b[ok][bucket == 1].mean()))
    if not ics:
        return FactorResult(factor, label, 0, 0, None, None, None, None, None, None)
    ics.sort()
    values = np.array([v for _, v in ics])
    half = len(values) // 2
    return FactorResult(
        factor=factor,
        label=label,
        months=len(values),
        names=float(np.mean(names)),
        mean_ic=float(values.mean()),
        t=newey_west_t(values, max(1, horizon // 21)),
        positive=float((values > 0).mean()),
        first_half=float(values[:half].mean()) if half else None,
        second_half=float(values[half:].mean()) if half else None,
        spread=float(np.mean(spreads)) * 100,
    )


# --- outcomes at month-ends -------------------------------------------------------


def month_end_sessions(calendar: pl.Series) -> list[date]:
    """The last trading day of each month in `calendar` (the latest month
    only once it is over is the caller's concern)."""
    days = pl.DataFrame({DATE: calendar})
    return (
        days.with_columns(pl.col(DATE).dt.truncate("1mo").alias("_m"))
        .group_by("_m")
        .agg(pl.col(DATE).max())
        .sort(DATE)[DATE]
        .to_list()
    )


def _long(wide: pl.DataFrame, dates: list[date], name: str) -> pl.DataFrame:
    at = pl.DataFrame({DATE: dates}).join(wide, on=DATE, how="left")
    tickers = [c for c in at.columns if c != DATE]
    return at.unpivot(index=DATE, on=tickers, variable_name="ticker", value_name=name)


def outcomes(panel: PricePanel, dates: list[date]) -> pl.DataFrame:
    """(date, ticker, fwd_{h}, fwd_{h}_badj, beta_252, vol_60d, gated) at
    each date: excess over SPY from the next close to h sessions on, the
    beta-adjusted version, and the price features known on the date."""
    feats = panel_features(panel.close, panel.high, panel.volume, panel.spy)
    out = _long(feats["beta_252"], dates, "beta_252")
    for name in ("vol_60d", "px_vs_sma200", "sma50_vs_sma200", "rs_6mo", "dist_from_52w_high"):
        out = out.join(_long(feats[name], dates, name), on=[DATE, "ticker"], how="left")
    for h in HORIZONS:
        ret, spy_ret = forward_returns(panel, h)
        spy_at = pl.DataFrame({DATE: ret[DATE], "_spy": spy_ret})
        r = _long(ret, dates, "_ret").join(spy_at, on=DATE, how="left")
        out = (
            out.join(
                r.select(
                    DATE,
                    "ticker",
                    (pl.col("_ret") - pl.col("_spy")).alias(f"fwd_{h}"),
                    pl.col("_spy").alias(f"_spy_{h}"),
                ),
                on=[DATE, "ticker"],
                how="left",
            )
            .with_columns(
                (
                    pl.col(f"fwd_{h}")
                    + pl.col(f"_spy_{h}")
                    - pl.col("beta_252") * pl.col(f"_spy_{h}")
                ).alias(f"fwd_{h}_badj")
            )
            .drop(f"_spy_{h}")
        )
    floats = [c for c, t in out.schema.items() if t == pl.Float64]
    out = out.with_columns(pl.col(floats).fill_nan(None))
    return out.with_columns(
        (
            (pl.col("px_vs_sma200") > 0)
            & (pl.col("sma50_vs_sma200") > 0)
            & (pl.col("rs_6mo") > 0)
            & (pl.col("dist_from_52w_high") >= -0.30)
        ).alias("gated")
    )


# --- fundamentals -----------------------------------------------------------------

# Each read so that higher is expected to be better.
FUNDAMENTAL_FACTORS: tuple[str, ...] = (
    "screen_points",  # the screen's 0-45 fundamentals points
    "revenue_growth",
    "operating_margin",
    "fcf_yield",
    "low_debt",  # -debt_to_equity
    "roe",
    "gross_profitability",
    "sue",
)
EXCLUDED_SECTORS = ("Financial Services",)  # bank cash flow and margins mean little


def screen_measures(fund: pl.DataFrame) -> pl.DataFrame:
    """`fund` plus the screen's own points and hard-rule pass, computed
    with screen.py on the point-in-time figures."""
    from ..discover import screen

    points, passes = [], []
    for row in fund.iter_rows(named=True):
        f = {
            "revenue_growth_yoy": row["revenue_growth"],
            "fcf_yield": row["fcf_yield"],
            "operating_margin": row["operating_margin"],
            "debt_to_equity": row["debt_to_equity"],
        }
        if row["revenue_growth"] is None or row["operating_margin"] is None:
            points.append(None)
        else:
            points.append(screen._score_fundamentals(f)[0])
        de = row["debt_to_equity"]
        passes.append(
            row["market_cap"] is not None
            and row["market_cap"] >= screen.MIN_MARKET_CAP
            and row["revenue_growth"] is not None
            and row["revenue_growth"] >= screen.MIN_REVENUE_GROWTH
            and (row["operating_cash_flow"] or 0) > 0
            and (row["free_cash_flow"] or 0) > 0
            and row["roe"] is not None
            and row["roe"] >= screen.MIN_RETURN_ON_EQUITY
            and (de is None or de <= screen.MAX_DEBT_TO_EQUITY)
        )
    return fund.with_columns(
        pl.Series("screen_points", points, dtype=pl.Float64),
        pl.Series("passes_rules", passes, dtype=pl.Boolean),
        (-pl.col("debt_to_equity")).alias("low_debt"),
    )


def _winsor(frame: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Clip each measure to its month's 1st-99th percentile: a ratio over a
    near-zero denominator should not dominate a quintile mean."""
    return frame.with_columns(
        [
            pl.col(c).clip(pl.col(c).quantile(0.01).over(DATE), pl.col(c).quantile(0.99).over(DATE))
            for c in columns
        ]
    )


def fundamentals_study(
    fund: pl.DataFrame, labels: pl.DataFrame, sectors: dict[str, str]
) -> dict[str, Any]:
    """{"results": [FactorResult], "rules": {...}, "coverage": {...}}."""
    keep = fund.filter(
        ~pl.col("ticker").replace_strict(sectors, default="").is_in(EXCLUDED_SECTORS)
    )
    frame = screen_measures(keep).join(labels, on=[DATE, "ticker"], how="inner")
    factors = list(FUNDAMENTAL_FACTORS)
    frame = _winsor(frame, factors)
    large = frame.filter(pl.col("market_cap") >= LARGE_THEN_CAP)
    results = []
    for f in factors:
        for h in HORIZONS:
            for label in (f"fwd_{h}", f"fwd_{h}_badj"):
                r = evaluate(frame, f, label, h)
                r.large = evaluate(large, f, label, h)
                results.append(r)
    rules = {}
    for h in HORIZONS:
        monthly = (
            frame.drop_nulls(f"fwd_{h}")
            .group_by(DATE, "passes_rules")
            .agg(pl.col(f"fwd_{h}").mean(), pl.len().alias("n"))
            .pivot(on="passes_rules", index=DATE, values=[f"fwd_{h}", "n"])
            .drop_nulls()
            .sort(DATE)
        )
        if monthly.is_empty():
            continue
        diff = (monthly[f"fwd_{h}_true"] - monthly[f"fwd_{h}_false"]).to_numpy()
        rules[h] = {
            "months": len(diff),
            "passed_avg": float(monthly["n_true"].to_numpy().mean()),
            "failed_avg": float(monthly["n_false"].to_numpy().mean()),
            "passed_pct": float(monthly[f"fwd_{h}_true"].to_numpy().mean()) * 100,
            "failed_pct": float(monthly[f"fwd_{h}_false"].to_numpy().mean()) * 100,
            "t": newey_west_t(diff, h // 21),
        }
    coverage = {
        f: float(frame[f].is_not_null().to_numpy().mean())
        for f in factors
        if f in frame.columns and frame.height
    }
    return {
        "results": results,
        "rules": rules,
        "coverage": coverage,
        "start": frame[DATE].min(),
        "end": frame[DATE].max(),
        "tickers": frame["ticker"].n_unique(),
    }


# --- risk ---------------------------------------------------------------------------


def realized_vol(panel: PricePanel, horizon: int = 126) -> pl.DataFrame:
    """Wide annualized volatility of daily returns over the `horizon`
    sessions after each date."""
    tickers = panel.tickers
    rets = panel.close.select(DATE, *(pl.col(t).pct_change().alias(t) for t in tickers))
    return rets.select(
        DATE,
        *(
            (pl.col(t).rolling_std(horizon).shift(-horizon) * math.sqrt(252)).alias(t)
            for t in tickers
        ),
    )


def next_month_returns(panel: PricePanel, dates: list[date]) -> pl.DataFrame:
    """(date, ticker, ret, spy): return from the session after each date to
    the session after the next date — one non-overlapping holding month."""
    cal = panel.close[DATE].to_list()
    pos = {d: i for i, d in enumerate(cal)}
    rows = []
    close = panel.close
    spy = panel.close.select(DATE).join(panel.spy, on=DATE, how="left")["SPY"].to_list()
    tickers = panel.tickers
    for a, b in zip(dates, dates[1:], strict=False):
        i, j = pos[a] + 1, pos[b] + 1
        if j >= len(cal):
            break
        start, stop = close.row(i), close.row(j)
        s0, s1 = spy[i], spy[j]
        for k, t in enumerate(tickers, 1):
            p0, p1 = start[k], stop[k]
            if p0 and p1 and p0 > 0 and math.isfinite(p0) and math.isfinite(p1):
                rows.append((a, t, p1 / p0 - 1, s1 / s0 - 1 if s0 and s1 else None))
    return pl.DataFrame(
        rows,
        schema={DATE: pl.Date, "ticker": pl.String, "ret": pl.Float64, "spy": pl.Float64},
        orient="row",
    )


def _portfolio_stats(monthly: np.ndarray) -> dict[str, float]:
    monthly = monthly[np.isfinite(monthly)]
    growth = np.cumprod(1 + monthly)
    years = len(monthly) / 12
    peak = np.maximum.accumulate(growth)
    vol = float(monthly.std() * math.sqrt(12))
    cagr = float(growth[-1] ** (1 / years) - 1) if years else float("nan")
    return {
        "cagr_pct": cagr * 100,
        "vol_pct": vol * 100,
        "return_per_risk": cagr / vol if vol else float("nan"),
        "max_drawdown_pct": float((growth / peak - 1).min()) * 100,
    }


def risk_study(panel: PricePanel, dates: list[date]) -> dict[str, Any]:
    """{"persistence", "quintiles", "portfolios", "start", "end"}."""
    labels = outcomes(panel, dates)
    fwd_vol = _long(realized_vol(panel), dates, "fwd_vol")
    frame = labels.join(fwd_vol, on=[DATE, "ticker"], how="left").with_columns(
        pl.col("fwd_vol").fill_nan(None)
    )
    pers = [
        ic
        for (_, g) in frame.group_by(DATE, maintain_order=True)
        if (
            ic := _rank_ic(
                g["vol_60d"].fill_null(np.nan).to_numpy(), g["fwd_vol"].fill_null(np.nan).to_numpy()
            )
        )
        is not None
    ]
    quint = (
        frame.drop_nulls(["vol_60d", "fwd_126", "fwd_vol"])
        .with_columns(
            (pl.col("vol_60d").rank("ordinal").over(DATE) / pl.len().over(DATE) * QUINTILES)
            .ceil()
            .cast(pl.Int8)
            .alias("q")
        )
        .group_by(DATE, "q")
        .agg(
            pl.col("vol_60d").mean(),
            pl.col("fwd_vol").mean(),
            pl.col("fwd_126").mean(),
            pl.col("fwd_126_badj").mean(),
            pl.col("fwd_252").mean(),
        )
        .group_by("q")
        .agg(pl.all().exclude(DATE).mean(), pl.len().alias("months"))
        .sort("q")
    )
    month = next_month_returns(panel, dates).join(
        frame.select(DATE, "ticker", "vol_60d", "gated"), on=[DATE, "ticker"], how="inner"
    )
    month = month.drop_nulls(["vol_60d", "ret"]).filter(pl.col("vol_60d") > 0)
    spy = month.group_by(DATE).agg(pl.col("spy").first()).sort(DATE)["spy"].to_numpy()

    def series(sub: pl.DataFrame, weighted: bool) -> np.ndarray:
        w = (1 / pl.col("vol_60d")) if weighted else pl.lit(1.0)
        return (
            sub.with_columns(w.alias("w"))
            .group_by(DATE)
            .agg(((pl.col("ret") * pl.col("w")).sum() / pl.col("w").sum()).alias("r"))
            .sort(DATE)["r"]
            .to_numpy()
        )

    calm = month.filter(pl.col("vol_60d") <= pl.col("vol_60d").median().over(DATE))
    gated = month.filter(pl.col("gated"))
    portfolios = {
        "SPY": _portfolio_stats(spy),
        "All, equal weight": _portfolio_stats(series(month, False)),
        "All, inverse-volatility weight": _portfolio_stats(series(month, True)),
        "Calmer half, equal weight": _portfolio_stats(series(calm, False)),
        "Trend-gated, equal weight": _portfolio_stats(series(gated, False)),
        "Trend-gated, inverse-volatility weight": _portfolio_stats(series(gated, True)),
    }
    return {
        "persistence": float(np.mean(pers)) if pers else None,
        "persistence_months": len(pers),
        "quintiles": quint.to_dicts(),
        "portfolios": portfolios,
        "start": month[DATE].min(),
        "end": month[DATE].max(),
    }


# --- report -------------------------------------------------------------------------


def _f(x: float | None, fmt: str) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not math.isfinite(x)) else format(x, fmt)


def format_fundamentals(study: dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        f"FUNDAMENTALS (SEC point-in-time), {study['start']} to {study['end']}, "
        f"{study['tickers']} S&P 500 names (financials excluded)",
        "=" * 78,
        "IC = per-month rank correlation with the excess return over SPY that followed;",
        "t = Newey-West; 'works' needs |t| >= 2 and the same sign in both halves;",
        f"'large then' = the same among companies already ${LARGE_THEN_CAP / 1e9:.0f}B+ on the "
        "date, where a verdict must hold too (else 'survivorship?').",
        "",
        f"{'measure':<20}{'label':<14}{'IC':>7}{'t':>7}{'IC>0':>6}{'1st/2nd half':>16}"
        f"{'Q5-Q1':>9}{'names':>7}{'large then':>15}  verdict",
    ]
    for r in study["results"]:
        big = r.large
        large_then = f"{_f(big.mean_ic, '+.3f')} t{_f(big.t, '+.1f')}" if big else "n/a"
        lines.append(
            f"{r.factor:<20}{r.label:<14}{_f(r.mean_ic, '+.3f'):>7}{_f(r.t, '+.1f'):>7}"
            f"{_f(r.positive, '.0%'):>6}"
            f"{_f(r.first_half, '+.3f') + ' / ' + _f(r.second_half, '+.3f'):>16}"
            f"{_f(r.spread, '+.1f') + '%':>9}{r.names:>7.0f}{large_then:>15}  {r.verdict}"
        )
    lines += ["", "Screen hard rules (passed minus failed, mean excess return over SPY):"]
    for h, r in study["rules"].items():
        lines.append(
            f"  {h} sessions: passed {r['passed_pct']:+.1f}% (~{r['passed_avg']:.0f} names) vs "
            f"failed {r['failed_pct']:+.1f}% (~{r['failed_avg']:.0f}); "
            f"difference t {_f(r['t'], '+.1f')} over {r['months']} months"
        )
    lines += ["", "Coverage (share of name-months with the measure):"]
    lines.append("  " + ", ".join(f"{k} {v:.0%}" for k, v in study["coverage"].items()))
    return "\n".join(lines)


def format_risk(study: dict[str, Any]) -> str:
    lines = [
        "=" * 78,
        f"RISK, {study['start']} to {study['end']}, S&P 500 names",
        "=" * 78,
        f"Is volatility predictable? Rank correlation of the last 60 days' volatility with the "
        f"next 126 days': {_f(study['persistence'], '+.2f')} "
        f"(over {study['persistence_months']} months).",
        "",
        "Quintiles by the last 60 days' volatility (1 = calmest):",
        f"  {'q':<3}{'vol now':>9}{'vol next':>10}{'126d vs SPY':>13}{'beta-adj':>10}{'252d vs SPY':>13}",
    ]
    for q in study["quintiles"]:
        lines.append(
            f"  {q['q']:<3}{q['vol_60d'] * 100:>8.0f}%{q['fwd_vol'] * 100:>9.0f}%"
            f"{q['fwd_126'] * 100:>+12.1f}%{q['fwd_126_badj'] * 100:>+9.1f}%"
            f"{q['fwd_252'] * 100:>+12.1f}%"
        )
    lines += [
        "",
        "Monthly-rebalanced portfolios:",
        f"  {'':<40}{'CAGR':>8}{'vol':>8}{'CAGR/vol':>10}{'worst drop':>12}",
    ]
    for name, p in study["portfolios"].items():
        lines.append(
            f"  {name:<40}{p['cagr_pct']:>7.1f}%{p['vol_pct']:>7.1f}%"
            f"{_f(p['return_per_risk'], '.2f'):>10}{p['max_drawdown_pct']:>11.1f}%"
        )
    lines.append(
        "  (today's S&P 500 members, so every stock basket is flattered by survivorship; "
        "compare the baskets with each other more than with SPY)"
    )
    return "\n".join(lines)


__all__ = [
    "FactorResult",
    "evaluate",
    "format_fundamentals",
    "format_risk",
    "fundamentals_study",
    "month_end_sessions",
    "newey_west_t",
    "outcomes",
    "risk_study",
    "screen_measures",
]
