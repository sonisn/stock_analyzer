"""How likely the portfolio is to reach its goal, and what would change it.

A Monte Carlo over monthly returns. Each simulated future is stitched from
12-month blocks of the holdings' own history (current weights, monthly
closes from the bar store), so it keeps their real volatility, their
correlation with each other and their runs of bad years — a concentrated
portfolio gets the wide range it actually has.

What it does not keep is their average. These are stocks held partly
because they went up, and fifteen years of that is not a forecast, so
every month's return is shifted to centre on an assumed annual return
(GOAL_EXPECTED_RETURN, 7% by default). The same money in SPY is run with
the same assumption and the same random draws: the gap between the two
is what concentration costs or buys in odds, not a guess about which
stocks will do better.

End value is linear in the monthly contribution on any one path
(V = a + c * b), which makes "what contribution gives 75% odds" an exact
quantile rather than a search.

Numbers are nominal dollars. No LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

N_PATHS = 10_000
BLOCK_MONTHS = 12
# Fixed so the same inputs give the same email twice.
SEED = 20260926
TARGET_ODDS = 0.75
DEEP_DRAWDOWN = 0.30
# At least this many months of history to resample from.
MIN_HISTORY_MONTHS = 36


@dataclass
class Projection:
    months: int
    start_value: float
    monthly_contribution: float
    expected_return: float
    p10: float
    p50: float
    p90: float
    # Odds of a fall of DEEP_DRAWDOWN or more from a peak along the way.
    deep_drawdown_odds: float
    target: float | None = None
    odds: float | None = None
    # Monthly contribution that would make the odds TARGET_ODDS.
    needed_contribution: float | None = None
    # The same money and contributions, in SPY's swings.
    spy_odds: float | None = None
    spy_p10: float | None = None
    spy_p50: float | None = None
    annual_volatility: float = 0.0
    spy_annual_volatility: float = 0.0
    history_months: int = 0
    # Holdings whose history is shorter than the window; SPY's months fill in.
    filled_from_spy: tuple[str, ...] = ()


def is_cash_like(frame: Any) -> bool:
    """A money-market fund: every close at its stable $1.00 NAV. Filling
    its missing history with SPY's months would model cash as stocks."""
    if frame is None or getattr(frame, "empty", True) or "Close" not in frame:
        return False
    closes = frame["Close"].dropna()
    return bool(len(closes)) and bool(((closes - 1.0).abs() <= 0.005).all())


def monthly_returns(bars: dict[str, Any]) -> pd.DataFrame:
    """(month x ticker) returns from daily adjusted bars."""
    closes = {}
    for ticker, frame in bars.items():
        if frame is None or getattr(frame, "empty", True) or "Close" not in frame:
            continue
        series = frame["Close"].dropna()
        idx = pd.DatetimeIndex(series.index)
        series.index = idx.tz_localize(None) if idx.tz is not None else idx
        closes[ticker] = series.resample("ME").last()
    if not closes:
        return pd.DataFrame()
    frame = pd.DataFrame(closes)
    # The current month is incomplete; a partial month is not a month.
    frame = frame[frame.index < pd.Timestamp.today().normalize().replace(day=1)]
    return frame.pct_change().iloc[1:]


def portfolio_returns(
    weights: dict[str, float], returns: pd.DataFrame, *, benchmark: str = "SPY"
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Monthly returns of a portfolio held at `weights` (rebalanced monthly).

    A holding without history for a month gets the benchmark's return that
    month — a young stock is assumed to move with the market until it has
    a record of its own."""
    spy = returns[benchmark]
    months = spy.dropna().index
    total = sum(w for w in weights.values() if w > 0)
    if total <= 0:
        return np.array([]), ()
    port = pd.Series(0.0, index=months)
    filled: list[str] = []
    for ticker, weight in weights.items():
        if weight <= 0:
            continue
        own = (
            returns[ticker].reindex(months)
            if ticker in returns
            else pd.Series(np.nan, index=months)
        )
        if own.isna().any():
            filled.append(ticker)
        port += (weight / total) * own.fillna(spy.reindex(months))
    return port.to_numpy(dtype=float), tuple(sorted(filled))


def _block_paths(n_history: int, months: int, rng: np.random.Generator) -> np.ndarray:
    """(N_PATHS x months) indexes into the history, in 12-month blocks."""
    block = min(BLOCK_MONTHS, n_history)
    n_blocks = -(-months // block)
    starts = rng.integers(0, n_history - block + 1, size=(N_PATHS, n_blocks))
    idx = starts[:, :, None] + np.arange(block)[None, None, :]
    return idx.reshape(N_PATHS, -1)[:, :months]


def _centred(history: np.ndarray, annual_return: float) -> np.ndarray:
    monthly = (1 + annual_return) ** (1 / 12) - 1
    return history - history.mean() + monthly


def _simulate(
    returns: np.ndarray, idx: np.ndarray, start_value: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(a, b, wealth-without-contributions) per path: end value is a + c*b
    for a monthly contribution c, paid at the start of each month."""
    growth = 1.0 + returns[idx]  # N x T
    # growth from the start of month k to the end: product of growth[k:]
    tail = np.cumprod(growth[:, ::-1], axis=1)[:, ::-1]
    a = start_value * tail[:, 0]
    b = tail.sum(axis=1)
    return a, b, growth


def _deep_drawdown_odds(growth: np.ndarray, start_value: float, contribution: float) -> float:
    wealth = np.empty_like(growth)
    level = np.full(growth.shape[0], start_value, dtype=float)
    for t in range(growth.shape[1]):
        level = (level + contribution) * growth[:, t]
        wealth[:, t] = level
    peak = np.maximum.accumulate(
        np.concatenate([np.full((len(level), 1), start_value), wealth], 1), 1
    )
    drawdown = 1 - wealth / peak[:, 1:]
    return float((drawdown.max(axis=1) >= DEEP_DRAWDOWN).mean())


def project(
    *,
    weights: dict[str, float],
    returns: pd.DataFrame,
    start_value: float,
    months: int,
    monthly_contribution: float,
    expected_return: float,
    target: float | None = None,
    benchmark: str = "SPY",
) -> Projection | None:
    """None when there is too little history to resample from."""
    if benchmark not in returns or months <= 0 or start_value <= 0:
        return None
    port, filled = portfolio_returns(weights, returns, benchmark=benchmark)
    spy = returns[benchmark].dropna().to_numpy(dtype=float)
    if len(port) < MIN_HISTORY_MONTHS:
        return None
    rng = np.random.default_rng(SEED)
    idx = _block_paths(len(port), months, rng)
    a, b, growth = _simulate(_centred(port, expected_return), idx, start_value)
    end = a + monthly_contribution * b
    out = Projection(
        months=months,
        start_value=start_value,
        monthly_contribution=monthly_contribution,
        expected_return=expected_return,
        p10=float(np.percentile(end, 10)),
        p50=float(np.percentile(end, 50)),
        p90=float(np.percentile(end, 90)),
        deep_drawdown_odds=_deep_drawdown_odds(growth, start_value, monthly_contribution),
        annual_volatility=float(port.std(ddof=1) * np.sqrt(12)),
        spy_annual_volatility=float(spy.std(ddof=1) * np.sqrt(12)),
        history_months=len(port),
        filled_from_spy=filled,
    )
    # SPY on the same draws (same months of history, same order).
    spy_series = returns[benchmark].reindex(returns[benchmark].dropna().index).to_numpy(dtype=float)
    sa, sb, _ = _simulate(_centred(spy_series, expected_return), idx, start_value)
    spy_end = sa + monthly_contribution * sb
    out.spy_p10 = float(np.percentile(spy_end, 10))
    out.spy_p50 = float(np.percentile(spy_end, 50))
    if target:
        out.target = target
        out.odds = float((end >= target).mean())
        out.spy_odds = float((spy_end >= target).mean())
        # Per path, the contribution that just reaches the target; the
        # TARGET_ODDS quantile of those reaches it on that share of paths.
        needed = (target - a) / b
        out.needed_contribution = max(float(np.quantile(needed, TARGET_ODDS)), 0.0)
    return out
