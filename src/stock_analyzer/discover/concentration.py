"""How much of the portfolio is really one bet, and how big each stock may be.

Oct 21 - Nov 20 2026 the portfolio fell 26% while SPY fell 11%. The
holdings were eleven stocks in six "sectors", but BE (industrials), NVDA,
AMD, MRVL, ARM, TSM (semis), AVGO, ANET (tech), POWL (industrials) and
TSLA (consumer) moved together — 6-month daily correlations of 0.5-0.8 —
and were ~83% of the money. The sector caps could not see that, and the
single-stock cap treated BE (114% annual volatility) like GOOGL (38%).

Two deterministic measures, from daily closes only:

  1. correlated groups: holdings clustered by average-linkage on their
     daily-return correlation (RISK_GROUP_MIN_CORR, default 0.5). A
     group's combined weight is held to RISK_GROUP_MAX_PCT (default 50%);
  2. a volatility-scaled position cap: a stock more volatile than
     RISK_VOL_REF (annualized, default 45%) gets the single-stock cap
     scaled down by REF/vol — BE at 114% gets ~14% of a 35% cap.

Used by the weekly email (a decision line and a table) and the rebalance
(buys into an over-cap group or past a stock's own cap are cut in code;
the prompt is told to trim toward the caps). It sizes; it says nothing
about whether the companies are good. Market timing was tested and
rejected: selling on the IBD-style "correction" status, 2005-2026, cost
QQQ 5.4 points a year for ~9 points less drawdown.

Broad index funds (the core fund and the usual S&P/total-market ETFs)
and cash-like holdings are never grouped: they are the diversification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any

CORRELATION_DAYS = 126  # ~6 months of daily returns
VOL_DAYS = 252  # one year
MIN_OVERLAP = 60  # fewer common days than this: no correlation read
MIN_CAP_PCT = 5.0  # the vol-scaled cap never goes below this
CASH_LIKE_VOL = 0.02  # annualized; a money-market fund is ~0
BROAD_FUNDS = frozenset({"VOO", "SPY", "IVV", "VTI", "ITOT", "SCHB", "SPLG", "FXAIX", "SWTSX"})

Series = dict[str, list[tuple[date, float]]]


@dataclass
class Group:
    tickers: list[str]
    value: float
    pct: float
    avg_corr: float
    worst_drop_pct: float | None  # the basket's worst peak-to-trough, past year
    over: bool = False
    trim_usd: float = 0.0  # down to the cap

    @property
    def loss_at_worst_usd(self) -> float | None:
        if self.worst_drop_pct is None:
            return None
        return self.value * self.worst_drop_pct / 100


@dataclass
class VolCap:
    ticker: str
    vol_pct: float  # annualized volatility, percent
    cap_pct: float
    pct: float  # share of the portfolio now
    value: float

    @property
    def over(self) -> bool:
        return self.pct > self.cap_pct + 0.5

    @property
    def trim_usd(self) -> float:
        return max(0.0, self.value - self.value * self.cap_pct / self.pct) if self.pct else 0.0


@dataclass
class Concentration:
    total: float
    max_group_pct: float
    groups: list[Group] = field(default_factory=list)
    # False (the default, the investor's call 2026-11-26): an over-cap group
    # blocks new buys and steers new money elsewhere, but no holding is sold
    # for concentration alone — sales stay for broken theses. The volatility
    # cap still asks for trims.
    group_trims: bool = False
    vol_caps: list[VolCap] = field(default_factory=list)

    @property
    def over_groups(self) -> list[Group]:
        return [g for g in self.groups if g.over]

    @property
    def over_vol(self) -> list[VolCap]:
        return [v for v in self.vol_caps if v.over]

    def group_of(self, ticker: str) -> Group | None:
        t = ticker.upper()
        return next((g for g in self.groups if t in g.tickers), None)

    def cap_for(self, ticker: str) -> float | None:
        t = ticker.upper()
        return next((v.cap_pct for v in self.vol_caps if v.ticker == t), None)


# --- arithmetic -------------------------------------------------------------------


def daily_returns(bars: list[tuple[date, float]], days: int) -> dict[date, float]:
    """{day: return} for the last `days` returns."""
    out: dict[date, float] = {}
    for (_, a), (d, b) in zip(bars, bars[1:], strict=False):
        if a and b:
            out[d] = b / a - 1
    keep = sorted(out)[-days:]
    return {d: out[d] for d in keep}


def correlation(a: dict[date, float], b: dict[date, float]) -> float | None:
    common = sorted(a.keys() & b.keys())
    if len(common) < MIN_OVERLAP:
        return None
    x = [a[d] for d in common]
    y = [b[d] for d in common]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    sxy = sum((i - mx) * (j - my) for i, j in zip(x, y, strict=True))
    sxx = sum((i - mx) ** 2 for i in x)
    syy = sum((j - my) ** 2 for j in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / math.sqrt(sxx * syy)


def annual_vol(rets: dict[date, float]) -> float | None:
    vals = list(rets.values())
    if len(vals) < MIN_OVERLAP:
        return None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals)) * math.sqrt(252)


def correlated_groups(
    rets: dict[str, dict[date, float]], *, min_corr: float
) -> list[tuple[list[str], float]]:
    """Average-linkage clusters whose members correlate at `min_corr` or
    more on average: [(tickers, average pairwise correlation)], every
    ticker in exactly one cluster. An unreadable pair counts as 0."""
    names = sorted(rets)
    corr: dict[tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            c = correlation(rets[a], rets[b])
            corr[(a, b)] = corr[(b, a)] = c if c is not None else 0.0
    clusters: list[list[str]] = [[n] for n in names]

    def link(x: list[str], y: list[str]) -> float:
        return sum(corr[(a, b)] for a in x for b in y) / (len(x) * len(y))

    while len(clusters) > 1:
        best, pair = -2.0, (0, 0)
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                v = link(clusters[i], clusters[j])
                if v > best:
                    best, pair = v, (i, j)
        if best < min_corr:
            break
        i, j = pair
        clusters[i] = sorted(clusters[i] + clusters[j])
        del clusters[j]

    def inner(c: list[str]) -> float:
        pairs = [(a, b) for k, a in enumerate(c) for b in c[k + 1 :]]
        return sum(corr[p] for p in pairs) / len(pairs) if pairs else 1.0

    return [(c, inner(c)) for c in clusters]


def basket_worst_drop(
    rets: dict[str, dict[date, float]], weights: dict[str, float]
) -> float | None:
    """Worst peak-to-trough of the members held at today's weights, percent."""
    members = [t for t in weights if t in rets and weights[t] > 0]
    if not members:
        return None
    days = sorted(set.intersection(*(set(rets[t]) for t in members)))
    if len(days) < MIN_OVERLAP:
        return None
    w = sum(weights[t] for t in members)
    level = peak = 1.0
    worst = 0.0
    for d in days:
        level *= 1 + sum(weights[t] * rets[t][d] for t in members) / w
        peak = max(peak, level)
        worst = max(worst, 1 - level / peak)
    return round(worst * 100, 1)


def vol_cap_pct(vol: float, *, max_pct: float, ref_vol: float) -> float:
    """The single-stock cap, scaled down for a stock swinging more than
    `ref_vol` a year."""
    if ref_vol <= 0 or vol <= ref_vol:
        return max_pct
    return max(MIN_CAP_PCT, round(max_pct * ref_vol / vol, 1))


# --- the assessment ---------------------------------------------------------------


def assess(
    values: dict[str, float],
    series: Series,
    *,
    total: float | None = None,
    max_group_pct: float = 50.0,
    min_corr: float = 0.5,
    max_position_pct: float = 35.0,
    ref_vol: float = 0.45,
    exclude: set[str] | frozenset[str] = frozenset(),
    group_trims: bool = False,
) -> Concentration:
    """Groups and volatility caps for `values` ({ticker: market value}).
    `total` is the whole portfolio (positions + cash); defaults to the sum
    of `values`. A holding without enough price history is left out of
    both, never guessed."""
    values = {t.upper(): float(v) for t, v in values.items() if v and v > 0}
    total = float(total) if total else sum(values.values())
    out = Concentration(total=total, max_group_pct=max_group_pct, group_trims=group_trims)
    if total <= 0:
        return out
    skip = {t.upper() for t in exclude} | BROAD_FUNDS
    corr_rets: dict[str, dict[date, float]] = {}
    year_rets: dict[str, dict[date, float]] = {}
    for t in values:
        bars = series.get(t)
        if t in skip or not bars:
            continue
        year = daily_returns(bars, VOL_DAYS)
        vol = annual_vol(year)
        if vol is None or vol < CASH_LIKE_VOL:
            continue
        year_rets[t] = year
        corr_rets[t] = daily_returns(bars, CORRELATION_DAYS)
        pct = values[t] / total * 100
        out.vol_caps.append(
            VolCap(
                ticker=t,
                vol_pct=round(vol * 100, 1),
                cap_pct=vol_cap_pct(vol, max_pct=max_position_pct, ref_vol=ref_vol),
                pct=round(pct, 1),
                value=values[t],
            )
        )
    out.vol_caps.sort(key=lambda v: -v.value)
    for tickers, avg in correlated_groups(corr_rets, min_corr=min_corr):
        if len(tickers) < 2:
            continue
        value = sum(values[t] for t in tickers)
        pct = value / total * 100
        g = Group(
            tickers=sorted(tickers, key=lambda t: -values[t]),
            value=value,
            pct=round(pct, 1),
            avg_corr=round(avg, 2),
            worst_drop_pct=basket_worst_drop(year_rets, {t: values[t] for t in tickers}),
        )
        if 0 < max_group_pct < 100 and pct > max_group_pct:
            g.over = True
            g.trim_usd = value - total * max_group_pct / 100
        out.groups.append(g)
    out.groups.sort(key=lambda g: -g.value)
    return out


def as_dict(c: Concentration) -> dict[str, Any]:
    """Plain data for the email and the rebalance state."""
    return {
        "total": c.total,
        "max_group_pct": c.max_group_pct,
        "group_trims": c.group_trims,
        "groups": [
            {
                "tickers": g.tickers,
                "value": g.value,
                "pct": g.pct,
                "avg_corr": g.avg_corr,
                "worst_drop_pct": g.worst_drop_pct,
                "loss_at_worst_usd": g.loss_at_worst_usd,
                "over": g.over,
                "trim_usd": g.trim_usd,
            }
            for g in c.groups
        ],
        "vol_caps": [
            {
                "ticker": v.ticker,
                "vol_pct": v.vol_pct,
                "cap_pct": v.cap_pct,
                "pct": v.pct,
                "value": v.value,
                "over": v.over,
                "trim_usd": v.trim_usd,
            }
            for v in c.vol_caps
        ],
    }


def _names(tickers: list[str], n: int = 6) -> str:
    return ", ".join(tickers[:n]) + (f" +{len(tickers) - n} more" if len(tickers) > n else "")


def rebalance_block(c: Concentration, *, step_pct: float, core_fund: str) -> str:
    """The rebalancer's prompt block ("" when nothing is over a cap)."""
    if not c.over_groups and not c.over_vol:
        return ""
    step = c.total * step_pct / 100
    lines = ["CONCENTRATION LIMITS (deterministic, from daily closes; enforced on buys in code):"]
    for g in c.over_groups:
        lines.append(
            f"- {', '.join(g.tickers)} move together (average 6-month daily correlation "
            f"{g.avg_corr:.2f}) and are {g.pct:.0f}% of the portfolio (${g.value:,.0f}) against "
            f"a {c.max_group_pct:g}% cap for any group of correlated holdings"
            + (
                f"; together they fell up to {g.worst_drop_pct:.0f}% in the past year"
                if g.worst_drop_pct
                else ""
            )
            + f". Over the cap by ${g.trim_usd:,.0f}. No BUY or ADD of these names."
            + (
                ""
                if c.group_trims
                else " Do NOT TRIM or SELL them for concentration alone (the investor sells only "
                "on a broken thesis): bring the share down by putting idle cash and new money "
                "into the core fund and holdings outside the group."
            )
        )
    for v in c.over_vol:
        lines.append(
            f"- {v.ticker} swings {v.vol_pct:.0f}% a year, so its cap is {v.cap_pct:g}% "
            f"(not the plain single-stock cap); it is {v.pct:.0f}% (${v.value:,.0f}), "
            f"${v.trim_usd:,.0f} over."
        )
    if not c.group_trims and not c.over_vol:
        return "\n".join(lines)
    lines.append(
        f"TRIM toward {'these caps' if c.group_trims else 'the volatility caps above'} "
        f"this rebalance, up to ~${step:,.0f} in total "
        f"({step_pct:g}% of the portfolio; the rest at later rebalances). Sell in tax-advantaged "
        "accounts (IRA, 401(k), HSA) first, where a sale owes no tax; the most over-cap and "
        "most volatile names first; respect shares promised to written calls. Proceeds go to "
        f"{core_fund} (the core fund) or to holdings outside the group — never back into it. "
        "This is position sizing, not a verdict on the companies: say so in the rationale."
    )
    return "\n".join(lines)


def load_and_assess(
    values: dict[str, float],
    *,
    total: float | None,
    settings: Any,
    today: date | None = None,
) -> Concentration | None:
    """`assess` with a year of closes from the bar store and the limits
    from settings. None when the closes can't be read: a missing check is
    named by the caller, never a reason to stop a run."""
    from datetime import timedelta

    from ..data import yf_gateway
    from ..logging import get_logger

    log = get_logger(__name__)
    today = today or date.today()
    tickers = [t for t, v in values.items() if v and v > 0]
    if not tickers:
        return None
    try:
        closes = yf_gateway.daily_closes(tickers, today - timedelta(days=400), what="concentration")
    except Exception as e:  # noqa: BLE001
        log.warning("Concentration check: closes unavailable (%s)", e)
        return None
    series = {
        t.upper(): list(zip(c["date"].to_list(), c["Close"].to_list(), strict=True))
        for t, c in closes.items()
        if c.height
    }
    result = assess(
        values,
        series,
        total=total,
        max_group_pct=settings.risk_group_max_pct,
        min_corr=settings.risk_group_min_corr,
        max_position_pct=settings.rebalance_max_position_pct,
        ref_vol=settings.risk_vol_ref,
        exclude={settings.core_fund},
        group_trims=settings.risk_group_trims,
    )
    for g in result.over_groups:
        log.info(
            "Concentration: %s move together, %.0f%% of the portfolio (cap %g%%)",
            ", ".join(g.tickers),
            g.pct,
            result.max_group_pct,
        )
    return result
