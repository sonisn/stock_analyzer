"""Daily charts drawn here, from the bars the package already keeps.

The fallback for chart-img: when its key is refused (every request 403'd
from 2026-10-14 on and the weekly email went out with no charts) the
email still gets one chart per stock. Same look as the chart-img ones —
dark, one year of daily candles, 50/200-day averages, volume below —
drawn from `yf_gateway.daily_bars`, so it costs no extra request when the
run has already loaded the history.
"""

from __future__ import annotations

import io
from datetime import date, timedelta

from ..logging import get_logger

logger = get_logger(__name__)

_SHOWN_DAYS = 365
# 200 trading days of warm-up so the 200-day average starts at the left edge.
_FETCH_DAYS = _SHOWN_DAYS + 300

_BG = "#131722"
_GRID = "#2a2e39"
_TEXT = "#d1d4dc"
_UP = "#26a69a"
_DOWN = "#ef5350"
_SMA50 = "#2962ff"
_SMA200 = "#ff9800"


def render_chart(ticker: str, *, today: date | None = None) -> bytes | None:
    """A PNG chart of `ticker`, or None when there are no bars or drawing
    fails. Never raises."""
    from . import yf_gateway

    today = today or date.today()
    try:
        bars = yf_gateway.daily_bars(
            ticker, start=today - timedelta(days=_FETCH_DAYS), what="local_chart"
        )
    except Exception as e:  # noqa: BLE001 — a missing chart must not stop the email
        logger.warning("No bars for %s chart: %s", ticker, e)
        return None
    if bars is None or bars.height < 20:
        return None
    try:
        return _draw(ticker, bars, shown_from=today - timedelta(days=_SHOWN_DAYS))
    except Exception as e:  # noqa: BLE001
        logger.warning("Drawing the %s chart failed: %s", ticker, e)
        return None


def _draw(ticker: str, bars, *, shown_from: date) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import polars as pl

    from .frames import DATE

    frame = bars.with_columns(
        pl.col("Close").rolling_mean(50).alias("sma50"),
        pl.col("Close").rolling_mean(200).alias("sma200"),
    ).filter(pl.col(DATE) >= shown_from)

    days = frame[DATE].to_list()
    opens = frame["Open"].to_list()
    highs = frame["High"].to_list()
    lows = frame["Low"].to_list()
    closes = frame["Close"].to_list()
    volumes = frame["Volume"].to_list() if "Volume" in frame.columns else [0.0] * len(days)
    x = list(range(len(days)))
    colors = [_UP if c >= o else _DOWN for o, c in zip(opens, closes, strict=True)]

    fig, (ax, vax) = plt.subplots(
        2,
        1,
        figsize=(8, 4.5),
        dpi=100,
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.04},
    )
    fig.patch.set_facecolor(_BG)
    for a in (ax, vax):
        a.set_facecolor(_BG)
        a.grid(color=_GRID, linewidth=0.6)
        a.tick_params(colors=_TEXT, labelsize=8)
        for spine in a.spines.values():
            spine.set_color(_GRID)
        a.yaxis.tick_right()

    ax.vlines(x, lows, highs, colors=colors, linewidth=0.7)
    bodies = [
        abs(c - o) or (h - lo) * 0.01
        for o, c, h, lo in zip(opens, closes, highs, lows, strict=True)
    ]
    bottoms = [min(o, c) for o, c in zip(opens, closes, strict=True)]
    ax.bar(x, bodies, bottom=bottoms, width=0.7, color=colors, linewidth=0)
    for col, color, label in (("sma50", _SMA50, "SMA 50"), ("sma200", _SMA200, "SMA 200")):
        ax.plot(x, frame[col].to_list(), color=color, linewidth=1.1, label=label)
    leg = ax.legend(loc="upper left", fontsize=8, frameon=False)
    for t in leg.get_texts():
        t.set_color(_TEXT)
    ax.set_title(
        f"{ticker} · 1D · last {closes[-1]:,.2f}",
        color=_TEXT,
        fontsize=10,
        loc="left",
    )

    vax.bar(x, volumes, width=0.7, color=colors, alpha=0.6, linewidth=0)
    vax.set_yticks([])

    # Month labels on the first trading day of each month.
    ticks = [i for i in x if i == 0 or days[i].month != days[i - 1].month]
    if len(ticks) > 1 and ticks[1] - ticks[0] < 10:  # a partial first month
        ticks = ticks[1:]
    vax.set_xticks(ticks)
    vax.set_xticklabels([days[i].strftime("%b") for i in ticks])
    vax.set_xlim(-1, len(x))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=_BG, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return buf.getvalue()


def render_charts(tickers: list[str]) -> dict[str, bytes]:
    """`render_chart` for each ticker; the ones that could not be drawn are
    left out."""
    out: dict[str, bytes] = {}
    for ticker in tickers:
        data = render_chart(ticker)
        if data:
            out[ticker] = data
    if tickers:
        logger.info("Drew %d/%d charts locally", len(out), len(tickers))
    return out
