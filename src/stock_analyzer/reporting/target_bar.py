"""Analysts' 12-month price targets as a small range bar for the emails.

The bar spans the lowest to the highest target (stretched to today's
price when it sits outside them), shades the analysts' range, and marks
today's price and the average target. Drawn with table cells rather than
an image: Gmail drops SVG and positioned elements, and a PNG per stock
would be one more attachment for a handful of numbers.
"""

from __future__ import annotations

import html
from typing import Any

_RANGE = "#bfdbfe"  # analysts' low-to-high band
_OUTSIDE = "#f3f4f6"  # scale beyond the band, when the price is outside it
_PRICE = "#111827"
_MEAN = "#2563eb"
_BAR_HEIGHT = 10


def _num(v: Any) -> float | None:
    try:
        f = float(v)
    except TypeError, ValueError:
        return None
    return f if f > 0 else None


def target_summary(data: dict[str, Any]) -> dict[str, float] | None:
    """{low, mean, high, price} when there is a band to draw, else None."""
    t = data.get("analyst_targets") or {}
    low, mean, high = _num(t.get("low")), _num(t.get("mean")), _num(t.get("high"))
    price = _num(data.get("price_value"))
    if not (low and mean and high and price) or high <= low:
        return None
    return {"low": low, "mean": mean, "high": high, "price": price}


def _marker_row(pct: float, label: str, color: str, *, above: bool) -> str:
    """A label whose arrow sits at `pct`% of the bar: right-aligned in a
    cell ending there, or left-aligned in one starting there, whichever
    side has room for the text."""
    arrow = "▼" if above else "▲"
    style = f"font-size:11px;color:{color};padding:0;white-space:nowrap"
    if pct >= 50:
        cells = (
            f'<td width="{pct:.0f}%" style="{style};text-align:right">{label} {arrow}</td>'
            f'<td width="{100 - pct:.0f}%" style="padding:0"></td>'
        )
    else:
        cells = (
            f'<td width="{pct:.0f}%" style="padding:0"></td>'
            f'<td width="{100 - pct:.0f}%" style="{style};text-align:left">{arrow} {label}</td>'
        )
    return f"<tr>{cells}</tr>"


def _bar_cells(lo: float, hi: float, low: float, high: float) -> str:
    span = hi - lo
    cells = []
    for start, end, color in ((lo, low, _OUTSIDE), (low, high, _RANGE), (high, hi, _OUTSIDE)):
        width = (end - start) / span * 100
        if width > 0.5:
            cells.append(
                f'<td width="{width:.1f}%" height="{_BAR_HEIGHT}" '
                f'style="background:{color};padding:0;font-size:0;line-height:0">&nbsp;</td>'
            )
    return "".join(cells)


def _table(rows: str) -> str:
    return (
        '<table role="presentation" width="100%" cellpadding="0" cellspacing="0" '
        'style="border-collapse:collapse;table-layout:fixed;margin:0">'
        f"{rows}</table>"
    )


def _describe(data: dict[str, Any], s: dict[str, float]) -> tuple[str, str]:
    """(who covers it and their rating, the numbers against today's price)."""
    low, mean, high, price = s["low"], s["mean"], s["high"], s["price"]
    t = data.get("analyst_targets") or {}
    count = t.get("count")
    rating = str(t.get("rating") or "").replace("_", " ")
    if rating.lower() == "none":  # Yahoo's word for no consensus rating
        rating = ""
    who = (f"{int(count)} analysts" if count else "Analysts") + (
        f" · rated {rating}" if rating else ""
    )
    if price > high:
        where = "above every analyst's target"
    elif price < low:
        where = "below every analyst's target"
    else:
        where = f"{(mean / price - 1) * 100:+.0f}% to the average target"
    facts = (
        f"Low ${low:,.2f} · Average ${mean:,.2f} · High ${high:,.2f} — now ${price:,.2f}, {where}"
    )
    return who, facts


def analyst_target_text(data: dict[str, Any]) -> str:
    """The same numbers as one line, for the PDF (it has no bar)."""
    s = target_summary(data)
    if s is None:
        return ""
    who, facts = _describe(data, s)
    return f"Analyst 12-month targets ({who}): {facts}"


def analyst_target_html(data: dict[str, Any]) -> str:
    """The bar plus one line of numbers, or "" without targets."""
    s = target_summary(data)
    if s is None:
        return ""
    low, mean, high, price = s["low"], s["mean"], s["high"], s["price"]
    lo, hi = min(low, price), max(high, price)
    pos = lambda v: (v - lo) / (hi - lo) * 100  # noqa: E731
    who, facts = _describe(data, s)
    return (
        '<div class="targets" style="margin:6px 0 14px;max-width:520px">'
        '<p style="font-size:13px;color:#374151;margin:0 0 4px">'
        f"<b>Analyst 12-month targets</b> · {html.escape(who)}</p>"
        + _table(_marker_row(pos(price), f"now ${price:,.2f}", _PRICE, above=True))
        + _table(f"<tr>{_bar_cells(lo, hi, low, high)}</tr>")
        + _table(_marker_row(pos(mean), f"avg ${mean:,.2f}", _MEAN, above=False))
        + f'<p style="font-size:12px;color:#6b7280;margin:4px 0 0">{html.escape(facts)}</p>'
        "</div>"
    )


def from_fundamentals(f: dict[str, Any] | None) -> dict[str, Any] | None:
    """The bar's input from a `batch_fundamentals` row (the discover and
    rebalance reports' data), or None when the row has no targets."""
    if not f or not f.get("analyst_target_mean"):
        return None
    data = {
        "price_value": f.get("quote_price"),
        "analyst_targets": {
            "low": f.get("analyst_target_low"),
            "mean": f.get("analyst_target_mean"),
            "high": f.get("analyst_target_high"),
            "count": f.get("analyst_count"),
            "rating": f.get("analyst_recommendation"),
        },
    }
    return data if target_summary(data) else None
