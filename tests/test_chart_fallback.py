"""Charts drawn locally when chart-img does not deliver.

chart-img refused the key (403) for every request from 2026-10-14 and the
weekly email went out with no charts. Now a refused key stops the chart-img
calls at the first 403 and every stock is drawn from its daily bars.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import patch

import polars as pl

from stock_analyzer.data import chart_img, local_chart
from stock_analyzer.http_client import ClientError


def _bars(n: int = 400) -> pl.DataFrame:
    start = date(2025, 1, 1)
    closes = [100 + (i % 30) - 15 + i * 0.1 for i in range(n)]
    return pl.DataFrame(
        {
            "date": [start + timedelta(days=i) for i in range(n)],
            "Open": [c - 0.5 for c in closes],
            "High": [c + 1 for c in closes],
            "Low": [c - 1 for c in closes],
            "Close": closes,
            "Volume": [1e6] * n,
        },
        schema_overrides={"date": pl.Date},
    )


def test_render_chart_is_a_png():
    with patch("stock_analyzer.data.yf_gateway.daily_bars", return_value=_bars()):
        png = local_chart.render_chart("NVDA", today=date(2026, 2, 4))
    assert png is not None and png.startswith(b"\x89PNG")


def test_render_chart_without_bars_is_none():
    with patch("stock_analyzer.data.yf_gateway.daily_bars", return_value=None):
        assert local_chart.render_chart("NVDA") is None


def test_refused_key_stops_calls_and_draws_the_rest(monkeypatch):
    monkeypatch.setenv("CHART_IMG_API_KEY", "k")
    calls: list[str] = []

    def post(*_a, **_k):
        calls.append("post")
        raise ClientError("chart-img: 403", status=403)

    with (
        patch.object(chart_img._HTTP, "post", side_effect=post),
        patch.object(chart_img, "_load_exchange_cache", return_value={"MMF": ""}),
        patch.object(chart_img, "_save_exchange_cache"),
        patch.object(
            chart_img, "_resolve_symbol", side_effect=lambda t, c: c.get(t, f"NASDAQ:{t}") or None
        ),
        patch.object(local_chart, "render_chart", side_effect=lambda t: f"png-{t}".encode()),
    ):
        charts = chart_img.fetch_charts(["NVDA", "AMD", "MMF", "TSM"])

    assert calls == ["post"]
    assert charts == {"NVDA": b"png-NVDA", "AMD": b"png-AMD", "TSM": b"png-TSM"}


def test_no_key_draws_locally(monkeypatch):
    monkeypatch.delenv("CHART_IMG_API_KEY", raising=False)
    with (
        patch.object(chart_img._HTTP, "post") as post,
        patch.object(chart_img, "_load_exchange_cache", return_value={}),
        patch.object(chart_img, "_save_exchange_cache"),
        patch.object(chart_img, "_resolve_symbol", side_effect=lambda t, c: f"NYSE:{t}"),
        patch.object(local_chart, "render_chart", side_effect=lambda t: b"png"),
    ):
        charts = chart_img.fetch_charts(["ANET"])
    post.assert_not_called()
    assert charts == {"ANET": b"png"}
