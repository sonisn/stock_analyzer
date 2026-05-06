"""Ticker fundamentals + news from yfinance. Pure deterministic — no LLM."""
from __future__ import annotations

from typing import Any

import yfinance as yf

from ..logging import get_logger

logger = get_logger(__name__)


def _fmt_money(n: float | int | None) -> str | None:
    if n is None:
        return None
    a = abs(n)
    if a >= 1e12:
        return f"${n / 1e12:.2f}T"
    if a >= 1e9:
        return f"${n / 1e9:.2f}B"
    if a >= 1e6:
        return f"${n / 1e6:.2f}M"
    return f"${n:,.2f}"


def _fmt_pct(n: float | None, *, signed: bool = True) -> str | None:
    if n is None:
        return None
    return f"{n:+.2f}%" if signed else f"{n:.2f}%"


def _trend_label(history, lookback_days: int) -> str | None:
    if history is None or history.empty or len(history) < 2:
        return None
    closes = history["Close"]
    end = float(closes.iloc[-1])
    start_idx = max(-lookback_days, -len(closes))
    start = float(closes.iloc[start_idx])
    if start == 0:
        return None
    pct = (end - start) / start * 100
    direction = "Up" if pct > 2 else "Down" if pct < -2 else "Neutral"
    return f"{direction} ({pct:+.1f}%)"


def _fetch_news(symbol: str, *, max_results: int = 20) -> list[dict]:
    try:
        items = yf.Ticker(symbol).news or []
    except Exception as e:
        logger.warning("yfinance news fetch failed for %s: %s", symbol, e)
        return []
    out: list[dict] = []
    seen_titles: set[str] = set()
    for n in items[:max_results]:
        content = n.get("content") or {}
        title = n.get("title") or content.get("title")
        link = (
            n.get("link")
            or (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
        )
        publisher = (
            n.get("publisher")
            or (content.get("provider") or {}).get("displayName")
        )
        snippet = (
            n.get("summary")
            or content.get("summary")
            or content.get("description")
            or ""
        )
        if not (title and link):
            continue
        norm = title.split(" - ")[0].strip().lower()
        if norm in seen_titles:
            continue
        seen_titles.add(norm)
        out.append(
            {
                "title": title,
                "link": link,
                "snippet": snippet[:250],
                "publisher": publisher,
            }
        )
    return out


def _latest_recommendations(rec) -> dict[str, int] | None:
    if rec is None or rec.empty:
        return None
    row = rec.iloc[0] if "period" in rec.columns else rec.iloc[-1]
    keys = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    out: dict[str, int] = {}
    for k in keys:
        try:
            out[k] = int(row[k])
        except (KeyError, ValueError, TypeError):
            continue
    return out or None


def _earnings_summary(t: yf.Ticker) -> dict[str, Any]:
    import math

    def _clean(rows: list[dict]) -> list[dict]:
        return [
            {
                k: (None if isinstance(v, float) and math.isnan(v) else v)
                for k, v in row.items()
            }
            for row in rows
        ]

    summary: dict[str, Any] = {}
    try:
        df = t.get_earnings_dates(limit=4)
        if df is not None and not df.empty:
            df = df.reset_index()
            summary["history"] = _clean(df.head(4).to_dict(orient="records"))
    except Exception as e:
        logger.debug("earnings_dates failed: %s", e)
    try:
        est = t.earnings_estimate
        if est is not None and not est.empty:
            summary["estimates"] = _clean(
                est.reset_index().head(2).to_dict(orient="records")
            )
    except Exception as e:
        logger.debug("earnings_estimate failed: %s", e)
    return summary


def fetch_ticker_data(symbol: str) -> dict[str, Any]:
    """Fetch all ticker data for a single symbol — fundamentals, trends, news, earnings."""
    logger.info("Fetching ticker data: %s", symbol)
    t = yf.Ticker(symbol)
    info = t.info or {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    prev_close = info.get("previousClose")
    pct_today = (
        (price - prev_close) / prev_close * 100
        if price is not None and prev_close
        else None
    )

    try:
        hist = t.history(period="1y")
    except Exception as e:
        logger.warning("history fetch failed for %s: %s", symbol, e)
        hist = None

    low_52 = info.get("fiftyTwoWeekLow")
    high_52 = info.get("fiftyTwoWeekHigh")
    range_52w = (
        f"{_fmt_money(low_52)} - {_fmt_money(high_52)}" if low_52 and high_52 else None
    )

    div_yield = info.get("dividendYield")
    div_yield_str = _fmt_pct(div_yield, signed=False) if div_yield else None

    name = info.get("longName") or info.get("shortName")
    news = _fetch_news(symbol)

    try:
        rec = t.recommendations
    except Exception:
        rec = None

    return {
        "symbol": symbol,
        "name": name,
        "price": _fmt_money(price),
        "pct_today": _fmt_pct(pct_today),
        "market_cap": _fmt_money(info.get("marketCap")),
        "range_52w": range_52w,
        "pe": f"{info.get('trailingPE'):.1f}" if info.get("trailingPE") else None,
        "dividend_yield": div_yield_str,
        "analyst_target": _fmt_money(info.get("targetMeanPrice")),
        "analysts": _latest_recommendations(rec),
        "trend_7days": _trend_label(hist, 7),
        "trend_1mo": _trend_label(hist, 21),
        "trend_3mo": _trend_label(hist, 63),
        "trend_6mo": _trend_label(hist, 126),
        "trend_1yr": _trend_label(hist, 252),
        "news": news,
        "earnings": _earnings_summary(t),
    }
