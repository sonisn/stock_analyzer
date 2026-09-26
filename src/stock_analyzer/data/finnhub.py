"""Finnhub signals — earnings surprises, recommendation trend, price
targets, and Form 4 insider transactions.

Free tier is 60 calls/min on US equities. We aim for ~55 calls/min to
stay comfortably under the limit, and ride a small thread pool so
concurrent fetches don't exceed that pace.

Each ticker burns 4 calls (one per signal); 25 candidates = 100 calls,
so a full discover run takes ~110s sequential and a bit less with the
thread pool. Acceptable given pipeline already spends 10x that on LLM
calls.

Functions silently return empty dicts for tickers Finnhub can't serve
(non-US, suspended, etc.) so the pipeline degrades gracefully rather
than failing.
"""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import Any

import finnhub

from ..logging import get_logger

logger = get_logger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(float(raw))
    except ValueError:
        logger.warning("%s=%r is not a number; using %d", name, raw, default)
        return default
    return value if value > 0 else default


# Stay under the 60/min free-tier ceiling. 55/min ≈ 1.09s between calls.
# Overridable because paid tiers raise the ceiling substantially.
_RATE_LIMIT_PER_MIN = _env_int("FINNHUB_RATE_LIMIT_PER_MIN", 55)
_MIN_INTERVAL = 60.0 / max(1, _RATE_LIMIT_PER_MIN)

# Three workers gives a small pipelining win (one waits while others run)
# without blowing the rate budget.
_MAX_WORKERS = 3

# A 429 means the pacing above was still too fast for this key right now
# (other processes share the quota, and bursts at a minute boundary can
# trip it even at 55/min). Retry with backoff and park every worker for
# the cooldown, rather than dropping the ticker's data on the floor —
# silently returning None here is how a run ends up with half its
# Finnhub signals missing.
_MAX_ATTEMPTS = _env_int("FINNHUB_MAX_ATTEMPTS", 3)
_BASE_COOLDOWN = float(_env_int("FINNHUB_COOLDOWN_SECONDS", 10))
_MAX_COOLDOWN = 90.0

_rate_lock = threading.Lock()
_last_call_time = 0.0
_cooldown_until = 0.0


def reload_from_env() -> None:
    """Re-read the pacing knobs after the CLI has loaded `.env`.

    Module import beats `load_dotenv()`, so without this the values in
    `.env` would never be seen. Entry points call this at startup.
    """
    global _RATE_LIMIT_PER_MIN, _MIN_INTERVAL, _MAX_ATTEMPTS, _BASE_COOLDOWN

    _RATE_LIMIT_PER_MIN = _env_int("FINNHUB_RATE_LIMIT_PER_MIN", 55)
    _MIN_INTERVAL = 60.0 / max(1, _RATE_LIMIT_PER_MIN)
    _MAX_ATTEMPTS = _env_int("FINNHUB_MAX_ATTEMPTS", 3)
    _BASE_COOLDOWN = float(_env_int("FINNHUB_COOLDOWN_SECONDS", 10))


def _client() -> finnhub.Client | None:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        return None
    return finnhub.Client(api_key=api_key)


def _throttle() -> None:
    """Reserve the next send slot, waiting out the interval and any cooldown.

    The slot is reserved under the lock and slept for outside it, so N
    workers space themselves out instead of all sleeping the same
    interval and then firing together.
    """
    global _last_call_time
    with _rate_lock:
        now = time.monotonic()
        start = max(now, _last_call_time + _MIN_INTERVAL, _cooldown_until)
        _last_call_time = start
        wait = start - now
    if wait > 0:
        time.sleep(wait)


def _is_rate_limited(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    msg = str(exc).lower()
    return "429" in msg or "too many requests" in msg or "api limit" in msg


def _cool_down(attempt: int) -> float:
    """Pause every worker after a 429. Returns the cooldown in seconds."""
    global _cooldown_until
    cooldown = min(_BASE_COOLDOWN * (2 ** (attempt - 1)), _MAX_COOLDOWN)
    cooldown *= 1.0 + random.random() * 0.25  # de-synchronize the workers
    with _rate_lock:
        _cooldown_until = max(_cooldown_until, time.monotonic() + cooldown)
    return cooldown


def _safe_call(label: str, ticker: str, fn, *args, **kwargs) -> Any:
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        _throttle()
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if not _is_rate_limited(e):
                logger.debug("Finnhub %s failed for %s: %s", label, ticker, e)
                return None
            if attempt >= _MAX_ATTEMPTS:
                logger.warning(
                    "Finnhub %s for %s: rate limited after %d attempts; giving up",
                    label,
                    ticker,
                    attempt,
                )
                return None
            cooldown = _cool_down(attempt)
            logger.info(
                "Finnhub rate limited on %s (%s) — pausing all calls %.0fs (attempt %d/%d)",
                label,
                ticker,
                cooldown,
                attempt,
                _MAX_ATTEMPTS,
            )
    return None


def fetch_quote(ticker: str) -> dict[str, float | None] | None:
    """{"price", "prev_close"} from Finnhub's real-time quote, or None
    (no key, rate limited, or a symbol Finnhub doesn't cover: it answers
    those with zeros rather than an error). The fallback for when Yahoo,
    the usual source, is throttling and returns no price."""
    client = _client()
    if client is None:
        return None
    q = _safe_call("quote", ticker, client.quote, ticker.upper())
    price = (q or {}).get("c") or 0
    if price <= 0:
        return None
    return {"price": float(price), "prev_close": float((q or {}).get("pc") or 0) or None}


def fetch_earnings_calendar(start: date, end: date) -> list[dict[str, Any]]:
    """Every US company reporting between `start` and `end`, one request:
    [{ticker, date, hour, eps_estimate, eps_actual, revenue_estimate,
    revenue_actual}]. Actuals are None until the company has reported;
    `hour` is "bmo" (before the open), "amc" (after the close) or "".
    Empty on no key or any failure."""
    client = _client()
    if client is None:
        return []
    raw = _safe_call(
        "earnings_calendar",
        "*",
        client.earnings_calendar,
        _from=start.isoformat(),
        to=end.isoformat(),
        symbol="",
        international=False,
    )
    return [
        {
            "ticker": r["symbol"].upper(),
            "date": r.get("date"),
            "hour": r.get("hour") or "",
            "eps_estimate": r.get("epsEstimate"),
            "eps_actual": r.get("epsActual"),
            "revenue_estimate": r.get("revenueEstimate"),
            "revenue_actual": r.get("revenueActual"),
        }
        for r in (raw or {}).get("earningsCalendar") or []
        if r.get("symbol") and r.get("date")
    ]


def fetch_earnings_surprise(client: finnhub.Client, ticker: str) -> list[dict[str, Any]]:
    """Last 4 quarters of actual vs estimate EPS.

    Returns [{period, actual, estimate, surprise, surprise_pct}, ...] with
    most recent first. Empty list if no data."""
    raw = _safe_call("earnings_surprise", ticker, client.company_earnings, ticker, limit=4)
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for q in raw:
        out.append(
            {
                "period": q.get("period"),
                "actual": q.get("actual"),
                "estimate": q.get("estimate"),
                "surprise": q.get("surprise"),
                "surprise_pct": q.get("surprisePercent"),
            }
        )
    return out


def fetch_recommendation_trend(client: finnhub.Client, ticker: str) -> list[dict[str, Any]]:
    """Last 4 months of analyst consensus.

    Returns [{period, strong_buy, buy, hold, sell, strong_sell}, ...]
    with most recent first. Captures DOWNGRADES / UPGRADES — compare
    period[0] vs period[3] to see direction of analyst opinion."""
    raw = _safe_call("recommendation_trend", ticker, client.recommendation_trends, ticker)
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for m in raw[:4]:  # last 4 months
        out.append(
            {
                "period": m.get("period"),
                "strong_buy": m.get("strongBuy"),
                "buy": m.get("buy"),
                "hold": m.get("hold"),
                "sell": m.get("sell"),
                "strong_sell": m.get("strongSell"),
            }
        )
    return out


def fetch_price_targets(client: finnhub.Client, ticker: str) -> dict[str, Any]:
    """Current analyst price targets aggregated across firms.

    Returns {mean, high, low, last_updated, n_analysts} or empty dict."""
    raw = _safe_call("price_target", ticker, client.price_target, ticker)
    if not raw:
        return {}
    return {
        "mean": raw.get("targetMean"),
        "high": raw.get("targetHigh"),
        "low": raw.get("targetLow"),
        "median": raw.get("targetMedian"),
        "last_updated": raw.get("lastUpdated"),
        "n_analysts": raw.get("numberOfAnalysts"),
    }


def fetch_insider_activity(client: finnhub.Client, ticker: str, days: int = 90) -> dict[str, Any]:
    """Recent Form 4 insider transactions for the ticker.

    Returns a structured summary preferable to a raw mention count:
      {
        net_shares: int,             # negative = net selling
        n_sells: int, n_buys: int,
        sell_value_usd: float, buy_value_usd: float,
        recent_transactions: [       # most recent 10, newest first
          {date, name, share, transactionCode, transactionPrice, ...}
        ]
      }
    """
    today = date.today()
    start = (today - timedelta(days=days)).isoformat()
    raw = _safe_call(
        "insider_transactions",
        ticker,
        client.stock_insider_transactions,
        ticker,
        start,
        today.isoformat(),
    )
    if not raw:
        return {}
    data = raw.get("data") or []
    if not data:
        return {}

    net_shares = 0
    n_sells = n_buys = 0
    sell_value = buy_value = 0.0
    for tx in data:
        change = tx.get("change") or 0
        price = tx.get("transactionPrice") or 0
        net_shares += change
        if change < 0:
            n_sells += 1
            sell_value += abs(change) * price
        elif change > 0:
            n_buys += 1
            buy_value += change * price

    data_sorted = sorted(data, key=lambda t: t.get("transactionDate") or "", reverse=True)
    return {
        "net_shares": net_shares,
        "n_sells": n_sells,
        "n_buys": n_buys,
        "sell_value_usd": round(sell_value, 2),
        "buy_value_usd": round(buy_value, 2),
        "recent_transactions": [
            {
                "date": tx.get("transactionDate"),
                "name": tx.get("name"),
                "shares": tx.get("change"),
                "price": tx.get("transactionPrice"),
                "code": tx.get("transactionCode"),
            }
            for tx in data_sorted[:10]
        ],
    }


def fetch_company_news(
    client: finnhub.Client, ticker: str, *, days: int = 30
) -> list[dict[str, Any]]:
    """Raw Finnhub company-news items for the last `days` days.

    Each item has headline, summary, source, url and `datetime` (unix
    seconds). Empty list on error or no coverage."""
    to = date.today()
    raw = _safe_call(
        "company_news",
        ticker,
        client.company_news,
        ticker,
        _from=(to - timedelta(days=days)).isoformat(),
        to=to.isoformat(),
    )
    return raw if isinstance(raw, list) else []


def fetch_general_news(client: finnhub.Client, category: str = "general") -> list[dict[str, Any]]:
    """Finnhub's market-wide news feed (headline, summary, source, url,
    `datetime` in unix seconds), newest first. Empty list on error."""
    raw = _safe_call("general_news", category, client.general_news, category, min_id=0)
    return raw if isinstance(raw, list) else []


def fetch_signals(client: finnhub.Client, ticker: str, *, insider_days: int = 90) -> dict[str, Any]:
    """Fetch all four signals for one ticker."""
    return {
        "earnings_surprise": fetch_earnings_surprise(client, ticker),
        "recommendation_trend": fetch_recommendation_trend(client, ticker),
        "price_targets": fetch_price_targets(client, ticker),
        "insider_activity": fetch_insider_activity(client, ticker, days=insider_days),
    }


def batch_finnhub_signals(
    tickers: list[str] | set[str], *, insider_days: int = 90
) -> dict[str, dict[str, Any]]:
    """Fetch the full signal bundle for every ticker.

    Returns {ticker: {earnings_surprise, recommendation_trend,
    price_targets, insider_activity}}. Tickers with no data appear with
    an empty-fields dict — never missing — so payload assembly stays
    simple downstream."""
    client = _client()
    if client is None:
        logger.warning("FINNHUB_API_KEY not set; skipping Finnhub signals (returning empty)")
        return {t: {} for t in tickers}

    out: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {
            ex.submit(fetch_signals, client, t, insider_days=insider_days): t for t in tickers
        }
        for fut in futures:
            ticker = futures[fut]
            try:
                out[ticker] = fut.result()
            except Exception as e:
                logger.warning("Finnhub batch failed for %s: %s", ticker, e)
                out[ticker] = {}

    n_with_data = sum(1 for v in out.values() if v)
    logger.info(
        "Finnhub signals fetched: %d/%d tickers returned data",
        n_with_data,
        len(out),
    )
    return out
