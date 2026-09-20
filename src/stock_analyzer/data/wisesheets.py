"""Wisesheets — fundamentals as filed with the SEC, with a citation.

Every value carries the XBRL tag, the accession number and a link to the
filing it came from, which is the point: yfinance's `info` numbers are
derived, undated and sometimes wrong in ways nothing in the response
shows. Measured against yfinance on the held tickers on 2026-09-20:

    NVDA  free cash flow   $127.0B filed   vs  $41.8B yfinance
    GOOGL free cash flow    $53.3B filed   vs  $22.7B yfinance
    AVGO  gross margin       68.8% (GAAP)  vs   75.5% yfinance
    TSM   debt/equity     not covered      vs   42.16 yfinance (TWD!)

It is not a replacement. There are no analyst estimates, price targets,
recommendations, insider trades or short interest here — yfinance and
Finnhub keep those — and it covers US SEC filers only, so a foreign
private issuer like TSM is simply absent. It has its own defects too
(ANET's 2025-12-31 `NetIncomeLoss` comes back as -$2,556M), which is why
callers cross-check rather than trust: see `discover/data_reconciliation`.

Free plan: 5,000 requests/month, 200/minute, 5 years of history, and at
most 100 tickers per request (the 1,000-ticker batch endpoint needs a
paid tier). A full S&P 500 fundamentals pull is therefore ~6 requests.

Endpoints used (docs: https://www.wisesheets.io/api/docs):
  - /v1/financials/  metrics for tickers over a period
  - /v1/me/          plan, limits and remaining quota
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date
from typing import Any

from ..http_client import HttpClient, HttpClientError
from ..logging import get_logger

logger = get_logger(__name__)

_BASE_URL = "https://api.wisesheets.io/v1"
# The plan allows 200/min; 150 leaves room for a second process sharing
# the key without either of them tripping a 429.
_RATE_LIMIT_PER_MIN = 150
# The API rejects a request resolving more than 100 identifiers.
MAX_TICKERS_PER_REQUEST = 100

# Metrics that are flows (summed across quarters for a trailing figure)
# rather than balances (read at a point in time).
_FLOW_METRICS = frozenset(
    {
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "free_cash_flow",
        "operating_cash_flow",
        "capital_expenditure",
        "ebitda",
        "eps_diluted",
        "eps_basic",
        "total_dividends_paid",
    }
)

_HTTP = HttpClient(timeout=60.0, rate_limit_per_min=_RATE_LIMIT_PER_MIN, name="wisesheets")


def api_key() -> str | None:
    """Read at call time, not import time — the CLIs load `.env` after
    this module is imported."""
    return os.getenv("WISESHEETS_API_KEY") or None


def is_configured() -> bool:
    return bool(api_key())


def _chunks(tickers: list[str], size: int = MAX_TICKERS_PER_REQUEST):
    for i in range(0, len(tickers), size):
        yield tickers[i : i + size]


def _get(path: str, params: dict[str, Any]) -> dict[str, Any] | None:
    key = api_key()
    if not key:
        return None
    try:
        return _HTTP.get_json(
            f"{_BASE_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {key}"},
        )
    except HttpClientError as e:
        # The plan's history window comes back as a flat 403 POLICY_DENIED
        # with no explanation: on the free plan, an `asof:` date before
        # 2022-01 (five years back) is refused. Say so, rather than
        # leaving it looking like an outage or a bad key.
        if "POLICY_DENIED" in str(e):
            logger.info(
                "Wisesheets declined %s — usually a date outside the plan's %s-year history window",
                params.get("period") or path,
                5,
            )
            return None
        # Otherwise: a missing provider must never be what takes a run
        # down, so the caller keeps whatever yfinance gave it.
        logger.warning("Wisesheets %s failed (%s) — falling back to yfinance", path, e)
        return None


def quota() -> dict[str, Any] | None:
    """Plan limits and what is left this month, or None when unconfigured."""
    body = _get("/me/", {})
    if not body:
        return None
    plan, q = body.get("plan") or {}, body.get("quota") or {}
    return {
        "plan": (plan.get("planCode") or "unknown"),
        "monthly_limit": (plan.get("limits") or {}).get("monthlyRequests"),
        "monthly_remaining": (q.get("month") or {}).get("remaining"),
        "history_years": (plan.get("limits") or {}).get("historyYears"),
        "bulk_access": bool((plan.get("capabilities") or {}).get("bulkAccess")),
    }


def fetch_metrics(
    tickers: list[str],
    metrics: list[str],
    *,
    period: str = "latest",
    frequency: str = "quarterly",
    as_reported: bool = False,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """{ticker: {metric: [observation, ...]}}, newest period first.

    An observation is {value, period_end, fiscal_year, fiscal_period,
    source}. Tickers the API doesn't cover are simply absent — callers
    fall back rather than treat that as an error.

    `as_reported` returns the figures from the newest filing that existed
    on the `asof:` date, rather than today's view of that period. It is
    what makes a backtest honest and it is not optional for one: with it
    off, `asof:2025-05-01` hands back NVDA's quarter ending 2025-04-27,
    which was not filed until 2025-05-28. Rolling windows (lastNq/lastNy)
    are rejected in this mode, so pass an explicit `asof:` date.
    """
    if not tickers or not metrics or not is_configured():
        return {}
    out: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for chunk in _chunks(sorted({t.upper() for t in tickers})):
        params: dict[str, Any] = {
            "tickers": ",".join(chunk),
            "metrics": ",".join(metrics),
            "period": period,
            "frequency": frequency,
        }
        if as_reported:
            params["asReported"] = "true"
        body = _get("/financials/", params)
        if not body:
            continue
        for row in body.get("data") or []:
            try:
                value = float(row["value"])
            except TypeError, ValueError, KeyError:
                continue
            out[str(row["ticker"]).upper()][str(row["metric"])].append(
                {
                    "value": value,
                    "period_end": row.get("periodEnd"),
                    "fiscal_year": row.get("fiscalYear"),
                    "fiscal_period": row.get("fiscalPeriod"),
                    "source": row.get("source") or {},
                }
            )
    for by_metric in out.values():
        for observations in by_metric.values():
            observations.sort(key=lambda o: o["period_end"] or "", reverse=True)
    return {t: dict(m) for t, m in out.items()}


def _trailing(observations: list[dict[str, Any]], quarters: int = 4) -> float | None:
    """Sum of the last `quarters` reported quarters, or None if short."""
    usable = [o for o in observations if o["value"] is not None][:quarters]
    return sum(o["value"] for o in usable) if len(usable) == quarters else None


def fetch_trailing_fundamentals(
    tickers: list[str], *, as_of: date | None = None
) -> dict[str, dict[str, Any]]:
    """{ticker: filing-derived fundamentals} on a trailing-twelve-month
    basis, plus `period_end` so callers can judge how current it is.

    `as_of` asks for the figures as they were known on that date — what a
    backtest needs, since the alternative is scoring a 2024 candidate on
    numbers filed in 2026.
    """
    if not tickers or not is_configured():
        return {}
    period = f"asof:{as_of.isoformat()}" if as_of else "last4q"
    flows = fetch_metrics(
        tickers,
        ["revenue", "gross_profit", "operating_income", "net_income", "free_cash_flow"],
        period=period,
        frequency="quarterly",
    )
    # Balances describe a moment, so they are read, not summed.
    balances = fetch_metrics(
        tickers,
        ["debt_to_equity", "roe", "current_ratio"],
        period=f"asof:{as_of.isoformat()}" if as_of else "latest",
        frequency="quarterly",
    )

    out: dict[str, dict[str, Any]] = {}
    for ticker in {t.upper() for t in tickers}:
        by_metric = flows.get(ticker) or {}
        revenue = _trailing(by_metric.get("revenue") or [])
        if not revenue:
            continue

        def ratio(metric: str, rev: float = revenue, m: dict = by_metric) -> float | None:
            total = _trailing(m.get(metric) or [])
            return total / rev if total is not None and rev else None

        latest = (by_metric.get("revenue") or [{}])[0]
        balance = balances.get(ticker) or {}
        out[ticker] = {
            "revenue_ttm": revenue,
            "gross_margin": ratio("gross_profit"),
            "operating_margin": ratio("operating_income"),
            "profit_margin": ratio("net_income"),
            "free_cash_flow": _trailing(by_metric.get("free_cash_flow") or []),
            "debt_to_equity": ((balance.get("debt_to_equity") or [{}])[0]).get("value"),
            "roe": ((balance.get("roe") or [{}])[0]).get("value"),
            "period_end": latest.get("period_end"),
            "filing_url": (latest.get("source") or {}).get("filingUrl"),
        }
    return out


# Ratios, deliberately. In as-reported mode the newest filing on a date
# may be a 10-Q or a 10-K, so the amounts cover a quarter for one company
# and a year for another and are not comparable across a cross-section.
# A margin taken from inside one filing is.
POINT_IN_TIME_RATIOS: tuple[str, ...] = (
    "gross_margin_pit",
    "net_margin_pit",
    "leverage_pit",
    "return_on_equity_pit",
)

# As-reported mode serves only the tags a filing actually carries, so the
# API's own `debt_to_equity` and `roe` — both `isCalculated` — are empty
# there, and so is `total_debt`. Measured across ten holdings at
# asof:2025-12-31: total_assets 10/10, total_liabilities 9/10,
# long_term_debt 9/10, but total_equity only 3/10. Leverage is therefore
# liabilities over assets, and equity is what is left of the assets.
_PIT_METRICS = (
    "revenue",
    "gross_profit",
    "net_income",
    "total_assets",
    "total_liabilities",
    "total_equity",
)


def fetch_point_in_time_ratios(tickers: list[str], as_of: date) -> dict[str, dict[str, float]]:
    """{ticker: ratios known on `as_of`} — one request per 100 tickers.

    Only what the filings said by that date: no restatement, no figure
    from a filing that had not been published yet.
    """
    if not tickers or not is_configured():
        return {}
    raw = fetch_metrics(
        tickers,
        list(_PIT_METRICS),
        period=f"asof:{as_of.isoformat()}",
        frequency="quarterly",
        as_reported=True,
    )
    out: dict[str, dict[str, float]] = {}
    for ticker, by_metric in raw.items():

        def newest(metric: str, m: dict = by_metric) -> float | None:
            observations = m.get(metric) or []
            return observations[0]["value"] if observations else None

        revenue = newest("revenue")
        gross, net = newest("gross_profit"), newest("net_income")
        assets, liabilities = newest("total_assets"), newest("total_liabilities")
        equity = newest("total_equity")
        if equity is None and assets is not None and liabilities is not None:
            equity = assets - liabilities
        ratios = {
            "gross_margin_pit": gross / revenue if revenue and gross is not None else None,
            "net_margin_pit": net / revenue if revenue and net is not None else None,
            "leverage_pit": (liabilities / assets if assets and liabilities is not None else None),
            # Equity is a balance and income is a flow over whatever period
            # the filing covered, so this is the return on equity for that
            # period — comparable across the cross-section, not annualized.
            "return_on_equity_pit": net / equity if equity and net is not None else None,
        }
        kept = {k: v for k, v in ratios.items() if v is not None}
        if kept:
            out[ticker] = kept
    return out


__all__ = [
    "MAX_TICKERS_PER_REQUEST",
    "POINT_IN_TIME_RATIOS",
    "fetch_point_in_time_ratios",
    "fetch_metrics",
    "fetch_trailing_fundamentals",
    "is_configured",
    "quota",
]
