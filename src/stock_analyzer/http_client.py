"""Production-grade sync HTTP client with retries, rate limits, typed errors.

Each external service in the codebase that talks raw HTTP gets its own
`HttpClient` instance configured with the right base URL, default
headers (e.g. SEC's required User-Agent), per-host rate limit, and retry
policy. Built on `httpx.Client` so we get connection pooling and modern
HTTP primitives.

What the client adds on top of httpx:

  - exponential-backoff retries on 408 / 429 / 5xx and network errors
  - Retry-After header honoring on 429s (capped at max_backoff)
  - token-bucket rate limiting at configurable calls/min
  - typed exceptions so callers can react to AuthError vs RateLimitError
    vs ServerError vs ClientError vs NetworkError differently
  - structured logging at DEBUG (request), INFO (retry), WARNING (final
    failure)

SDK-based integrations (finnhub-python, tavily-python, snaptrade_client,
yfinance) stay on their SDKs — they have their own clients and
rewriting on raw HTTP loses the SDKs' response parsing for no real win.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Mapping

import httpx
from pydantic import BaseModel, ConfigDict

from .logging import get_logger

logger = get_logger(__name__)


# --- exceptions -------------------------------------------------------------


class HttpClientError(RuntimeError):
    """Base class for errors raised by `HttpClient`."""

    def __init__(
        self,
        msg: str,
        *,
        status: int | None = None,
        url: str | None = None,
    ):
        super().__init__(msg)
        self.status = status
        self.url = url


class AuthError(HttpClientError):
    """401 Unauthorized — credentials missing or invalid. Not retried."""


class RateLimitError(HttpClientError):
    """429 Too Many Requests, retries exhausted."""

    def __init__(
        self,
        msg: str,
        *,
        status: int,
        url: str,
        retry_after: float | None = None,
    ):
        super().__init__(msg, status=status, url=url)
        self.retry_after = retry_after


class ClientError(HttpClientError):
    """4xx other than 401/429 — usually a request bug. Not retried."""


class ServerError(HttpClientError):
    """5xx, retries exhausted."""


class NetworkError(HttpClientError):
    """Connection / DNS / read / timeout error, retries exhausted."""


# --- retry config -----------------------------------------------------------


class RetryPolicy(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_attempts: int = 4
    initial_backoff: float = 1.0
    max_backoff: float = 30.0
    backoff_multiplier: float = 2.0
    # Status codes we treat as transient and retry.
    retry_statuses: tuple[int, ...] = (408, 429, 500, 502, 503, 504)


# --- client -----------------------------------------------------------------


class HttpClient:
    """A configured httpx.Client wrapper with retries + rate limit + typed errors.

    Use one instance per external service:

        FRED = HttpClient(
            base_url="https://api.stlouisfed.org/fred/",
            name="fred",
        )
        data = FRED.get_json("series/observations", params={...})
    """

    def __init__(
        self,
        *,
        base_url: str = "",
        default_headers: Mapping[str, str] | None = None,
        timeout: float = 15.0,
        rate_limit_per_min: int | None = None,
        retry_policy: RetryPolicy | None = None,
        name: str = "http",
    ):
        self._name = name
        self._client = httpx.Client(
            base_url=base_url,
            headers=dict(default_headers or {}),
            timeout=timeout,
        )
        self._retry = retry_policy or RetryPolicy()
        # 0 disables rate limiting; otherwise enforce min interval between calls.
        self._min_interval = (
            60.0 / rate_limit_per_min if rate_limit_per_min else 0.0
        )
        self._rate_lock = threading.Lock()
        self._last_call_ts = 0.0

    # Context-manager / cleanup ------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # Internals ---------------------------------------------------------------

    def _throttle(self) -> None:
        if self._min_interval <= 0:
            return
        with self._rate_lock:
            delta = time.monotonic() - self._last_call_ts
            if delta < self._min_interval:
                time.sleep(self._min_interval - delta)
            self._last_call_ts = time.monotonic()

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        retry = self._retry
        attempt = 0
        backoff = retry.initial_backoff
        while True:
            attempt += 1
            self._throttle()
            try:
                logger.debug(
                    "%s %s %s (attempt %d)", self._name, method, url, attempt
                )
                resp = self._client.request(method, url, **kwargs)
            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.TimeoutException,
            ) as e:
                if attempt >= retry.max_attempts:
                    raise NetworkError(
                        f"{self._name}: network error after {attempt} attempts: {e}",
                        url=url,
                    ) from e
                logger.info(
                    "%s network error (%s) — retry %d/%d after %.1fs",
                    self._name, e, attempt, retry.max_attempts, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * retry.backoff_multiplier, retry.max_backoff)
                continue

            if resp.status_code < 400:
                return resp

            if resp.status_code == 401:
                raise AuthError(
                    f"{self._name}: 401 Unauthorized for {url}",
                    status=401, url=url,
                )

            if resp.status_code == 429:
                retry_after = _parse_retry_after(resp.headers.get("retry-after"))
                if attempt >= retry.max_attempts:
                    raise RateLimitError(
                        f"{self._name}: 429 after {attempt} attempts ({url})",
                        status=429, url=url, retry_after=retry_after,
                    )
                sleep_for = retry_after if retry_after is not None else backoff
                sleep_for = min(sleep_for, retry.max_backoff)
                logger.info(
                    "%s 429 — retry %d/%d after %.1fs (retry-after=%s)",
                    self._name, attempt, retry.max_attempts, sleep_for, retry_after,
                )
                time.sleep(sleep_for)
                backoff = min(backoff * retry.backoff_multiplier, retry.max_backoff)
                continue

            if 500 <= resp.status_code < 600:
                if attempt >= retry.max_attempts:
                    raise ServerError(
                        f"{self._name}: {resp.status_code} after {attempt} attempts ({url})",
                        status=resp.status_code, url=url,
                    )
                logger.info(
                    "%s %d — retry %d/%d after %.1fs",
                    self._name, resp.status_code, attempt, retry.max_attempts, backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * retry.backoff_multiplier, retry.max_backoff)
                continue

            # Other 4xx — no retry. Include body preview for debuggability.
            body_preview = (resp.text or "")[:200]
            raise ClientError(
                f"{self._name}: {resp.status_code} {url}: {body_preview}",
                status=resp.status_code, url=url,
            )

    # Convenience wrappers ----------------------------------------------------

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return self.get(url, **kwargs).json()

    def post_json(self, url: str, **kwargs: Any) -> Any:
        return self.post(url, **kwargs).json()

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        return self.get(url, **kwargs).content


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value. Returns seconds or None.

    Honors the numeric-seconds form (most common); falls back to None
    for the rare HTTP-date form to keep the parser dependency-free."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None
