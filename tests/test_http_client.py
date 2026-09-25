"""HttpClient: retries with jittered backoff, Retry-After, typed errors.

Driven through httpx.MockTransport, so nothing leaves the process, and with
`time.sleep` recorded rather than slept.
"""

from __future__ import annotations

import httpx
import pytest

from stock_analyzer import http_client
from stock_analyzer.http_client import (
    AuthError,
    ClientError,
    HttpClient,
    RateLimitError,
    RetryPolicy,
    ServerError,
)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(http_client.time, "sleep", slept.append)
    return slept


def _client(responses: list[httpx.Response | Exception], **policy) -> HttpClient:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = HttpClient(base_url="https://api.example.com", retry_policy=RetryPolicy(**policy))
    client._client = httpx.Client(
        base_url="https://api.example.com", transport=httpx.MockTransport(handler)
    )
    return client


def test_server_errors_are_retried_with_growing_jittered_pauses(sleeps):
    client = _client(
        [httpx.Response(503), httpx.Response(502), httpx.Response(200, json={"ok": True})],
        initial_backoff=1.0,
        jitter=0.25,
    )
    assert client.get_json("x") == {"ok": True}
    assert len(sleeps) == 2
    assert 1.0 <= sleeps[0] <= 1.25
    assert 2.0 <= sleeps[1] <= 2.5


def test_jitter_zero_gives_the_plain_backoff(sleeps):
    client = _client([httpx.Response(500), httpx.Response(200)], initial_backoff=1.0, jitter=0.0)
    client.get("x")
    assert sleeps == [1.0]


def test_retry_after_is_honored_exactly(sleeps):
    client = _client(
        [httpx.Response(429, headers={"retry-after": "7"}), httpx.Response(200)], jitter=0.25
    )
    client.get("x")
    assert sleeps == [7.0]


def test_network_errors_are_retried(sleeps):
    client = _client([httpx.ConnectError("reset"), httpx.Response(200)], jitter=0.0)
    assert client.get("x").status_code == 200
    assert sleeps == [1.0]


def test_errors_are_typed_after_retries_run_out(sleeps):
    with pytest.raises(ServerError):
        _client([httpx.Response(500)] * 2, max_attempts=2).get("x")
    with pytest.raises(RateLimitError):
        _client([httpx.Response(429)] * 2, max_attempts=2).get("x")


def test_auth_and_client_errors_are_not_retried(sleeps):
    with pytest.raises(AuthError):
        _client([httpx.Response(401)]).get("x")
    with pytest.raises(ClientError):
        _client([httpx.Response(404, text="nope")]).get("x")
    assert sleeps == []
