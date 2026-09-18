"""Reviewer and PeerFinder retry once on the fallback provider.

Both used to call `agent.run()` directly, so a provider outage or a
model-id deprecation failed every holding review / peer lookup with no
second chance, unlike the other discover agents.
"""

from __future__ import annotations

from types import SimpleNamespace

from agno.exceptions import ModelProviderError

from stock_analyzer.discover import peers, reviewer


class _FakeAgent:
    def __init__(self, provider, *, fails=False, content=None):
        self.provider = provider
        self.name = "fake"
        self.model_id = "m"
        self._fails = fails
        self._content = content
        self.calls = 0

    def run(self, *args, **kwargs):
        self.calls += 1
        if self._fails:
            raise ModelProviderError("boom", status_code=502)
        return SimpleNamespace(content=self._content)


def _patch_builder(monkeypatch, module, agents):
    built = iter(agents)
    monkeypatch.setattr(module, "_build_agent", lambda provider, model: next(built))


def test_reviewer_retries_on_fallback_provider(monkeypatch):
    primary, fallback = _FakeAgent("claude", fails=True), _FakeAgent("gemini")
    _patch_builder(monkeypatch, reviewer, [primary, fallback])

    r = reviewer.Reviewer("claude", "sonnet", fallback=("gemini", "gemini-pro-latest"))
    assert r.review("AAPL", {}) is None  # fallback returned no content
    assert (primary.calls, fallback.calls) == (1, 1)


def test_reviewer_same_provider_fallback_is_not_retried(monkeypatch):
    primary = _FakeAgent("claude", fails=True)
    _patch_builder(monkeypatch, reviewer, [primary])

    r = reviewer.Reviewer("claude", "sonnet", fallback=("claude", "claude-opus-4-7"))
    # review_batch swallows per-ticker exceptions; the call itself raises.
    assert reviewer.review_batch(r, {"AAPL": {}}) == {}
    assert primary.calls == 1


def test_peer_finder_retries_on_fallback_provider(monkeypatch):
    primary = _FakeAgent("claude", fails=True)
    fallback = _FakeAgent("openai", content='["MSFT", "GOOG", "AAPL"]')
    _patch_builder(monkeypatch, peers, [primary, fallback])
    monkeypatch.setattr(peers, "load_ticker_cik_map", lambda: {})

    finder = peers.PeerFinder(fallback=("openai", "gpt-6-astra"))
    assert finder.find("AAPL") == ["MSFT", "GOOG"]
    assert (primary.calls, fallback.calls) == (1, 1)


def test_peer_finder_without_fallback_returns_empty_on_error(monkeypatch):
    _patch_builder(monkeypatch, peers, [_FakeAgent("claude", fails=True)])
    assert peers.PeerFinder().find("AAPL") == []
