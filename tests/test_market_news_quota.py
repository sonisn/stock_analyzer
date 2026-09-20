"""A plan that is out of calls is news about the plan, not a run fault."""

from __future__ import annotations


def test_an_exhausted_tavily_plan_is_news_not_a_fault(monkeypatch, caplog):
    """A permanent, handled condition must not file three warnings a day."""
    import logging

    from stock_analyzer.data import market_news

    class _OutOfQuota:
        def __init__(self, api_key):
            pass

        def search(self, **_):
            raise RuntimeError(
                "This request exceeds your plan's set usage limit. Please upgrade your plan."
            )

    monkeypatch.setenv("TAVILY_API_KEY", "k")
    monkeypatch.setattr(market_news, "TavilyClient", _OutOfQuota)
    with caplog.at_level(logging.INFO, logger="stock_analyzer.data.market_news"):
        assert market_news._tavily_market_news(max_results=10) == []
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings == []
    assert sum("out of quota" in r.getMessage() for r in caplog.records) == 1

    # A genuine failure is still a warning, once per query.
    class _Broken(_OutOfQuota):
        def search(self, **_):
            raise RuntimeError("connection reset")

    caplog.clear()
    monkeypatch.setattr(market_news, "TavilyClient", _Broken)
    with caplog.at_level(logging.INFO, logger="stock_analyzer.data.market_news"):
        market_news._tavily_market_news(max_results=10)
    assert sum(r.levelno >= logging.WARNING for r in caplog.records) == 3
