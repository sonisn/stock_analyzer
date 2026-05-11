"""Centralized settings — single source of truth for env-driven values."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

from .llm import Provider

_VALID_PROVIDERS = ("claude", "gemini")


def _provider(name: str, value: str | None) -> Provider | None:
    if not value:
        return None
    if value not in _VALID_PROVIDERS:
        raise ValueError(
            f"{name}={value!r} — expected one of {_VALID_PROVIDERS}"
        )
    return cast(Provider, value)


@dataclass(frozen=True)
class Settings:
    # LLM provider keys
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # LLM selection — `llm_provider` + `llm_model` are the defaults used by
    # every agent. Override any single role via the `*_provider`/`*_model`
    # vars below; unset ones fall back to the defaults.
    llm_provider: Provider = "claude"
    llm_model: str = "claude-haiku-4-5"
    sentiment_provider: Provider | None = None
    sentiment_model: str | None = None
    ticker_provider: Provider | None = None
    ticker_model: str | None = None
    rerank_provider: Provider | None = None
    rerank_model: str | None = None
    insider_provider: Provider | None = None
    insider_model: str | None = None

    # Data sources
    tavily_api_key: str | None = None
    snaptrade_client_id: str | None = None
    snaptrade_consumer_key: str | None = None
    snaptrade_user_id: str | None = None
    snaptrade_user_secret: str | None = None
    chart_img_api_key: str | None = None

    # SMTP
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    email_to: str | None = None

    # Behavior
    use_cached_analysis: bool = True
    insider_lookback_days: int = 5

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            llm_provider=_provider("LLM_PROVIDER", os.getenv("LLM_PROVIDER")) or "claude",
            llm_model=os.getenv("LLM_MODEL", "claude-haiku-4-5"),
            sentiment_provider=_provider("SENTIMENT_PROVIDER", os.getenv("SENTIMENT_PROVIDER")),
            sentiment_model=os.getenv("SENTIMENT_MODEL") or None,
            ticker_provider=_provider("TICKER_PROVIDER", os.getenv("TICKER_PROVIDER")),
            ticker_model=os.getenv("TICKER_MODEL") or None,
            rerank_provider=_provider("RERANK_PROVIDER", os.getenv("RERANK_PROVIDER")),
            rerank_model=os.getenv("RERANK_MODEL") or None,
            insider_provider=_provider("INSIDER_PROVIDER", os.getenv("INSIDER_PROVIDER")),
            insider_model=os.getenv("INSIDER_MODEL") or None,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            snaptrade_client_id=os.getenv("SNAPTRADE_CLIENT_ID"),
            snaptrade_consumer_key=os.getenv("SNAPTRADE_CONSUMER_KEY"),
            snaptrade_user_id=os.getenv("SNAPTRADE_USER_ID"),
            snaptrade_user_secret=os.getenv("SNAPTRADE_USER_SECRET"),
            chart_img_api_key=os.getenv("CHART_IMG_API_KEY"),
            smtp_host=os.getenv("SMTP_HOST"),
            smtp_port=int(os.getenv("SMTP_PORT")) if os.getenv("SMTP_PORT") else None,
            smtp_user=os.getenv("SMTP_USER"),
            smtp_password=os.getenv("SMTP_PASSWORD"),
            smtp_from=os.getenv("SMTP_FROM"),
            email_to=os.getenv("EMAIL_TO"),
            use_cached_analysis=os.getenv("USE_CACHED_ANALYSIS", "1") == "1",
            insider_lookback_days=int(os.getenv("INSIDER_LOOKBACK_DAYS", "5")),
        )
