"""Centralized settings — single source of truth for env-driven values."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # LLM provider keys
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

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
