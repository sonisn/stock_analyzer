"""Application configuration — single source of truth for env-driven knobs."""

from __future__ import annotations

from typing import Literal

from pydantic import EmailStr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=False,
    )

    # Anthropic / agno
    anthropic_api_key: SecretStr
    anthropic_model: str = "claude-sonnet-4-6"
    anthropic_max_tokens: int = 4096
    anthropic_prompt_caching: bool = True

    # SnapTrade
    snaptrade_client_id: SecretStr
    snaptrade_consumer_key: SecretStr
    snaptrade_user_id: str
    snaptrade_user_secret: SecretStr

    # Finnhub
    finnhub_api_key: SecretStr

    # SMTP (Stalwart)
    smtp_host: str
    smtp_port: int = 587
    smtp_use_tls: bool = True
    smtp_username: str
    smtp_password: SecretStr
    smtp_from_address: EmailStr
    smtp_from_name: str = "Stock Analyzer"
    smtp_to_address: EmailStr

    # Storage
    database_url: SecretStr
    failed_emails_dir: str = "/var/lib/stock-analyzer/failed_emails"

    # Behavior
    run_timezone: str = "America/New_York"
    drawdown_threshold_pct: float = 5.0
    politician_lookback_months: int = 24
    politician_fresh_disclosure_days: int = 2
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "pretty"] = "json"

    # Feature flags
    dry_run: bool = False
    skip_nyse_holidays: bool = True
    stock_analyzer_env: Literal["production", "development"] = "production"
