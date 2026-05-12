"""Centralized settings — single source of truth for env-driven values.

Built on `pydantic-settings.BaseSettings` so every field is:
  - typed (no manual `int(...)` / `float(...)` casts at the boundary)
  - validated (bad provider/aggressiveness fails fast at startup)
  - env-name-mapped automatically (field `anthropic_api_key`
    binds to env `ANTHROPIC_API_KEY`)

Override behavior with env vars or a `.env` file at the project root.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from .llm import Provider

Aggressiveness = Literal["conservative", "balanced", "aggressive"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # treat `FOO=` as unset (use the default)
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # ---- LLM provider keys ------------------------------------------------
    anthropic_api_key: str | None = None
    # GOOGLE_API_KEY is preferred; fall back to GEMINI_API_KEY for back-compat.
    google_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    )

    # ---- LLM selection ----------------------------------------------------
    # `llm_provider` + `llm_model` are the defaults used by every agent.
    # Override any single role via the `*_provider`/`*_model` vars below;
    # unset roles fall back to the defaults.
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

    # ---- Data sources -----------------------------------------------------
    tavily_api_key: str | None = None
    snaptrade_client_id: str | None = None
    snaptrade_consumer_key: str | None = None
    snaptrade_user_id: str | None = None
    snaptrade_user_secret: str | None = None
    chart_img_api_key: str | None = None
    fred_api_key: str | None = None
    finnhub_api_key: str | None = None

    # ---- SMTP -------------------------------------------------------------
    smtp_host: str | None = None
    smtp_port: int | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str | None = None
    email_to: str | None = None

    # ---- Discover pipeline (cli/discover.py) ------------------------------
    # Models per stage — Opus for big-stakes reasoning, Sonnet for
    # per-candidate analysis, Haiku for any cheap data prep.
    discover_opus_model: str = "claude-opus-4-7"
    discover_sonnet_model: str = "claude-sonnet-4-6"
    # `NoDecode` tells pydantic-settings to skip its default JSON parse for
    # complex types so the raw "AAPL,NVDA" string reaches our validator below.
    discover_watchlist: Annotated[tuple[str, ...], NoDecode] = ()
    discover_cash_budget: float | None = None
    discover_db_path: str = "~/.stock_analyzer/discover.db"
    # Run the ranker N times and consensus-vote on picks (N >= 2). With
    # temperature=0 set in all stages, runs should agree near-perfectly,
    # but this catches any residual variance from data freshness.
    discover_consensus_runs: int = 3
    # Rebalance aggressiveness:
    #   conservative — strict tax-after-EV bar (10%), forward deterioration
    #                  required for any SELL/TRIM
    #   balanced     — risk-reduction trims allowed on overbought + above-
    #                  target positions even with short-term tax cost (5% bar)
    #   aggressive   — tax-aware but not tax-blocked; recommend churn where
    #                  forward signal is meaningfully better (0% bar)
    # The report ALWAYS includes a "tax-agnostic alternative" section so the
    # user sees the opportunity cost regardless of which mode is selected.
    discover_rebalance_aggressiveness: Aggressiveness = "balanced"

    # ---- Behavior ---------------------------------------------------------
    use_cached_analysis: bool = True
    insider_lookback_days: int = 5

    # ---- Coercers ---------------------------------------------------------
    @field_validator("discover_watchlist", mode="before")
    @classmethod
    def _split_watchlist(cls, v: object) -> object:
        # Env vars arrive as comma-separated strings: "AAPL,NVDA,googl".
        # Already-tuple/list values pass through untouched.
        if isinstance(v, str):
            return tuple(t.strip().upper() for t in v.split(",") if t.strip())
        return v

    @field_validator("discover_rebalance_aggressiveness", mode="before")
    @classmethod
    def _lower_aggressiveness(cls, v: object) -> object:
        # Accept "Balanced", " AGGRESSIVE " etc.; normalize so the Literal
        # check below succeeds without surprising the user.
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @classmethod
    def from_env(cls) -> "Settings":
        # Back-compat shim — `Settings()` already loads from env. Existing
        # callers (`Settings.from_env()`) keep working without churn.
        return cls()
