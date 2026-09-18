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
    openai_api_key: str | None = None

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
    # Two-tier per-candidate analysis: the top `discover_analyst_deep_count`
    # survivors by screen score go to the Sonnet model, the rest to this
    # cheaper model (retried on Sonnet if it fails). A count of 0 or a blank
    # model sends every survivor to Sonnet.
    discover_haiku_model: str = "claude-haiku-4-5"
    discover_analyst_deep_count: int = 10
    # `NoDecode` tells pydantic-settings to skip its default JSON parse for
    # complex types so the raw "AAPL,NVDA" string reaches our validator below.
    discover_watchlist: Annotated[tuple[str, ...], NoDecode] = ()
    discover_cash_budget: float | None = None
    discover_db_path: str = "~/.stock_analyzer/discover.db"
    # Ceiling on how many names reach the per-ticker fundamentals + EPS
    # fetches (~3 Yahoo requests each). The screen's trend rules are
    # applied first, from technicals alone, and usually narrow a ~500-name
    # frame well below this; the cap is the backstop for a broad tape when
    # most of the index is in an uptrend. Survivors are kept by 6-month
    # relative strength, and holdings/watchlist names are never capped out.
    discover_max_screen_candidates: int = 250
    # Lookback window for the per-ticker Tavily news that grounds the
    # Analyst/Reviewer's upcoming-catalyst extraction (one Tavily search
    # per survivor/holding per run).
    discover_catalyst_news_days: int = 30
    # A new pick that reports earnings within the alert window (5 days) is
    # capped at this % of new capital; the rest waits for the print.
    discover_earnings_blackout_max_pct: float = 5.0
    # Hard sector caps applied after the Sizer responds. With a cash budget,
    # no sector may exceed `discover_max_sector_pct` of the combined book
    # (current holdings + new money). Always, no sector may take more than
    # `discover_max_sector_new_pct` of the new capital. Trimmed dollars are
    # held as cash rather than redistributed, like the other caps.
    discover_max_sector_pct: float = 30.0
    discover_max_sector_new_pct: float = 50.0
    # Tax-loss harvesting candidates in the rebalance report: a taxable
    # position slice qualifies when its unrealized loss is at least this
    # many dollars AND this many percent below cost basis.
    harvest_min_loss_usd: float = 1000.0
    harvest_min_loss_pct: float = 10.0
    # Where `train-model` caches the multi-year price panel it trains on.
    model_cache_dir: str = "~/.stock_analyzer/cache"
    # Ranker consensus: one round per (provider, model) pair listed here,
    # each a full high-effort ranking pass; picks are kept if a majority of
    # rounds agree. A blank model in `discover_ranker_models` (or too few
    # entries) falls back to that provider's default model below — see
    # `resolve_ranker_rounds()`. Cross-provider rounds disagree because the
    # providers are genuinely different models, not just different samples
    # of one model's stochasticity the way same-provider N-of-N resampling
    # used to.
    discover_ranker_providers: str = "claude,gemini,openai"
    discover_ranker_models: str = ""
    discover_gemini_model: str = "gemini-pro-latest"
    discover_openai_model: str = "gpt-6-astra"
    # Red-team critiques the ranker's picks from outside whatever blind
    # spots the ranker's own provider(s) might share — default to a
    # different provider than the primary Claude pipeline for that reason.
    discover_redteam_provider: Provider = "gemini"
    discover_redteam_model: str = ""
    # If a stage's primary provider call fails with an auth/rate-limit/
    # provider error, retry once on this provider instead of failing the
    # whole run.
    discover_fallback_provider: Provider = "claude"
    discover_fallback_model: str = ""
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

    # ---- Covered-call writing (cli/rebalance.py extension) ---------------
    cc_enabled: bool = True
    cc_target_delta_min: float = 0.35
    cc_target_delta_max: float = 0.45
    cc_dte_min: int = 30
    cc_dte_max: int = 45
    cc_denylist: Annotated[tuple[str, ...], NoDecode] = ()
    cc_min_premium_usd: float = 500.0
    cc_slippage_buffer: float = 0.10
    cc_stub_optimization: bool = True
    cc_min_stub_usd: float = 1000.0

    # ---- Tradier options data (primary chain provider) -------------------
    tradier_api_key: str | None = None
    tradier_base_url: str = "https://api.tradier.com/v1"

    # ---- Coercers ---------------------------------------------------------
    @field_validator("discover_watchlist", mode="before")
    @classmethod
    def _split_watchlist(cls, v: object) -> object:
        # Env vars arrive as comma-separated strings: "AAPL,NVDA,googl".
        # Already-tuple/list values pass through untouched.
        if isinstance(v, str):
            return tuple(t.strip().upper() for t in v.split(",") if t.strip())
        return v

    @field_validator("cc_denylist", mode="before")
    @classmethod
    def _split_cc_denylist(cls, v: object) -> object:
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
    def from_env(cls) -> Settings:
        # Back-compat shim — `Settings()` already loads from env. Existing
        # callers (`Settings.from_env()`) keep working without churn.
        return cls()

    # ---- Provider/model resolution -----------------------------------------

    def _default_model_for(self, provider: Provider) -> str:
        if provider == "claude":
            return self.discover_opus_model
        if provider == "gemini":
            return self.discover_gemini_model
        if provider == "openai":
            return self.discover_openai_model
        raise ValueError(f"Unsupported provider {provider!r}.")

    def resolve_ranker_rounds(self) -> list[tuple[Provider, str]]:
        """One (provider, model) pair per Ranker consensus round.

        `discover_ranker_providers` is a CSV list of providers, one round
        each; `discover_ranker_models` is the matching CSV of models,
        positionally paired. A blank or missing model entry falls back to
        that provider's default model (`discover_opus_model` for claude,
        `discover_gemini_model`/`discover_openai_model` for the others).
        """
        providers = [p.strip() for p in self.discover_ranker_providers.split(",") if p.strip()]
        models = (
            [m.strip() for m in self.discover_ranker_models.split(",")]
            if self.discover_ranker_models
            else []
        )
        rounds: list[tuple[Provider, str]] = []
        for i, provider in enumerate(providers):
            if provider not in ("claude", "gemini", "openai"):
                raise ValueError(f"Unsupported provider {provider!r} in DISCOVER_RANKER_PROVIDERS.")
            model = models[i].strip() if i < len(models) and models[i].strip() else ""
            rounds.append((provider, model or self._default_model_for(provider)))  # type: ignore[arg-type]
        return rounds

    def resolve_redteam_model(self) -> str:
        return self.discover_redteam_model or self._default_model_for(
            self.discover_redteam_provider
        )

    def resolve_fallback_model(self) -> str:
        return self.discover_fallback_model or self._default_model_for(
            self.discover_fallback_provider
        )
