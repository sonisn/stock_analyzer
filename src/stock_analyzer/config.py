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
    discover_opus_model: str = "claude-opus-5-5"
    discover_sonnet_model: str = "claude-sonnet-5"
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
    # Price rules of the discover screen: "soft" (default: only skip names
    # 40%+ below their 52-week high — long-term holds may be bought on a
    # dip), "strict" (the old four uptrend rules), or "off".
    discover_trend_gate: Literal["strict", "soft", "off"] = "soft"
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
    # Per-run cap on estimated model spend (USD). Unset = no cap. Stages
    # with cheaper options plan to fit (Analyst tier/count, extra Ranker
    # rounds, Reviewer model) and any single call that would pass the cap is
    # refused. Only priced models count: Claude is built in; add others via
    # LLM_PRICES, e.g. "gemini-pro-latest=1.25:10,gpt-6-astra=2:8" (USD per
    # million input:output tokens).
    discover_max_cost_usd: float | None = None
    llm_prices: str = ""
    # Per-run history upkeep (db/retention.py), the last step of every run:
    # adds backfilled pick fields + survivor outcome labels, then trims LLM
    # prose older than the text window, agno step logs, failed-screen
    # candidates past the longest lookback any check uses, old model
    # versions, and stale log / price-cache files. Analysis rows the track
    # record, calibration and model read are never deleted.
    # Deterministic "Portfolio health" block at the top of the daily
    # analyze-portfolio email (reporting/health.py): stop-loss watch, thesis
    # check on held former picks, sector weight vs the cap, tax-loss
    # harvesting candidates, earnings this week. No LLM calls.
    portfolio_health: bool = True
    # Whether the daily email proposes stocks you do NOT own: the "Ideas
    # for new money" blocks and the "reinvest the proceeds in X" tail on
    # a sale line. Off leaves every action on a current holding intact —
    # sells, drawdown re-checks, tax-loss candidates, covered calls — and
    # stops the email from sourcing new names out of an ageing pick pool.
    # Turn it off when the picks are stale relative to how you are now
    # investing; `discover-stocks` is where fresh ones come from.
    daily_email_new_ideas: bool = True
    history_upkeep: bool = True
    history_text_retention_days: int = 365
    history_session_retention_days: int = 30
    history_candidate_retention_days: int = 540
    history_keep_model_versions: int = 12
    history_file_retention_days: int = 30
    # Per-stock reference rows (sector, earnings date) untouched this long go.
    history_reference_retention_days: int = 365
    # Compact the database file once trimming has freed this share of it.
    history_vacuum_min_free_pct: float = 20.0
    # `ops backup` (nightly from cron) writes consistent copies of the
    # database here and keeps the newest `backup_keep`. Keep it on a
    # different disk from `discover_db_path`.
    backup_dir: str = "~/.stock_analyzer/backups"
    backup_keep: int = 14
    # `ops backup` also deletes log files older than this many days from
    # LOG_DIR and the repo's cron `logs/` (every process writes its own
    # file, so they pile up by the hundreds). 0 keeps them all.
    log_keep_days: int = 90
    # The monthly review flags the database when it passes this size.
    history_db_warn_mb: float = 50.0
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
    # Daily email: reuse each stock's long-term view until it is this old,
    # the price moves this much, or the company reports (agents/stock_views.py).
    stock_view_max_age_days: int = 7
    stock_view_move_pct: float = 8.0
    use_cached_analysis: bool = True
    insider_lookback_days: int = 5

    # ---- Covered-call writing (cli/rebalance.py extension) ---------------
    cc_enabled: bool = True
    # Written to be kept, not to be exercised. A 0.40-delta call is
    # roughly a 40% chance of losing the shares; these are 3-5 year
    # holdings, so the band sits far out of the money and the premium is
    # whatever that is worth.
    cc_target_delta_min: float = 0.10
    cc_target_delta_max: float = 0.25
    # Longer expiries collect more total premium per contract. They also
    # collect LESS per day (theta is slowest far out) and lock the cap in
    # for longer, so the report shows premium per day beside the total.
    cc_dte_min: int = 60
    cc_dte_max: int = 120
    # A hard floor under the strike, independent of delta: never write a
    # call that caps the position less than this far above today's price,
    # however rich the premium looks.
    cc_min_upside_pct: float = 15.0
    # Only write when the options market is paying more than the stock's
    # own realized volatility (IV/HV >= this). Below it the premium is
    # cheap and the right move is to wait, not to sell the upside
    # anyway. 1.20+ is "elevated" in `_label_iv_hv_ratio`.
    cc_min_iv_hv_ratio: float = 1.0
    cc_min_premium_usd: float = 500.0
    cc_slippage_buffer: float = 0.10
    cc_stub_optimization: bool = True
    cc_min_stub_usd: float = 1000.0
    # Tickers never to write options on — covered calls or cash-secured
    # puts. CC_DENYLIST is the old name, still honored.
    options_denylist: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(),
        validation_alias=AliasChoices("OPTIONS_DENYLIST", "CC_DENYLIST"),
    )

    # ---- Cash-secured puts (front half of the wheel, cli/rebalance.py) ---
    # Sell puts on recent discover picks you don't own yet: paid to wait,
    # assigned only at a price below today's. Premium-harvest posture —
    # low delta, so assignment is the exception.
    csp_enabled: bool = True
    csp_target_delta_min: float = 0.10
    csp_target_delta_max: float = 0.25
    csp_dte_min: int = 30
    csp_dte_max: int = 45
    # Collateral caps, as fractions of available cash: per put, and all
    # puts together (the rest stays free for BUYs and dry powder).
    csp_max_pct_per_put: float = 0.25
    csp_max_pct_total: float = 0.80
    # Only sell a put when its implied vol is at least this multiple of the
    # stock's realized vol — the same floor CC_MIN_IV_HV_RATIO applies to
    # written calls. Without it the put side would offer premium priced
    # below what the underlying actually moves, which is selling insurance
    # under cost; on a put that matters more than on a call, because the
    # premium is the entire return. A missing vol reading keeps the
    # candidate: an unknown IV is not evidence of a cheap one.
    csp_min_iv_hv_ratio: float = 1.0
    # Rewrite the static dashboard at the end of a run, so the page and the
    # email never disagree about what was decided. Off makes the page
    # change only on its own schedule.
    dashboard_after_run: bool = True
    dashboard_path: str = "~/.stock_analyzer/reports/dashboard.html"
    # How many past runs' picks are put candidates (plus this run's).
    csp_pick_lookback_runs: int = 3
    # Accounts approved to sell puts (comma-separated labels as the
    # reports show them). Empty = every account. HSAs and some IRAs can't.
    options_accounts: Annotated[tuple[str, ...], NoDecode] = ()

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

    @field_validator("options_accounts", mode="before")
    @classmethod
    def _split_options_accounts(cls, v: object) -> object:
        if isinstance(v, str):
            return tuple(t.strip() for t in v.split(",") if t.strip())
        return v

    @field_validator("options_denylist", mode="before")
    @classmethod
    def _split_options_denylist(cls, v: object) -> object:
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
