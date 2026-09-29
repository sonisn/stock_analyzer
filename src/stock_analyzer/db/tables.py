"""SQLModel table classes mapped 1:1 to the existing SQLite schema.

JSON-blob columns (fail_reasons, score_components, score_breakdown,
sources, dashboard_data) stay as Optional[str] here; repository
functions own the json.dumps/json.loads boundary. This keeps the
on-disk format byte-identical to the legacy raw-sqlite schema.

Composite primary keys use multiple Field(primary_key=True) entries.
Foreign keys preserve ON DELETE CASCADE via the ondelete arg.
"""

from __future__ import annotations

from sqlalchemy import UniqueConstraint
from sqlmodel import Field, SQLModel


class Run(SQLModel, table=True):
    __tablename__ = "runs"

    id: int | None = Field(default=None, primary_key=True)
    run_at: str
    kind: str = Field(default="discover")
    universe_size: int
    survivors: int
    picks: int
    opus_model: str | None = None
    sonnet_model: str | None = None
    cash_budget: float | None = None


class Candidate(SQLModel, table=True):
    __tablename__ = "candidates"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    passed_filter: int
    fail_reasons: str | None = None  # JSON list
    score: float | None = None
    score_components: str | None = None  # JSON
    score_breakdown: str | None = None  # JSON
    sources: str | None = None  # JSON list
    conviction: int | None = None
    sector: str | None = None
    price: float | None = None


class Scorecard(SQLModel, table=True):
    __tablename__ = "scorecards"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    analyst_text: str | None = None


class Pick(SQLModel, table=True):
    """One of a discover run's top picks.

    The forecast columns below (`conviction`, `ev_pct`, `entry_price`) and
    the `PickScenario` rows exist so the ranker's own calibration can be
    measured. The ranker prompt tells Opus it will be "measured on the EV
    vs realized return"; until these were persisted that promise could not
    be kept, because only the prose rendering reached disk and the
    probabilities were recomputed for the report and then discarded.

    `entry_price` is the screen-time price, stored rather than refetched so
    calibration never reprices a historical pick with today's data.
    """

    __tablename__ = "picks"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    rank: int = Field(primary_key=True)
    ticker: str
    ranker_text: str
    bear_case_text: str | None = None
    allocation_text: str | None = None
    # --- forecast, for calibration scoring ---
    conviction: int | None = None
    ev_pct: float | None = None  # Σ(probability × target_return_pct)
    entry_price: float | None = None
    time_horizon: str | None = None
    # --- ranker consensus provenance, for track-record-by-provider ---
    # Fraction of consensus rounds that picked this ticker (e.g. 2/3 ->
    # 0.667). NULL for single-round runs, which have no agreement signal.
    agreement_ratio: float | None = None
    # Comma-joined provider names (e.g. "claude,openai") whose consensus
    # round picked this ticker. NULL for single-round runs.
    voting_providers: str | None = None


class PickScenario(SQLModel, table=True):
    """One bull/base/bear scenario behind a pick's expected return.

    Stored per scenario rather than as a JSON blob on `picks` so a
    reliability check ("of the picks where you said bear was 15% likely,
    how often did the bear case actually land?") is a plain query.
    """

    __tablename__ = "pick_scenarios"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    rank: int = Field(primary_key=True)
    label: str = Field(primary_key=True)  # bull | base | bear
    ticker: str
    probability: float
    target_return_pct: float


class PickCatalyst(SQLModel, table=True):
    """An upcoming catalyst the Analyst named for a pick, kept so it can be
    graded once its date passes (discover/catalyst_grading.py)."""

    __tablename__ = "pick_catalysts"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    seq: int = Field(primary_key=True)
    event: str
    expected_date: str | None = None  # YYYY-MM-DD
    direction: str  # positive | negative | uncertain
    impact: str  # high | medium | low
    source: str


class HoldingReviewRow(SQLModel, table=True):
    """ORM table for holdings reviews. The `Row` suffix avoids collision
    with `stock_analyzer.models.llm.HoldingReview` (the Pydantic DTO that
    represents the LLM's structured output)."""

    __tablename__ = "holdings_reviews"

    run_id: int = Field(
        foreign_key="runs.id",
        primary_key=True,
        ondelete="CASCADE",
    )
    ticker: str = Field(primary_key=True)
    verdict: str | None = None
    confidence: int | None = None
    review_text: str | None = None


class RunOutput(SQLModel, table=True):
    __tablename__ = "run_outputs"

    run_id: int = Field(
        primary_key=True,
        foreign_key="runs.id",
        ondelete="CASCADE",
    )
    ranker_full: str | None = None
    redteam_full: str | None = None
    sizer_full: str | None = None
    holdings_summary: str | None = None
    rebalance_text: str | None = None
    dashboard_data: str | None = None  # JSON


class CandidateSnapshot(SQLModel, table=True):
    """Point-in-time fundamentals a run saw for one candidate.

    Prices can be refetched for any past date, but yfinance fundamentals are
    a live snapshot with no history — once a run is over, what the screen
    saw is gone. Storing a compact numeric copy lets a future model train
    on fundamentals without leaking today's values into past rows. Only
    names whose fundamentals were actually fetched get a row.
    """

    __tablename__ = "candidate_snapshots"

    run_id: int = Field(foreign_key="runs.id", primary_key=True, ondelete="CASCADE")
    ticker: str = Field(primary_key=True)
    data: str  # JSON {field: number}


class CandidateOutcome(SQLModel, table=True):
    """Realized forward excess return of a screened candidate: entry at the
    first close after the run, exit `horizon_days` trading days later."""

    __tablename__ = "candidate_outcomes"

    run_id: int = Field(foreign_key="runs.id", primary_key=True, ondelete="CASCADE")
    ticker: str = Field(primary_key=True)
    horizon_days: int = Field(primary_key=True)
    entry_date: str
    exit_date: str
    return_pct: float
    spy_return_pct: float
    excess_pct: float


class ModelVersion(SQLModel, table=True):
    """One trained forward-return model and its walk-forward validation.
    Only rows with accepted=1 are ever used by the screen."""

    __tablename__ = "model_versions"

    id: int | None = Field(default=None, primary_key=True)
    created_at: str
    horizon_days: int
    population: str
    features: str  # JSON list
    coefficients: str  # JSON {feature: weight}
    metrics: str  # JSON
    train_start: str
    train_end: str
    accepted: int = 0


class Suggestion(SQLModel, table=True):
    """One piece of advice the user was given — a daily-email decision line
    or a rebalance plan action — kept so the quarterly review can grade it
    against what the stock (and the suggested reinvestment) did next.

    The daily email repeats a standing suggestion every day; one row per
    (day, source, action, ticker) is kept, and the review grades the first.
    """

    __tablename__ = "suggestions"
    __table_args__ = (
        UniqueConstraint("suggested_on", "source", "action", "ticker", name="uq_suggestion_day"),
    )

    id: int | None = Field(default=None, primary_key=True)
    suggested_on: str = Field(index=True)  # ISO date
    source: str  # "daily" | "rebalance"
    action: str  # SELL / TRIM / TAX_LOSS / REVIEW / BUY / ADD / WRITE_CALL / SELL_PUT / STANDOUT / INSIDER_BUYS
    ticker: str = Field(index=True)
    detail: str = ""
    price: float | None = None  # price when suggested, when known
    units_held: float | None = None  # position size then — "did the user act?"
    reinvest_into: str | None = None  # where the proceeds were suggested to go
    run_id: int | None = None  # rebalance run, when from one


class PortfolioSnapshot(SQLModel, table=True):
    """End-of-run value of the whole portfolio, one row per day (the daily
    email writes it). With the brokerage's deposit/withdrawal/transfer
    history this gives a time-weighted return to compare against SPY."""

    __tablename__ = "portfolio_snapshots"

    day: str = Field(primary_key=True)  # ISO date
    holdings_value: float
    cash: float
    total: float
    # JSON {account label: {"value": holdings, "cash": cash}} — which
    # accounts this total covered. Connecting a new brokerage account adds
    # its whole balance between two snapshots; without this the jump reads
    # as a return. NULL on rows written before the column existed.
    accounts: str | None = None


class TickerPrice(SQLModel, table=True):
    """One close per ticker per trading day, kept permanently.

    Every price this system has ever fetched was used once and discarded,
    so "what was AVGO worth on the day we said sell it?" had no answer.
    Grading advice needs exactly that, and re-deriving it later is both a
    network call and a chance to get it wrong. Append-only and tiny — a
    year of sixteen holdings is about four thousand rows.
    """

    __tablename__ = "ticker_prices"

    ticker: str = Field(primary_key=True)
    day: str = Field(primary_key=True)  # ISO trading day
    close: float


class BrokerageActivity(SQLModel, table=True):
    """One brokerage activity (buy, sell, dividend, reinvestment, deposit,
    transfer, option event), kept permanently so tax lots and cash flows
    come from the full history rather than a re-downloaded API window.
    `data` is a compact JSON of only the fields the app reads (~300 bytes)."""

    __tablename__ = "brokerage_activities"

    id: str = Field(primary_key=True)  # SnapTrade activity id (or a content hash)
    account: str = Field(index=True)  # account label (data/brokerage.account_labels)
    trade_date: str = Field(index=True)  # ISO date
    type: str
    data: str


class TickerReference(SQLModel, table=True):
    """Slow-changing facts about a stock, one row per ticker, overwritten on
    refresh (so the table is bounded by the number of stocks ever seen):
    sector/industry/name, and the next earnings date."""

    __tablename__ = "ticker_reference"

    ticker: str = Field(primary_key=True)
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    profile_updated: str | None = None  # ISO date sector/industry were fetched
    next_earnings: str | None = None  # ISO date, None when none is scheduled
    earnings_updated: str | None = None  # ISO date next_earnings was fetched


class EarningsEvent(SQLModel, table=True):
    """One quarterly/annual report worth following (discover/earnings_standouts.py):
    a clear beat anywhere in the market, or any report by a past pick.
    Other reporters are never stored, and rows age out after a year, so
    the table stays small."""

    __tablename__ = "earnings_events"

    ticker: str = Field(primary_key=True)
    report_date: str = Field(primary_key=True)  # ISO date
    hour: str = ""  # "bmo" / "amc" / "" (unknown)
    eps_estimate: float | None = None
    eps_actual: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    # Two-session move around the report minus SPY's, percent.
    reaction_pct: float | None = None
    # Change in next-year consensus EPS over the 30 days after, fraction.
    revision_pct: float | None = None
    # The year before: EPS beats in the prior quarters, and revenue growth
    # over the same quarter a year earlier, percent. Filled at the last gate.
    prior_beats: int | None = None
    prior_quarters: int | None = None
    revenue_yoy_pct: float | None = None
    # "pending" until decided, then "standout" or "no".
    status: str = Field(default="pending", index=True)
    decided_on: str | None = None  # ISO date status left "pending"


class FundPosition(SQLModel, table=True):
    """One stock position of a tracked hedge fund at a quarter end, from its
    13F filing (data/hedge_funds_13f.py). Common stock only (no options),
    the last FUND_KEEP_PERIODS quarters per fund: ~22 funds x ~60 positions."""

    __tablename__ = "fund_positions"

    cik: str = Field(primary_key=True)
    period: str = Field(primary_key=True)  # ISO quarter-end date
    cusip: str = Field(primary_key=True)
    ticker: str | None = Field(default=None, index=True)
    shares: float = 0.0
    value_usd: float = 0.0
    filed: str = ""  # ISO date the 13F was filed
    accession: str = ""


class CusipTicker(SQLModel, table=True):
    """CUSIP -> ticker, from OpenFIGI, cached forever (a CUSIP never moves).
    `ticker` None means OpenFIGI had no US common stock for it."""

    __tablename__ = "cusip_tickers"

    cusip: str = Field(primary_key=True)
    ticker: str | None = None
    checked_on: str = ""


class InsiderBuy(SQLModel, table=True):
    """One open-market purchase (Form 4, code P) by an insider of a watched
    company (data/insider_buying.py). Purchases only — sales are routine —
    and pruned after INSIDER_KEEP_DAYS, so a few hundred rows a year."""

    __tablename__ = "insider_buys"

    ticker: str = Field(primary_key=True)
    filing_id: str = Field(primary_key=True)  # SEC accession number
    name: str = Field(primary_key=True)
    filed: str = Field(index=True)  # ISO date the Form 4 was filed
    traded: str = ""  # ISO date of the purchase
    shares: float = 0.0
    price: float | None = None


class AnalystAction(SQLModel, table=True):
    """One analyst's rating or price-target action on a stock that reported
    a result worth following (data/analyst_actions.py, filled by the nightly
    earnings watch for recorded earnings events): who acted, when, and how.
    Kept from 90 days before the report on; tens of rows per stock."""

    __tablename__ = "analyst_actions"

    ticker: str = Field(primary_key=True)
    graded_at: str = Field(primary_key=True)  # ISO timestamp of the action
    firm: str = Field(primary_key=True)
    action: str = ""  # up / down / init / main (maintained) / reit
    to_grade: str = ""
    from_grade: str = ""
    target_action: str = ""  # Raises / Lowers / Maintains / Announces / ...
    target: float | None = None
    prior_target: float | None = None


class ForecastSnapshot(SQLModel, table=True):
    """What analysts expected of a tracked stock on one day
    (data/forecast_snapshots.py). Yahoo keeps only 90 days of estimate
    history; this keeps all of it, point in time, so a model can one day
    learn from revisions without seeing the future. Append-only, about
    175 rows per weekday (~4 MB a year); never pruned."""

    __tablename__ = "forecast_snapshots"

    ticker: str = Field(primary_key=True)
    day: str = Field(primary_key=True)  # ISO date the snapshot was taken
    price: float | None = None
    eps_current_year: float | None = None  # consensus EPS, current fiscal year
    eps_next_year: float | None = None
    revenue_current_year: float | None = None
    revenue_next_year: float | None = None
    analysts: int | None = None
    target_mean: float | None = None
    target_high: float | None = None
    target_low: float | None = None
    recommendation_mean: float | None = None  # 1 strong buy .. 5 sell
    # From the same quote summary, no extra request. Free short-interest
    # history is patchy, so this is where ours starts (FINRA's twice-monthly
    # figure, as Yahoo shows it).
    short_pct_float: float | None = None  # fraction of the float sold short
    short_ratio: float | None = None  # days to cover at average volume
    shares_outstanding: float | None = None
    institutions_pct: float | None = None  # fraction held by institutions
    insiders_pct: float | None = None


class StockView(SQLModel, table=True):
    """The daily email's latest long-term view per stock, reused until
    something changes (see agents/stock_views.py). One row per ticker,
    overwritten; rows for stocks no longer held age out."""

    __tablename__ = "stock_views"

    ticker: str = Field(primary_key=True)
    written_on: str  # ISO date
    price: float | None = None  # price when written
    view: str
    # Headline links already shown for this stock, newest first, JSON. Keeps
    # a reused view from repeating the same five headlines all week. Capped,
    # so the row never grows (see agents/stock_views.py).
    shown_links: str | None = None
    news_on: str | None = None  # ISO date the links were last written


class IbdRating(SQLModel, table=True):
    """This morning's IBD-style ratings (discover/ibd_ratings.py), one row
    per stock in the $2B+ universe. Replaced whole each morning, so the
    table stays ~1,900 rows."""

    __tablename__ = "ibd_ratings"

    ticker: str = Field(primary_key=True)
    as_of: str = ""  # ISO date of the last close rated
    price: float | None = None
    industry: str | None = None
    composite: int | None = None
    rs_rating: int | None = None
    eps_rating: int | None = None
    ad_grade: str | None = None
    group_rank: int | None = None
    groups_ranked: int | None = None
    rs_line_high: bool | None = None
    off_high: float | None = None  # fraction below the 52-week high
    six_month: float | None = None
    eps_q1: float | None = None  # latest quarter's EPS growth, fraction
    eps_source: str | None = None  # "sec" or "yahoo" (reported, adjusted EPS)
    base: str | None = None
    base_weeks: int | None = None
    base_depth: float | None = None
    pivot: float | None = None
    vs_pivot: float | None = None
    base_status: str | None = None


class IbdMarket(SQLModel, table=True):
    """Market direction each morning (distribution and follow-through days
    on SPY and QQQ). One row a day, a few hundred bytes."""

    __tablename__ = "ibd_market"

    day: str = Field(primary_key=True)  # ISO date of the last close
    status: str = ""
    detail: str = ""
    indexes: str = ""  # JSON: per-index counts


class IbdHistory(SQLModel, table=True):
    """IBD-style ratings over time, so a stock's progress (and the ratings'
    worth) can be compared. Daily for the names that matter (Composite 90+,
    in a buy zone, held or tracked); every stock once a week (`full`), which
    is the point-in-time panel a study of the ratings needs. ~160k small
    rows a year."""

    __tablename__ = "ibd_history"

    day: str = Field(primary_key=True)  # ISO date of the close rated
    ticker: str = Field(primary_key=True)
    full: bool = False  # part of the weekly all-stocks snapshot
    backfilled: bool = False  # rebuilt from stored bars, not recorded that morning
    price: float | None = None
    composite: int | None = None
    rs_rating: int | None = None
    eps_rating: int | None = None
    ad_grade: str | None = None
    group_rank: int | None = None
    base_status: str | None = None
    pivot: float | None = None


class IbdSignal(SQLModel, table=True):
    """A stock entering a buy zone or breaking out with Composite 80+ — the
    IBD-style "buy" moment, logged once per 30 days per stock and graded
    later against SPY (reporting/leaders.signal_scorecard)."""

    __tablename__ = "ibd_signals"

    ticker: str = Field(primary_key=True)
    day: str = Field(primary_key=True)  # ISO date of the close it fired on
    status: str = ""  # "buy zone" or "breakout"
    price: float | None = None
    pivot: float | None = None
    composite: int | None = None
    rs_rating: int | None = None
    eps_rating: int | None = None
    base: str | None = None
    industry: str | None = None
    backfilled: bool = False  # rebuilt from stored bars, not recorded that morning


class IbdSector(SQLModel, table=True):
    """Each sector's direction each morning (discover/ibd_ratings.
    sector_direction): Leading, Uptrend, Caution or Correction, and why.
    Twelve rows a day."""

    __tablename__ = "ibd_sectors"

    day: str = Field(primary_key=True)
    sector: str = Field(primary_key=True)
    status: str = ""
    reasons: str = ""  # "; "-joined
    rank: int | None = None
    stocks: int = 0
    breadth: float | None = None  # share of its stocks above their 50-day
    breadth_before: float | None = None  # the same, ten sessions earlier
    median_six: float | None = None
    leaders: int = 0  # Composite 90+
    etf: str | None = None
    etf_above_50: bool | None = None
    etf_above_200: bool | None = None
    etf_dist_days: int | None = None
    backfilled: bool = False


class StockNews(SQLModel, table=True):
    """The top news items per featured stock for the dashboard (cli/ibd.py,
    each morning): company-specific, ranked by materiality. Replaced whole
    every morning, so the table stays a few hundred rows."""

    __tablename__ = "stock_news"

    ticker: str = Field(primary_key=True)
    rank: int = Field(primary_key=True)
    fetched: str = ""  # ISO date
    title: str = ""
    url: str | None = None
    source: str | None = None
    published: str | None = None
    snippet: str | None = None


class OpenRouterSpend(SQLModel, table=True):
    """What OpenRouter billed, per day, model and stage — the ledger the
    daily cap is checked against (openrouter.py). A few rows a day."""

    __tablename__ = "openrouter_spend"

    day: str = Field(primary_key=True)  # ISO date, UTC (OpenRouter bills in UTC)
    model: str = Field(primary_key=True)
    stage: str = Field(primary_key=True)
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class FilingFacts(SQLModel, table=True):
    """Structured facts read out of a 10-Q/10-K/20-F by an open model
    (agents/filing_reader.py), each with a supporting quote checked against
    the filing text. Tier "A" (stocks acted on) keeps its two newest
    filings, so a quarter can be compared with the one before; tier "B"
    (the rest of the universe) keeps one. ~5 KB a row."""

    __tablename__ = "filing_facts"

    accession: str = Field(primary_key=True)
    ticker: str = Field(index=True)
    tier: str = "B"
    form: str = ""
    filed_on: str = ""  # ISO date
    period_end: str | None = None  # ISO date the filing reports on
    url: str = ""
    read_on: str = ""  # ISO date
    reader_model: str = ""
    provider: str | None = None  # the host that served the read
    facts: str = ""  # JSON, the reader's answer
    quotes_checked: int = 0
    quotes_found: int = 0  # found verbatim (normalised) in the filing text
    flagged: bool = False
    flag_reasons: str = ""  # "; "-joined
    # The bulk model whose flagged read this one replaced, if any.
    escalated_from: str | None = None
    cost_usd: float = 0.0
    # 10-K/20-F only: the risk factors against last year's (data/text_change).
    risk_kept: float | None = None  # share of sentences carried over verbatim
    risk_cosine: float | None = None


class EightKAlert(SQLModel, table=True):
    """A material 8-K on a held stock, read and emailed the evening it was
    found (reporting/filing_alert.py) — also what stops it being emailed
    twice — or the latest earnings release of a stock acted on, read by
    the weekly run for its guidance (never emailed). A few hundred small
    rows a year."""

    __tablename__ = "eightk_alerts"

    accession: str = Field(primary_key=True)
    ticker: str = Field(index=True)
    filed_on: str = ""
    items: str = ""  # comma-joined 8-K item codes
    url: str = ""
    read_on: str = ""
    reader_model: str = ""
    summary: str = ""  # JSON
    quotes_checked: int = 0
    quotes_found: int = 0
    cost_usd: float = 0.0


class SecEvent(SQLModel, table=True):
    """An SEC event filing on a stock we follow (data/sec_events): a late-
    filing notice, a shelf or offering, a planned insider sale (Form 144),
    or a Schedule 13D. Holdings' events are emailed the evening they are
    found; 13Ds across the universe feed discover as an idea source. Also
    what stops a filing being read or emailed twice. Small rows; a few
    thousand a year, mostly Form 144s below the alert threshold."""

    __tablename__ = "sec_events"

    accession: str = Field(primary_key=True)
    ticker: str = Field(index=True)
    form: str = ""
    kind: str = Field(default="", index=True)  # late_filing|shelf|offering|planned_sale|activist
    filed_on: str = Field(default="", index=True)
    url: str = ""
    read_on: str = ""
    reader_model: str = ""
    facts: str = ""  # JSON: parsed fields plus the reader's answer
    alerted: bool = False  # met the alert bar (a Form 144 below it is only recorded)


class OpenRouterHostCheck(SQLModel, table=True):
    """A known-answer check of one OpenRouter host serving one model
    (openrouter_hosts.canary), run before each weekly filing read. A host
    that failed its latest check is skipped until it passes one."""

    __tablename__ = "openrouter_host_checks"

    day: str = Field(primary_key=True)  # ISO date
    model: str = Field(primary_key=True)
    host: str = Field(primary_key=True)  # OpenRouter slug, e.g. "io-net"
    passed: bool = False
    detail: str = ""
    cost_usd: float = 0.0


class FilingSpotCheck(SQLModel, table=True):
    """Claude re-reading a filing an open model read, field by field
    (cli/filings.py --spot-check): the running measure of the open readers'
    quality, per model and host. Run by hand only (not scheduled)."""

    __tablename__ = "filing_spot_checks"

    accession: str = Field(primary_key=True)
    checked_on: str = ""  # ISO date
    ticker: str = Field(default="", index=True)
    reader_model: str = ""
    provider: str | None = None
    claude_model: str = ""
    agreed: int = 0
    compared: int = 0
    fields: str = ""  # JSON {field: agreed}
    cost_usd: float = 0.0


__all__ = [
    "SecEvent",
    "FilingSpotCheck",
    "OpenRouterHostCheck",
    "EightKAlert",
    "FilingFacts",
    "OpenRouterSpend",
    "StockNews",
    "IbdSector",
    "IbdHistory",
    "IbdSignal",
    "IbdMarket",
    "IbdRating",
    "AnalystAction",
    "CusipTicker",
    "FundPosition",
    "InsiderBuy",
    "EarningsEvent",
    "ForecastSnapshot",
    "StockView",
    "BrokerageActivity",
    "TickerReference",
    "PortfolioSnapshot",
    "TickerPrice",
    "Suggestion",
    "Run",
    "Candidate",
    "Scorecard",
    "Pick",
    "PickScenario",
    "PickCatalyst",
    "HoldingReviewRow",
    "RunOutput",
    "CandidateSnapshot",
    "CandidateOutcome",
    "ModelVersion",
]
