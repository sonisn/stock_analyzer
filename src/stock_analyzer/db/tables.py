"""SQLAlchemy 2.x table classes mapped 1:1 to the existing SQLite schema.

JSON-blob columns (fail_reasons, score_components, score_breakdown,
sources, dashboard_data) stay as Optional[str] here; repository
functions own the json.dumps/json.loads boundary. This keeps the
on-disk format byte-identical to the legacy raw-sqlite schema.

Composite primary keys use several mapped_column(primary_key=True) entries.
Foreign keys preserve ON DELETE CASCADE via ForeignKey(..., ondelete=...).
The classes were SQLModel until 2026-10; the schema they map is unchanged.
"""

from __future__ import annotations

from sqlalchemy import Float, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, MappedAsDataclass, mapped_column


class Base(MappedAsDataclass, DeclarativeBase):
    """Every table is a dataclass declared `kw_only=True`: keyword-only
    constructors that apply each column's default at construction, as the
    SQLModel classes did. (Declared per table: type checkers do not carry
    the flag down from a base class.)"""

    # SQLAlchemy 2.1 maps `float` to DOUBLE; the schema has always said FLOAT.
    type_annotation_map = {float: Float}


def column_names(table: type[Base]) -> list[str]:
    """The mapped column attributes of a table class, in order."""
    return [c.key for c in table.__mapper__.column_attrs]


class Run(Base, kw_only=True):
    __tablename__ = "runs"

    id: Mapped[int | None] = mapped_column(default=None, primary_key=True, nullable=False)
    run_at: Mapped[str]
    kind: Mapped[str] = mapped_column(default="discover")
    universe_size: Mapped[int]
    survivors: Mapped[int]
    picks: Mapped[int]
    opus_model: Mapped[str | None] = mapped_column(default=None)
    sonnet_model: Mapped[str | None] = mapped_column(default=None)
    cash_budget: Mapped[float | None] = mapped_column(default=None)


class Candidate(Base, kw_only=True):
    __tablename__ = "candidates"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ticker: Mapped[str] = mapped_column(primary_key=True)
    passed_filter: Mapped[int]
    fail_reasons: Mapped[str | None] = mapped_column(default=None)  # JSON list
    score: Mapped[float | None] = mapped_column(default=None)
    score_components: Mapped[str | None] = mapped_column(default=None)  # JSON
    score_breakdown: Mapped[str | None] = mapped_column(default=None)  # JSON
    sources: Mapped[str | None] = mapped_column(default=None)  # JSON list
    conviction: Mapped[int | None] = mapped_column(default=None)
    sector: Mapped[str | None] = mapped_column(default=None)
    price: Mapped[float | None] = mapped_column(default=None)


class Scorecard(Base, kw_only=True):
    __tablename__ = "scorecards"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ticker: Mapped[str] = mapped_column(primary_key=True)
    analyst_text: Mapped[str | None] = mapped_column(default=None)


class Pick(Base, kw_only=True):
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

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    rank: Mapped[int] = mapped_column(primary_key=True)
    ticker: Mapped[str]
    ranker_text: Mapped[str]
    bear_case_text: Mapped[str | None] = mapped_column(default=None)
    allocation_text: Mapped[str | None] = mapped_column(default=None)
    # --- forecast, for calibration scoring ---
    conviction: Mapped[int | None] = mapped_column(default=None)
    ev_pct: Mapped[float | None] = mapped_column(default=None)  # Σ(probability × target_return_pct)
    entry_price: Mapped[float | None] = mapped_column(default=None)
    time_horizon: Mapped[str | None] = mapped_column(default=None)
    # --- ranker consensus provenance, for track-record-by-provider ---
    # Fraction of consensus rounds that picked this ticker (e.g. 2/3 ->
    # 0.667). NULL for single-round runs, which have no agreement signal.
    agreement_ratio: Mapped[float | None] = mapped_column(default=None)
    # Comma-joined provider names (e.g. "claude,openai") whose consensus
    # round picked this ticker. NULL for single-round runs.
    voting_providers: Mapped[str | None] = mapped_column(default=None)


class PickScenario(Base, kw_only=True):
    """One bull/base/bear scenario behind a pick's expected return.

    Stored per scenario rather than as a JSON blob on `picks` so a
    reliability check ("of the picks where you said bear was 15% likely,
    how often did the bear case actually land?") is a plain query.
    """

    __tablename__ = "pick_scenarios"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    rank: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(primary_key=True)  # bull | base | bear
    ticker: Mapped[str]
    probability: Mapped[float]
    target_return_pct: Mapped[float]


class PickCatalyst(Base, kw_only=True):
    """An upcoming catalyst the Analyst named for a pick, kept so it can be
    graded once its date passes (discover/catalyst_grading.py)."""

    __tablename__ = "pick_catalysts"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ticker: Mapped[str] = mapped_column(primary_key=True)
    seq: Mapped[int] = mapped_column(primary_key=True)
    event: Mapped[str]
    expected_date: Mapped[str | None] = mapped_column(default=None)  # YYYY-MM-DD
    direction: Mapped[str]  # positive | negative | uncertain
    impact: Mapped[str]  # high | medium | low
    source: Mapped[str]


class HoldingReviewRow(Base, kw_only=True):
    """ORM table for holdings reviews. The `Row` suffix avoids collision
    with `stock_analyzer.models.llm.HoldingReview` (the Pydantic DTO that
    represents the LLM's structured output)."""

    __tablename__ = "holdings_reviews"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ticker: Mapped[str] = mapped_column(primary_key=True)
    verdict: Mapped[str | None] = mapped_column(default=None)
    confidence: Mapped[int | None] = mapped_column(default=None)
    review_text: Mapped[str | None] = mapped_column(default=None)


class RunOutput(Base, kw_only=True):
    __tablename__ = "run_outputs"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ranker_full: Mapped[str | None] = mapped_column(default=None)
    redteam_full: Mapped[str | None] = mapped_column(default=None)
    sizer_full: Mapped[str | None] = mapped_column(default=None)
    holdings_summary: Mapped[str | None] = mapped_column(default=None)
    rebalance_text: Mapped[str | None] = mapped_column(default=None)
    dashboard_data: Mapped[str | None] = mapped_column(default=None)  # JSON


class CandidateSnapshot(Base, kw_only=True):
    """Point-in-time fundamentals a run saw for one candidate.

    Prices can be refetched for any past date, but yfinance fundamentals are
    a live snapshot with no history — once a run is over, what the screen
    saw is gone. Storing a compact numeric copy lets a future model train
    on fundamentals without leaking today's values into past rows. Only
    names whose fundamentals were actually fetched get a row.
    """

    __tablename__ = "candidate_snapshots"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ticker: Mapped[str] = mapped_column(primary_key=True)
    data: Mapped[str]  # JSON {field: number}


class CandidateOutcome(Base, kw_only=True):
    """Realized forward excess return of a screened candidate: entry at the
    first close after the run, exit `horizon_days` trading days later."""

    __tablename__ = "candidate_outcomes"

    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    ticker: Mapped[str] = mapped_column(primary_key=True)
    horizon_days: Mapped[int] = mapped_column(primary_key=True)
    entry_date: Mapped[str]
    exit_date: Mapped[str]
    return_pct: Mapped[float]
    spy_return_pct: Mapped[float]
    excess_pct: Mapped[float]


class ModelVersion(Base, kw_only=True):
    """One trained forward-return model and its walk-forward validation.
    Only rows with accepted=1 are ever used by the screen."""

    __tablename__ = "model_versions"

    id: Mapped[int | None] = mapped_column(default=None, primary_key=True, nullable=False)
    created_at: Mapped[str]
    horizon_days: Mapped[int]
    population: Mapped[str]
    features: Mapped[str]  # JSON list
    coefficients: Mapped[str]  # JSON {feature: weight}
    metrics: Mapped[str]  # JSON
    train_start: Mapped[str]
    train_end: Mapped[str]
    accepted: Mapped[int] = mapped_column(default=0)


class Suggestion(Base, kw_only=True):
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

    id: Mapped[int | None] = mapped_column(default=None, primary_key=True, nullable=False)
    suggested_on: Mapped[str] = mapped_column(index=True)  # ISO date
    source: Mapped[str]  # "daily" | "rebalance"
    action: Mapped[
        str
    ]  # SELL / TRIM / TAX_LOSS / REVIEW / BUY / ADD / WRITE_CALL / SELL_PUT / STANDOUT / INSIDER_BUYS
    ticker: Mapped[str] = mapped_column(index=True)
    detail: Mapped[str] = mapped_column(default="")
    price: Mapped[float | None] = mapped_column(default=None)  # price when suggested, when known
    units_held: Mapped[float | None] = mapped_column(
        default=None
    )  # position size then — "did the user act?"
    reinvest_into: Mapped[str | None] = mapped_column(
        default=None
    )  # where the proceeds were suggested to go
    run_id: Mapped[int | None] = mapped_column(default=None)  # rebalance run, when from one


class PortfolioSnapshot(Base, kw_only=True):
    """End-of-run value of the whole portfolio, one row per day (the daily
    email writes it). With the brokerage's deposit/withdrawal/transfer
    history this gives a time-weighted return to compare against SPY."""

    __tablename__ = "portfolio_snapshots"

    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date
    holdings_value: Mapped[float]
    cash: Mapped[float]
    total: Mapped[float]
    # JSON {account label: {"value": holdings, "cash": cash}} — which
    # accounts this total covered. Connecting a new brokerage account adds
    # its whole balance between two snapshots; without this the jump reads
    # as a return. NULL on rows written before the column existed.
    accounts: Mapped[str | None] = mapped_column(default=None)


class TickerPrice(Base, kw_only=True):
    """One close per ticker per trading day, kept permanently.

    Every price this system has ever fetched was used once and discarded,
    so "what was AVGO worth on the day we said sell it?" had no answer.
    Grading advice needs exactly that, and re-deriving it later is both a
    network call and a chance to get it wrong. Append-only and tiny — a
    year of sixteen holdings is about four thousand rows.
    """

    __tablename__ = "ticker_prices"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    day: Mapped[str] = mapped_column(primary_key=True)  # ISO trading day
    close: Mapped[float]


class BrokerageActivity(Base, kw_only=True):
    """One brokerage activity (buy, sell, dividend, reinvestment, deposit,
    transfer, option event), kept permanently so tax lots and cash flows
    come from the full history rather than a re-downloaded API window.
    `data` is a compact JSON of only the fields the app reads (~300 bytes)."""

    __tablename__ = "brokerage_activities"

    id: Mapped[str] = mapped_column(primary_key=True)  # SnapTrade activity id (or a content hash)
    account: Mapped[str] = mapped_column(
        index=True
    )  # account label (data/brokerage.account_labels)
    trade_date: Mapped[str] = mapped_column(index=True)  # ISO date
    type: Mapped[str]
    data: Mapped[str]


class TickerReference(Base, kw_only=True):
    """Slow-changing facts about a stock, one row per ticker, overwritten on
    refresh (so the table is bounded by the number of stocks ever seen):
    sector/industry/name, and the next earnings date."""

    __tablename__ = "ticker_reference"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    name: Mapped[str | None] = mapped_column(default=None)
    sector: Mapped[str | None] = mapped_column(default=None)
    industry: Mapped[str | None] = mapped_column(default=None)
    profile_updated: Mapped[str | None] = mapped_column(
        default=None
    )  # ISO date sector/industry were fetched
    next_earnings: Mapped[str | None] = mapped_column(
        default=None
    )  # ISO date, None when none is scheduled
    earnings_updated: Mapped[str | None] = mapped_column(
        default=None
    )  # ISO date next_earnings was fetched


class EarningsEvent(Base, kw_only=True):
    """One quarterly/annual report worth following (discover/earnings_standouts.py):
    a clear beat anywhere in the market, or any report by a past pick.
    Other reporters are never stored, and rows age out after a year, so
    the table stays small."""

    __tablename__ = "earnings_events"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    report_date: Mapped[str] = mapped_column(primary_key=True)  # ISO date
    hour: Mapped[str] = mapped_column(default="")  # "bmo" / "amc" / "" (unknown)
    eps_estimate: Mapped[float | None] = mapped_column(default=None)
    eps_actual: Mapped[float | None] = mapped_column(default=None)
    revenue_estimate: Mapped[float | None] = mapped_column(default=None)
    revenue_actual: Mapped[float | None] = mapped_column(default=None)
    # Two-session move around the report minus SPY's, percent.
    reaction_pct: Mapped[float | None] = mapped_column(default=None)
    # Change in next-year consensus EPS over the 30 days after, fraction.
    revision_pct: Mapped[float | None] = mapped_column(default=None)
    # The year before: EPS beats in the prior quarters, and revenue growth
    # over the same quarter a year earlier, percent. Filled at the last gate.
    prior_beats: Mapped[int | None] = mapped_column(default=None)
    prior_quarters: Mapped[int | None] = mapped_column(default=None)
    revenue_yoy_pct: Mapped[float | None] = mapped_column(default=None)
    # "pending" until decided, then "standout" or "no".
    status: Mapped[str] = mapped_column(default="pending", index=True)
    decided_on: Mapped[str | None] = mapped_column(default=None)  # ISO date status left "pending"


class FundPosition(Base, kw_only=True):
    """One stock position of a tracked hedge fund at a quarter end, from its
    13F filing (data/hedge_funds_13f.py). Common stock only (no options),
    the last FUND_KEEP_PERIODS quarters per fund: ~22 funds x ~60 positions."""

    __tablename__ = "fund_positions"

    cik: Mapped[str] = mapped_column(primary_key=True)
    period: Mapped[str] = mapped_column(primary_key=True)  # ISO quarter-end date
    cusip: Mapped[str] = mapped_column(primary_key=True)
    ticker: Mapped[str | None] = mapped_column(default=None, index=True)
    shares: Mapped[float] = mapped_column(default=0.0)
    value_usd: Mapped[float] = mapped_column(default=0.0)
    filed: Mapped[str] = mapped_column(default="")  # ISO date the 13F was filed
    accession: Mapped[str] = mapped_column(default="")


class CusipTicker(Base, kw_only=True):
    """CUSIP -> ticker, from OpenFIGI, cached forever (a CUSIP never moves).
    `ticker` None means OpenFIGI had no US common stock for it."""

    __tablename__ = "cusip_tickers"

    cusip: Mapped[str] = mapped_column(primary_key=True)
    ticker: Mapped[str | None] = mapped_column(default=None)
    checked_on: Mapped[str] = mapped_column(default="")


class InsiderBuy(Base, kw_only=True):
    """One open-market purchase (Form 4, code P) by an insider of a watched
    company (data/insider_buying.py). Purchases only — sales are routine —
    and pruned after INSIDER_KEEP_DAYS, so a few hundred rows a year."""

    __tablename__ = "insider_buys"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    filing_id: Mapped[str] = mapped_column(primary_key=True)  # SEC accession number
    name: Mapped[str] = mapped_column(primary_key=True)
    filed: Mapped[str] = mapped_column(index=True)  # ISO date the Form 4 was filed
    traded: Mapped[str] = mapped_column(default="")  # ISO date of the purchase
    shares: Mapped[float] = mapped_column(default=0.0)
    price: Mapped[float | None] = mapped_column(default=None)


class AnalystAction(Base, kw_only=True):
    """One analyst's rating or price-target action on a stock that reported
    a result worth following (data/analyst_actions.py, filled by the nightly
    earnings watch for recorded earnings events): who acted, when, and how.
    Kept from 90 days before the report on; tens of rows per stock."""

    __tablename__ = "analyst_actions"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    graded_at: Mapped[str] = mapped_column(primary_key=True)  # ISO timestamp of the action
    firm: Mapped[str] = mapped_column(primary_key=True)
    action: Mapped[str] = mapped_column(default="")  # up / down / init / main (maintained) / reit
    to_grade: Mapped[str] = mapped_column(default="")
    from_grade: Mapped[str] = mapped_column(default="")
    target_action: Mapped[str] = mapped_column(
        default=""
    )  # Raises / Lowers / Maintains / Announces / ...
    target: Mapped[float | None] = mapped_column(default=None)
    prior_target: Mapped[float | None] = mapped_column(default=None)


class ForecastSnapshot(Base, kw_only=True):
    """What analysts expected of a tracked stock on one day
    (data/forecast_snapshots.py). Yahoo keeps only 90 days of estimate
    history; this keeps all of it, point in time, so a model can one day
    learn from revisions without seeing the future. Append-only, about
    175 rows per weekday (~4 MB a year); never pruned."""

    __tablename__ = "forecast_snapshots"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date the snapshot was taken
    price: Mapped[float | None] = mapped_column(default=None)
    eps_current_year: Mapped[float | None] = mapped_column(
        default=None
    )  # consensus EPS, current fiscal year
    eps_next_year: Mapped[float | None] = mapped_column(default=None)
    revenue_current_year: Mapped[float | None] = mapped_column(default=None)
    revenue_next_year: Mapped[float | None] = mapped_column(default=None)
    analysts: Mapped[int | None] = mapped_column(default=None)
    target_mean: Mapped[float | None] = mapped_column(default=None)
    target_high: Mapped[float | None] = mapped_column(default=None)
    target_low: Mapped[float | None] = mapped_column(default=None)
    recommendation_mean: Mapped[float | None] = mapped_column(
        default=None
    )  # 1 strong buy .. 5 sell
    # From the same quote summary, no extra request. Free short-interest
    # history is patchy, so this is where ours starts (FINRA's twice-monthly
    # figure, as Yahoo shows it).
    short_pct_float: Mapped[float | None] = mapped_column(
        default=None
    )  # fraction of the float sold short
    short_ratio: Mapped[float | None] = mapped_column(
        default=None
    )  # days to cover at average volume
    shares_outstanding: Mapped[float | None] = mapped_column(default=None)
    institutions_pct: Mapped[float | None] = mapped_column(
        default=None
    )  # fraction held by institutions
    insiders_pct: Mapped[float | None] = mapped_column(default=None)


class StockView(Base, kw_only=True):
    """The daily email's latest long-term view per stock, reused until
    something changes (see agents/stock_views.py). One row per ticker,
    overwritten; rows for stocks no longer held age out."""

    __tablename__ = "stock_views"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    written_on: Mapped[str]  # ISO date
    price: Mapped[float | None] = mapped_column(default=None)  # price when written
    view: Mapped[str]
    # Headline links already shown for this stock, newest first, JSON. Keeps
    # a reused view from repeating the same five headlines all week. Capped,
    # so the row never grows (see agents/stock_views.py).
    shown_links: Mapped[str | None] = mapped_column(default=None)
    news_on: Mapped[str | None] = mapped_column(
        default=None
    )  # ISO date the links were last written


class IbdRating(Base, kw_only=True):
    """This morning's IBD-style ratings (discover/ibd_ratings.py), one row
    per stock in the $2B+ universe. Replaced whole each morning, so the
    table stays ~1,900 rows."""

    __tablename__ = "ibd_ratings"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    as_of: Mapped[str] = mapped_column(default="")  # ISO date of the last close rated
    price: Mapped[float | None] = mapped_column(default=None)
    industry: Mapped[str | None] = mapped_column(default=None)
    composite: Mapped[int | None] = mapped_column(default=None)
    rs_rating: Mapped[int | None] = mapped_column(default=None)
    eps_rating: Mapped[int | None] = mapped_column(default=None)
    ad_grade: Mapped[str | None] = mapped_column(default=None)
    group_rank: Mapped[int | None] = mapped_column(default=None)
    groups_ranked: Mapped[int | None] = mapped_column(default=None)
    rs_line_high: Mapped[bool | None] = mapped_column(default=None)
    off_high: Mapped[float | None] = mapped_column(default=None)  # fraction below the 52-week high
    six_month: Mapped[float | None] = mapped_column(default=None)
    eps_q1: Mapped[float | None] = mapped_column(
        default=None
    )  # latest quarter's EPS growth, fraction
    eps_source: Mapped[str | None] = mapped_column(
        default=None
    )  # "sec" or "yahoo" (reported, adjusted EPS)
    base: Mapped[str | None] = mapped_column(default=None)
    base_weeks: Mapped[int | None] = mapped_column(default=None)
    base_depth: Mapped[float | None] = mapped_column(default=None)
    pivot: Mapped[float | None] = mapped_column(default=None)
    vs_pivot: Mapped[float | None] = mapped_column(default=None)
    base_status: Mapped[str | None] = mapped_column(default=None)


class IbdMarket(Base, kw_only=True):
    """Market direction each morning (distribution and follow-through days
    on SPY and QQQ). One row a day, a few hundred bytes."""

    __tablename__ = "ibd_market"

    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date of the last close
    status: Mapped[str] = mapped_column(default="")
    detail: Mapped[str] = mapped_column(default="")
    indexes: Mapped[str] = mapped_column(default="")  # JSON: per-index counts


class IbdHistory(Base, kw_only=True):
    """IBD-style ratings over time, so a stock's progress (and the ratings'
    worth) can be compared. Daily for the names that matter (Composite 90+,
    in a buy zone, held or tracked); every stock once a week (`full`), which
    is the point-in-time panel a study of the ratings needs. ~160k small
    rows a year."""

    __tablename__ = "ibd_history"

    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date of the close rated
    ticker: Mapped[str] = mapped_column(primary_key=True)
    full: Mapped[bool] = mapped_column(default=False)  # part of the weekly all-stocks snapshot
    backfilled: Mapped[bool] = mapped_column(
        default=False
    )  # rebuilt from stored bars, not recorded that morning
    price: Mapped[float | None] = mapped_column(default=None)
    composite: Mapped[int | None] = mapped_column(default=None)
    rs_rating: Mapped[int | None] = mapped_column(default=None)
    eps_rating: Mapped[int | None] = mapped_column(default=None)
    ad_grade: Mapped[str | None] = mapped_column(default=None)
    group_rank: Mapped[int | None] = mapped_column(default=None)
    base_status: Mapped[str | None] = mapped_column(default=None)
    pivot: Mapped[float | None] = mapped_column(default=None)


class IbdSignal(Base, kw_only=True):
    """A stock entering a buy zone or breaking out with Composite 80+ — the
    IBD-style "buy" moment, logged once per 30 days per stock and graded
    later against SPY (reporting/leaders.signal_scorecard)."""

    __tablename__ = "ibd_signals"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date of the close it fired on
    status: Mapped[str] = mapped_column(default="")  # "buy zone" or "breakout"
    price: Mapped[float | None] = mapped_column(default=None)
    pivot: Mapped[float | None] = mapped_column(default=None)
    composite: Mapped[int | None] = mapped_column(default=None)
    rs_rating: Mapped[int | None] = mapped_column(default=None)
    eps_rating: Mapped[int | None] = mapped_column(default=None)
    base: Mapped[str | None] = mapped_column(default=None)
    industry: Mapped[str | None] = mapped_column(default=None)
    backfilled: Mapped[bool] = mapped_column(
        default=False
    )  # rebuilt from stored bars, not recorded that morning


class IbdSector(Base, kw_only=True):
    """Each sector's direction each morning (discover/ibd_ratings.
    sector_direction): Leading, Uptrend, Caution or Correction, and why.
    Twelve rows a day."""

    __tablename__ = "ibd_sectors"

    day: Mapped[str] = mapped_column(primary_key=True)
    sector: Mapped[str] = mapped_column(primary_key=True)
    status: Mapped[str] = mapped_column(default="")
    reasons: Mapped[str] = mapped_column(default="")  # "; "-joined
    rank: Mapped[int | None] = mapped_column(default=None)
    stocks: Mapped[int] = mapped_column(default=0)
    breadth: Mapped[float | None] = mapped_column(
        default=None
    )  # share of its stocks above their 50-day
    breadth_before: Mapped[float | None] = mapped_column(
        default=None
    )  # the same, ten sessions earlier
    median_six: Mapped[float | None] = mapped_column(default=None)
    leaders: Mapped[int] = mapped_column(default=0)  # Composite 90+
    etf: Mapped[str | None] = mapped_column(default=None)
    etf_above_50: Mapped[bool | None] = mapped_column(default=None)
    etf_above_200: Mapped[bool | None] = mapped_column(default=None)
    etf_dist_days: Mapped[int | None] = mapped_column(default=None)
    backfilled: Mapped[bool] = mapped_column(default=False)


class StockNews(Base, kw_only=True):
    """The top news items per featured stock for the dashboard (cli/ibd.py,
    each morning): company-specific, ranked by materiality. Replaced whole
    every morning, so the table stays a few hundred rows."""

    __tablename__ = "stock_news"

    ticker: Mapped[str] = mapped_column(primary_key=True)
    rank: Mapped[int] = mapped_column(primary_key=True)
    fetched: Mapped[str] = mapped_column(default="")  # ISO date
    title: Mapped[str] = mapped_column(default="")
    url: Mapped[str | None] = mapped_column(default=None)
    source: Mapped[str | None] = mapped_column(default=None)
    published: Mapped[str | None] = mapped_column(default=None)
    snippet: Mapped[str | None] = mapped_column(default=None)


class OpenRouterSpend(Base, kw_only=True):
    """What OpenRouter billed, per day, model and stage — the ledger the
    daily cap is checked against (openrouter.py). A few rows a day."""

    __tablename__ = "openrouter_spend"

    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date, UTC (OpenRouter bills in UTC)
    model: Mapped[str] = mapped_column(primary_key=True)
    stage: Mapped[str] = mapped_column(primary_key=True)
    calls: Mapped[int] = mapped_column(default=0)
    input_tokens: Mapped[int] = mapped_column(default=0)
    output_tokens: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)


class FilingFacts(Base, kw_only=True):
    """Structured facts read out of a 10-Q/10-K/20-F by an open model
    (agents/filing_reader.py), each with a supporting quote checked against
    the filing text. Tier "A" (stocks acted on) keeps its two newest
    filings, so a quarter can be compared with the one before; tier "B"
    (the rest of the universe) keeps one. ~5 KB a row."""

    __tablename__ = "filing_facts"

    accession: Mapped[str] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    tier: Mapped[str] = mapped_column(default="B")
    form: Mapped[str] = mapped_column(default="")
    filed_on: Mapped[str] = mapped_column(default="")  # ISO date
    period_end: Mapped[str | None] = mapped_column(default=None)  # ISO date the filing reports on
    url: Mapped[str] = mapped_column(default="")
    read_on: Mapped[str] = mapped_column(default="")  # ISO date
    reader_model: Mapped[str] = mapped_column(default="")
    provider: Mapped[str | None] = mapped_column(default=None)  # the host that served the read
    facts: Mapped[str] = mapped_column(default="")  # JSON, the reader's answer
    quotes_checked: Mapped[int] = mapped_column(default=0)
    quotes_found: Mapped[int] = mapped_column(
        default=0
    )  # found verbatim (normalised) in the filing text
    flagged: Mapped[bool] = mapped_column(default=False)
    flag_reasons: Mapped[str] = mapped_column(default="")  # "; "-joined
    # The bulk model whose flagged read this one replaced, if any.
    escalated_from: Mapped[str | None] = mapped_column(default=None)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    # 10-K/20-F only: the risk factors against last year's (data/text_change).
    risk_kept: Mapped[float | None] = mapped_column(
        default=None
    )  # share of sentences carried over verbatim
    risk_cosine: Mapped[float | None] = mapped_column(default=None)


class EightKAlert(Base, kw_only=True):
    """A material 8-K on a held stock, read and emailed the evening it was
    found (reporting/filing_alert.py) — also what stops it being emailed
    twice — or the latest earnings release of a stock acted on, read by
    the weekly run for its guidance (never emailed). A few hundred small
    rows a year."""

    __tablename__ = "eightk_alerts"

    accession: Mapped[str] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    filed_on: Mapped[str] = mapped_column(default="")
    items: Mapped[str] = mapped_column(default="")  # comma-joined 8-K item codes
    url: Mapped[str] = mapped_column(default="")
    read_on: Mapped[str] = mapped_column(default="")
    reader_model: Mapped[str] = mapped_column(default="")
    summary: Mapped[str] = mapped_column(default="")  # JSON
    quotes_checked: Mapped[int] = mapped_column(default=0)
    quotes_found: Mapped[int] = mapped_column(default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)


class SecEvent(Base, kw_only=True):
    """An SEC event filing on a stock we follow (data/sec_events): a late-
    filing notice, a shelf or offering, a planned insider sale (Form 144),
    or a Schedule 13D. Holdings' events are emailed the evening they are
    found; 13Ds across the universe feed discover as an idea source. Also
    what stops a filing being read or emailed twice. Small rows; a few
    thousand a year, mostly Form 144s below the alert threshold."""

    __tablename__ = "sec_events"

    accession: Mapped[str] = mapped_column(primary_key=True)
    ticker: Mapped[str] = mapped_column(index=True)
    form: Mapped[str] = mapped_column(default="")
    kind: Mapped[str] = mapped_column(
        default="", index=True
    )  # late_filing|shelf|offering|planned_sale|activist
    filed_on: Mapped[str] = mapped_column(default="", index=True)
    url: Mapped[str] = mapped_column(default="")
    read_on: Mapped[str] = mapped_column(default="")
    reader_model: Mapped[str] = mapped_column(default="")
    facts: Mapped[str] = mapped_column(default="")  # JSON: parsed fields plus the reader's answer
    alerted: Mapped[bool] = mapped_column(
        default=False
    )  # met the alert bar (a Form 144 below it is only recorded)


class OpenRouterHostCheck(Base, kw_only=True):
    """A known-answer check of one OpenRouter host serving one model
    (openrouter_hosts.canary), run before each weekly filing read. A host
    that failed its latest check is skipped until it passes one."""

    __tablename__ = "openrouter_host_checks"

    day: Mapped[str] = mapped_column(primary_key=True)  # ISO date
    model: Mapped[str] = mapped_column(primary_key=True)
    host: Mapped[str] = mapped_column(primary_key=True)  # OpenRouter slug, e.g. "io-net"
    passed: Mapped[bool] = mapped_column(default=False)
    detail: Mapped[str] = mapped_column(default="")
    cost_usd: Mapped[float] = mapped_column(default=0.0)


class FilingSpotCheck(Base, kw_only=True):
    """Claude re-reading a filing an open model read, field by field
    (cli/filings.py --spot-check): the running measure of the open readers'
    quality, per model and host. Run by hand only (not scheduled)."""

    __tablename__ = "filing_spot_checks"

    accession: Mapped[str] = mapped_column(primary_key=True)
    checked_on: Mapped[str] = mapped_column(default="")  # ISO date
    ticker: Mapped[str] = mapped_column(default="", index=True)
    reader_model: Mapped[str] = mapped_column(default="")
    provider: Mapped[str | None] = mapped_column(default=None)
    claude_model: Mapped[str] = mapped_column(default="")
    agreed: Mapped[int] = mapped_column(default=0)
    compared: Mapped[int] = mapped_column(default=0)
    fields: Mapped[str] = mapped_column(default="")  # JSON {field: agreed}
    cost_usd: Mapped[float] = mapped_column(default=0.0)


class PipelineStep(Base, kw_only=True):
    """One step of a discover/rebalance run (pipeline.py): when it ran, how
    long it took, and its summary line or error."""

    __tablename__ = "pipeline_steps"

    id: Mapped[int | None] = mapped_column(default=None, primary_key=True, nullable=False)
    run_key: Mapped[str] = mapped_column(index=True)  # one id per pipeline run
    pipeline: Mapped[str]
    step: Mapped[str]
    block: Mapped[str | None] = mapped_column(default=None)  # the parallel block it ran in, if any
    started_at: Mapped[str] = mapped_column(index=True)
    seconds: Mapped[float]
    status: Mapped[str]  # "ok" | "failed"
    detail: Mapped[str | None] = mapped_column(default=None)


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
    "PipelineStep",
]
