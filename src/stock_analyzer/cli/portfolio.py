"""Portfolio analysis pipeline: SnapTrade holdings → analysis → email."""

from __future__ import annotations

from datetime import date

from dotenv import load_dotenv

from ..agents.portfolio import PortfolioAgent
from ..agents.stock_views import is_equity
from ..config import Settings
from ..data import finnhub, yf_gateway
from ..data.brokerage import (
    fetch_account_sync_status,
    fetch_covered_call_obligations,
    fetch_portfolio_holdings,
    listed_tickers,
    stale_account_notes,
)
from ..data.chart_img import fetch_charts
from ..data.pricing import quotes_from_ticker_data, reconcile_prices
from ..logging import get_logger
from ..reporting.html import format_html
from ..reporting.smtp import SmtpServer

# Caching is disabled. To re-enable for iterating on email format without
# re-running data fetch + LLM:
#   from ..cache import FileCache
#   CACHE = FileCache("last_analysis.txt")
# Then in run_analysis(): check CACHE.read() first, write CACHE.write(result)
# at the end. Settings already has `use_cached_analysis` as the toggle.

logger = get_logger(__name__)


def _build_agent(settings: Settings) -> PortfolioAgent:
    return PortfolioAgent(
        "Portfolio Manager",
        settings.llm_provider,
        settings.llm_model,
        sentiment_provider=settings.sentiment_provider,
        sentiment_model=settings.sentiment_model,
        ticker_provider=settings.ticker_provider,
        ticker_model=settings.ticker_model,
        rerank_provider=settings.rerank_provider,
        rerank_model=settings.rerank_model,
        db_path=settings.discover_db_path,
        view_max_age_days=settings.stock_view_max_age_days,
        view_move_pct=settings.stock_view_move_pct,
    )


def portfolio_health(
    settings: Settings,
    holdings: dict[str, list[dict]],
    *,
    ticker_data: dict[str, dict] | None = None,
    prices: dict[str, float] | None = None,
    data_notes: list[str] | None = None,
    stale_accounts: list[str] | None = None,
    covered_calls: dict[str, dict] | None = None,
    roll_ideas: dict[str, str] | None = None,
    sector_rotation: dict | None = None,
    backlog: dict | None = None,
    optionable: set[str] | None = None,
    world_markets: list[dict] | None = None,
):
    """The deterministic PortfolioHealth (reporting/health.py) with the live
    data sources wired in. None when disabled or on any failure — the daily
    email must still go out."""
    if not settings.portfolio_health:
        return None
    from ..reporting.health import build_portfolio_health

    db = settings.discover_db_path
    prices = prices or {}

    def sector_of(tickers: list[str]) -> dict[str, str]:
        from ..data.reference import profiles

        return {t: p["sector"] for t, p in profiles(tickers, db).items() if p.get("sector")}

    def held_thesis_checks(held: set[str]) -> list[dict]:
        from ..data.eps_revisions import batch_eps_revisions
        from ..discover.thesis_tracker import check_theses, load_open_picks, thesis_report_data

        picks = [p for p in load_open_picks(db) if p.ticker in held]
        if not picks:
            return []
        # Estimate cuts are what turn a price signal into a BROKEN thesis.
        revisions = batch_eps_revisions([p.ticker for p in picks])
        return thesis_report_data(check_theses(picks, eps_revisions=revisions))

    def harvest() -> list[dict]:
        from ..data.brokerage import fetch_account_meta, fetch_covered_call_obligations
        from ..data.transactions import fetch_transaction_history, to_tax_payloads
        from ..discover.reinvest import sector_peers
        from ..discover.tax_harvest import find_harvest_candidates, harvest_report_data
        from .rebalance import _build_position_splits

        splits = _build_position_splits(holdings, fetch_account_meta())
        lot_prices = {
            h["ticker"]: prices.get(str(h["ticker"]).upper()) or h.get("price")
            for items in holdings.values()
            for h in items
            if h.get("ticker")
        }
        return harvest_report_data(
            find_harvest_candidates(
                splits,
                lot_prices,
                to_tax_payloads(fetch_transaction_history(db_path=db)),
                # Same-sector names keep the exposure after a loss sale —
                # still a stock you do not own, so it follows the same
                # switch. The harvest itself stays either way.
                sector_peers(db, list(splits), held=set(splits))
                if settings.daily_email_new_ideas
                else {},
                min_loss_usd=settings.harvest_min_loss_usd,
                min_loss_pct=settings.harvest_min_loss_pct,
                # Shares backing a short call are not sellable, so they
                # are not harvestable either.
                covered_calls=fetch_covered_call_obligations(),
            )
        )

    def earnings(tickers: list[str]) -> dict[str, dict]:
        from ..data.earnings_calendar import batch_earnings_flags

        return batch_earnings_flags(tickers, within_days=7, db_path=db)

    def income(units: dict[str, float], values: dict[str, float]) -> dict:
        from ..data.transactions import fetch_cash_activity
        from ..discover.income import dividend_income, forward_dividend_rates

        return dividend_income(
            units=units,
            values=values,
            rates=forward_dividend_rates(sorted(units)),
            received=fetch_cash_activity(days_back=370, db_path=db)["dividends"],
        )

    def add_on(**kwargs) -> list[dict]:
        from ..data.eps_revisions import batch_eps_revisions
        from ..discover.add_on import add_on_candidates, price_vs_high

        def estimates_cut(tickers: list[str]) -> set[str]:
            revisions = batch_eps_revisions(tickers)
            return {t for t, r in revisions.items() if (r or {}).get("direction_30d") == "lowering"}

        return add_on_candidates(
            highs=price_vs_high(sorted(kwargs["values"])), estimates_cut=estimates_cut, **kwargs
        )

    def earnings_results() -> list[dict]:
        from ..data.eps_revisions import batch_eps_revisions
        from ..discover.post_earnings import recent_results, with_revisions

        recent = recent_results(ticker_data or {}, today=date.today())
        if not recent:
            return []
        return with_revisions(recent, batch_eps_revisions([r["ticker"] for r in recent]))

    def reinvest(held: set[str], over_cap: set[str], n: int) -> list[dict]:
        from ..discover.reinvest import load_pick_pool, reinvest_ideas, with_sector_bias

        ideas = reinvest_ideas(load_pick_pool(db), held=held, avoid_sectors=over_cap, n=n)
        # The same rotation the report shows, so the tag on an idea and
        # the table above it can never disagree — and it is fetched once.
        return with_sector_bias(ideas, sector_rotation)

    try:
        return build_portfolio_health(
            holdings,
            prices=prices,
            data_notes=data_notes,
            stale_accounts=stale_accounts,
            covered_calls=covered_calls,
            roll_ideas=roll_ideas,
            sector_rotation=sector_rotation,
            backlog=backlog,
            optionable=optionable,
            options_accounts=settings.options_accounts,
            world_markets=world_markets,
            max_sector_pct=settings.discover_max_sector_pct,
            sector_of=sector_of,
            held_thesis_checks=held_thesis_checks,
            harvest=harvest,
            earnings=earnings,
            # With new ideas off, a sale line stops naming somewhere to
            # put the money: the pick pool it would draw from is the
            # thing being distrusted.
            reinvest=reinvest if settings.daily_email_new_ideas else None,
            income=income,
            add_on=add_on,
            earnings_results=earnings_results if ticker_data else None,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Portfolio health block failed (%s) — sending the email without it", e)
        return None


def attach_idea_details(health) -> list[str]:
    """Fetch market data for every stock the report suggests buying and
    hang it on `health`. Returns the tickers, in suggestion order.

    Deterministic — the same yfinance snapshot a holding's block is built
    from, with no model call: an idea is worth a chart, not another
    round of tokens.
    """
    from ..data.ticker import fetch_ticker_data
    from ..reporting.health import suggested_tickers

    if health is None:
        return []
    tickers = suggested_tickers(health)
    if not tickers:
        return []
    reasons = {i["ticker"]: i.get("reason") for i in health.reinvest if i.get("ticker")}
    for ticker in tickers:
        try:
            data = fetch_ticker_data(ticker)
        except Exception as e:  # noqa: BLE001 — one idea's data is not the email
            logger.warning("No market data for suggested %s (%s)", ticker, e)
            continue
        if not data:
            continue
        health.idea_details[ticker] = {**data, "reason": reasons.get(ticker)}
    return [t for t in tickers if t in health.idea_details]


def held_backlog(tickers: list[str]) -> dict:
    """Contracted-but-undelivered revenue per holding, or {}.

    Free (SEC XBRL) and guarded: a company that does not tag the concept
    is simply absent, and a failure costs the section rather than the
    email.
    """
    try:
        from ..data.backlog import batch_rpo

        return batch_rpo(tickers)
    except Exception as e:  # noqa: BLE001
        logger.warning("Order-book data unavailable (%s)", e)
        return {}


def market_rotation() -> dict:
    """Six-month sector returns, or {} — one batch call, no LLM.

    Guarded: the report loses a table when it fails, not the email.
    """
    try:
        from ..data.sector_rotation import sector_rotation_summary

        return sector_rotation_summary()
    except Exception as e:  # noqa: BLE001
        logger.warning("Sector rotation unavailable (%s)", e)
        return {}


def roll_ideas_for(
    settings: Settings,
    covered_calls: dict[str, dict],
    prices: dict[str, float],
) -> dict[str, str]:
    """A costed roll for every written call the stock has run at.

    Only for positions close to their strike — a chain fetch per holding
    would be a lot of requests to tell you that a call 40% out of the
    money is fine. Never raises: a missing chain costs the suggestion,
    not the email.
    """
    from ..data.options_chain import fetch_chains
    from ..discover.cc_roll import roll_suggestion
    from ..reporting.health import ASSIGNMENT_WATCH_PCT

    at_risk = []
    for ticker, rec in (covered_calls or {}).items():
        strike, price = rec.get("lowest_strike"), prices.get(ticker.upper())
        if not strike or not price:
            continue
        if 0 <= (strike / price - 1) * 100 <= ASSIGNMENT_WATCH_PCT:
            at_risk.append(ticker)
    if not at_risk:
        return {}
    try:
        # Wide window: a roll that pays often sits past the band new
        # calls are written in, and the suggestion says so when it does.
        chains = fetch_chains(at_risk, dte_min=30, dte_max=500)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not fetch chains for a roll suggestion (%s)", e)
        return {}
    out = {}
    for ticker in at_risk:
        chain = chains.get(ticker)
        if chain is None or chain.source == "missing":
            continue
        try:
            idea = roll_suggestion(
                ticker=ticker,
                obligation=covered_calls[ticker],
                chain=chain,
                min_upside_pct=settings.cc_min_upside_pct,
                delta_max=settings.cc_target_delta_max,
                dte_max=settings.cc_dte_max,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Roll suggestion for %s failed (%s)", ticker, e)
            continue
        if idea:
            out[ticker] = idea
    return out


def record_portfolio_snapshot(
    settings: Settings,
    holdings: dict[str, list[dict]],
    *,
    prices: dict[str, float] | None = None,
) -> None:
    """Today's total value (holdings + cash) for the portfolio-vs-SPY
    comparison. Skipped, not guessed, when cash can't be read."""
    from ..data.brokerage import fetch_account_cash
    from ..db.repository import record_snapshot
    from ..db.session import get_session
    from ..reporting.health import aggregate_positions

    try:
        cash = fetch_account_cash()
        if not cash:
            logger.warning("No cash balance readable — portfolio snapshot skipped today")
            return
        value = sum(p["value"] for p in aggregate_positions(holdings, prices).values())
        with get_session(settings.discover_db_path) as session:
            record_snapshot(
                session,
                day=date.today().isoformat(),
                holdings_value=value,
                cash=sum(cash.values()),
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record today's portfolio snapshot (%s)", e)


def record_daily_suggestions(settings: Settings, health) -> None:
    """Keep today's actionable advice for the quarterly review. Never
    blocks the email."""
    if health is None:
        return
    from ..db.repository import record_suggestions
    from ..db.session import get_session
    from ..reporting.health import suggestion_rows

    try:
        rows = suggestion_rows(health, today=date.today().isoformat())
        if rows:
            with get_session(settings.discover_db_path) as session:
                added = record_suggestions(session, rows)
            logger.info("Recorded %d new suggestion(s) for the quarterly review", added)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not record today's suggestions (%s)", e)


def run_analysis(
    settings: Settings,
    holdings: dict[str, list[dict]] | None = None,
    *,
    agent: PortfolioAgent | None = None,
) -> tuple[str, list[str]]:
    """Return (report_text, tickers). Tickers are exposed so callers can fetch
    per-ticker chart images for the email."""
    if holdings is None:
        holdings = fetch_portfolio_holdings()
    # A CUSIP for a revoked listing or a 401(k) commingled pool has no
    # quote, no news and nothing for a model to say — analyzing it costs a
    # fetch and an LLM call to produce an empty block.
    tickers, unlisted = listed_tickers(holdings)
    if unlisted:
        logger.info("Skipping %s — no market data for these symbols", ", ".join(unlisted))
    if not tickers:
        raise RuntimeError("No tickers returned from SnapTrade — check connected accounts.")
    logger.info("Analyzing %d tickers: %s", len(tickers), ", ".join(tickers))

    agent = agent or _build_agent(settings)
    text = agent.run_analysis(tickers, holdings=holdings)
    logger.info(
        "Long-term views: %d rewritten, %d reused from the database",
        agent.views_written,
        agent.views_reused,
    )
    return text, tickers


def build_email(result: str, health, chart_cids: dict[str, str]) -> tuple[str, str]:
    """(subject, HTML body). With a health result, the email opens with the
    short "Decide today" list, the subject carries how many decisions are
    waiting, and flagged holdings are listed first."""
    from ..reporting.health import (
        decision_count,
        flagged_tickers,
        render_decisions_html,
        render_health_html,
    )

    day = date.today().strftime("%b-%d")
    if health is None:
        subject = f"Portfolio Analysis - {day}"
        return subject, format_html(result, title=subject, chart_cids=chart_cids)
    n = decision_count(health)
    subject = f"Portfolio {day}: " + (f"{n} to decide" if n else "nothing to decide")
    return subject, format_html(
        result,
        title=f"Portfolio Analysis - {day}",
        chart_cids=chart_cids,
        health_html=render_decisions_html(health) + render_health_html(health),
        first_tickers=flagged_tickers(health),
    )


def _chart_cid(ticker: str) -> str:
    # Periods/dashes are valid in CIDs but normalize for safety.
    return "chart-" + ticker.replace(".", "-").replace("/", "-")


def main() -> None:
    from ..market_time import use_market_timezone

    use_market_timezone()
    load_dotenv()
    # Pacing knobs live in the environment, and these modules are
    # imported before `.env` is loaded — re-read them now.
    yf_gateway.reload_from_env()
    finnhub.reload_from_env()
    settings = Settings.from_env()

    holdings = fetch_portfolio_holdings()
    agent = _build_agent(settings)
    result, tickers = run_analysis(settings, holdings, agent=agent)
    # One price per ticker — the live quotes the run just fetched — so a
    # stale brokerage feed can't inflate the value, the sector weights or
    # the snapshot the quarterly vs-SPY return is built from.
    prices, price_notes = reconcile_prices(holdings, quotes_from_ticker_data(agent.ticker_data))
    # A broker that has stopped syncing reports July's shares, cash and
    # prices as if they were today's, so name the account instead of
    # quietly valuing stale data.
    stale = stale_account_notes(fetch_account_sync_status())
    # The exchanges that priced these holdings overnight. Guarded: a
    # missing index is context lost, not an email lost.
    try:
        from ..data.world_markets import fetch_world_markets

        world = fetch_world_markets()
    except Exception as e:  # noqa: BLE001
        logger.warning("World markets unavailable (%s)", e)
        world = []
    # Shares backing a written call are already promised, so every sale
    # suggestion below has to say what closing the position would take.
    try:
        covered_calls = fetch_covered_call_obligations()
    except Exception as e:  # noqa: BLE001
        logger.warning("Covered-call positions unavailable (%s)", e)
        covered_calls = {}
    _, unlisted = listed_tickers(holdings)
    if unlisted:
        price_notes.append(
            "no market data for " + ", ".join(unlisted) + " — held and valued, but not analyzed"
        )
    health = portfolio_health(
        settings,
        holdings,
        ticker_data=agent.ticker_data,
        prices=prices,
        roll_ideas=roll_ideas_for(settings, covered_calls, prices),
        sector_rotation=market_rotation(),
        # Signed orders not yet delivered — the one forward number in the
        # email that is not a forecast.
        backlog=held_backlog(tickers),
        data_notes=price_notes,
        stale_accounts=stale,
        covered_calls=covered_calls,
        # A money-market fund has no options chain, whatever its quote
        # looks like — SPAXX's 20,846 units are not 208 contracts.
        optionable={t.upper() for t, d in (agent.ticker_data or {}).items() if is_equity(d)}
        or None,
        world_markets=world,
    )
    record_daily_suggestions(settings, health)
    record_portfolio_snapshot(settings, holdings, prices=prices)
    if not settings.email_to:
        logger.error("EMAIL_TO not set; printing report instead of emailing")
        print(result)
        return

    # A suggested stock gets the same look as a held one: its own chart,
    # trends and valuation. Without it an idea is a ticker and a
    # sentence, which is not enough to act on.
    ideas = attach_idea_details(health) if settings.daily_email_new_ideas else []
    charts = fetch_charts(tickers + [t for t in ideas if t not in tickers])
    chart_cids = {t: _chart_cid(t) for t in charts}
    inline_images = {_chart_cid(t): data for t, data in charts.items()}
    for ticker in ideas:
        if ticker in charts and health is not None:
            health.idea_details[ticker]["chart_cid"] = _chart_cid(ticker)

    subject, body = build_email(result, health, chart_cids)
    SmtpServer().send_email(
        settings.email_to,
        subject,
        body,
        content_type="html",
        inline_images=inline_images or None,
    )


if __name__ == "__main__":
    main()
