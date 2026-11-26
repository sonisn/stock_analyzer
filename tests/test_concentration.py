"""Correlated groups, volatility caps and the portfolio drawdown alert
(discover/concentration.py, reporting/portfolio_drawdown.py,
discover/position_cap.py's group and per-stock caps)."""

from __future__ import annotations

import random
from datetime import date, timedelta

from stock_analyzer.discover.concentration import (
    BROAD_FUNDS,
    as_dict,
    assess,
    rebalance_block,
    vol_cap_pct,
)
from stock_analyzer.discover.position_cap import cap_positions
from stock_analyzer.models.rebalance import RebalanceAction, RebalanceActionType, RebalancePlan
from stock_analyzer.reporting.health import PortfolioHealth, decision_items, render_health_html
from stock_analyzer.reporting.portfolio_drawdown import (
    Drawdown,
    attribution,
    build_alert,
    change_between,
    drawdown,
)

END = date(2026, 11, 25)


def _closes(rets: list[float], end: date = END) -> list[tuple[date, float]]:
    closes = [100.0]
    for r in rets:
        closes.append(closes[-1] * (1 + r))
    return [(end - timedelta(days=len(closes) - 1 - i), c) for i, c in enumerate(closes)]


def _book(n: int = 300, seed: int = 7):
    """A shared 'AI' factor driving three names, two loners, a calm fund."""
    rnd = random.Random(seed)
    ai = [rnd.gauss(0, 0.025) for _ in range(n)]
    noise = lambda s: [rnd.gauss(0, s) for _ in range(n)]  # noqa: E731
    return {
        "NVDA": _closes([a + e for a, e in zip(ai, noise(0.01), strict=True)]),
        "AMD": _closes([a + e for a, e in zip(ai, noise(0.012), strict=True)]),
        # BE: the factor plus a lot of its own noise — very volatile
        "BE": _closes([1.5 * a + e for a, e in zip(ai, noise(0.05), strict=True)]),
        "GOOGL": _closes(noise(0.015)),
        "KO": _closes(noise(0.008)),
        "VOO": _closes([0.4 * a + e for a, e in zip(ai, noise(0.006), strict=True)]),
        "SPAXX": _closes([0.0] * n),
    }


VALUES = {
    "NVDA": 200_000.0,
    "AMD": 150_000.0,
    "BE": 150_000.0,
    "GOOGL": 100_000.0,
    "KO": 100_000.0,
    "VOO": 80_000.0,
    "SPAXX": 20_000.0,
}


def test_names_that_move_together_form_one_group_held_to_the_cap():
    c = assess(VALUES, _book(), max_group_pct=50, min_corr=0.5)
    assert [g.tickers for g in c.groups] == [["NVDA", "AMD", "BE"]]
    g = c.groups[0]
    # $500k of $800k: 62.5%, $100k over a 50% cap.
    assert g.pct == 62.5 and g.over and round(g.trim_usd) == 100_000
    assert g.worst_drop_pct and g.worst_drop_pct > 0
    # Broad funds and cash are never grouped or capped.
    assert "VOO" in BROAD_FUNDS
    assert {v.ticker for v in c.vol_caps} == {"NVDA", "AMD", "BE", "GOOGL", "KO"}


def test_a_volatile_stock_gets_a_smaller_cap():
    assert vol_cap_pct(0.30, max_pct=35, ref_vol=0.45) == 35
    assert vol_cap_pct(0.90, max_pct=35, ref_vol=0.45) == 17.5
    assert vol_cap_pct(5.0, max_pct=35, ref_vol=0.45) == 5.0  # floor
    assert vol_cap_pct(0.90, max_pct=35, ref_vol=0) == 35  # off
    c = assess(VALUES, _book(), max_position_pct=35, ref_vol=0.45)
    be = next(v for v in c.vol_caps if v.ticker == "BE")
    assert be.vol_pct > 45 and be.cap_pct < 35
    assert be.over and be.trim_usd > 0


def test_off_switches_and_the_rebalance_block():
    c = assess(VALUES, _book(), max_group_pct=100, ref_vol=0)
    assert not c.over_groups and not c.over_vol
    assert rebalance_block(c, step_pct=10, core_fund="VOO") == ""
    # Default: the group is never sold for concentration alone; BE's
    # volatility cap still asks for a trim.
    c = assess(VALUES, _book())
    block = rebalance_block(c, step_pct=10, core_fund="VOO")
    assert "No BUY or ADD of these names" in block and "Do NOT TRIM or SELL them" in block
    assert "TRIM toward the volatility caps above" in block and "~$80,000" in block
    assert as_dict(c)["groups"][0]["over"] is True and as_dict(c)["group_trims"] is False
    # Group only, no volatile stock over its cap: no trim instruction at all.
    calm = assess(VALUES, _book(), ref_vol=0)
    assert "TRIM toward" not in rebalance_block(calm, step_pct=10, core_fund="VOO")
    # Trims switched on.
    c = assess(VALUES, _book(), group_trims=True)
    block = rebalance_block(c, step_pct=10, core_fund="VOO")
    assert "TRIM toward these caps" in block and "Do NOT TRIM" not in block
    assert "VOO" in block and "tax-advantaged" in block


def test_a_holding_without_history_is_left_out_not_guessed():
    c = assess({"NEWCO": 1000.0, **VALUES}, _book())
    assert all(v.ticker != "NEWCO" for v in c.vol_caps)


# --- buys cut in the rebalance ------------------------------------------------------


def _plan(*actions: tuple[RebalanceActionType, str, str]) -> RebalancePlan:
    return RebalancePlan(
        status="ACTION",
        aggressiveness_applied="balanced",
        actions=[RebalanceAction(action=a, ticker=t, sizing=s) for a, t, s in actions],
        summary="Plan.",
        full_text="plan",
    )


POSITIONS = {
    "NVDA": {"units": 1000.0, "value": 300_000.0},
    "AMD": {"units": 1000.0, "value": 200_000.0},
    "GOOGL": {"units": 1000.0, "value": 100_000.0},
}  # + $400k cash = $1M book


def _cap(plan, **kw):
    return cap_positions(plan, positions=POSITIONS, cash=400_000.0, max_pct=35.0, **kw)


def test_a_buy_into_an_over_cap_group_is_dropped():
    plan, warnings = _cap(
        _plan(("ADD", "AMD", "~$20,000"), ("ADD", "GOOGL", "~$20,000")),
        groups=[["NVDA", "AMD"]],
        max_group_pct=50,
    )
    assert [a.ticker for a in plan.actions] == ["GOOGL"]
    assert "AMD: dropped ADD" in warnings[0] and "move together" in warnings[0]


def test_a_trim_in_the_group_makes_room_and_the_vol_cap_cuts():
    plan, warnings = _cap(
        _plan(("TRIM", "NVDA", "~$50,000"), ("ADD", "AMD", "~$80,000")),
        groups=[["NVDA", "AMD"]],
        max_group_pct=50,
        ticker_max_pct={"AMD": 25.0},
    )
    # Group: $500k - $50k = $450k, $50k of room; AMD's own 25% cap: $50k. Both $50k.
    assert plan.actions[1].sizing.startswith("~$50,000 (cut from: ~$80,000)")
    assert "cut from ~$80,000 to ~$50,000" in warnings[0]


def test_without_concentration_inputs_the_plain_cap_is_unchanged():
    plan, warnings = _cap(_plan(("ADD", "AMD", "~$200,000")))
    # AMD $200k + $200k past 35% of $1M: cut to $150k.
    assert plan.actions[0].sizing.startswith("~$150,000 (cut from")
    assert "35% single-stock cap" in warnings[0]


# --- whole-portfolio drawdown ---------------------------------------------------------


def _totals(values: list[float]) -> list[tuple[date, float]]:
    start = date(2026, 10, 1)
    return [(start + timedelta(days=i), v) for i, v in enumerate(values)]


def _dd(rows: list[tuple[date, float]]) -> Drawdown:
    dd = drawdown(rows)
    assert dd is not None
    return dd


def test_the_alert_fires_once_per_step_and_resets_at_a_new_high():
    assert _dd(_totals([100, 95])).band == 0  # 5%: under the threshold
    dd = _dd(_totals([100, 95, 89]))
    assert dd.band == 10 and dd.is_new_step and round(dd.pct) == 11
    dd = _dd(_totals([100, 89, 92, 88]))  # back to the same step
    assert dd.band == 10 and not dd.is_new_step
    dd = _dd(_totals([100, 89, 84]))
    assert dd.band == 15 and dd.is_new_step
    dd = _dd(_totals([100, 74, 105, 94]))  # new high at 105, now -10.5%
    assert dd.peak == 105 and dd.band == 10 and dd.is_new_step
    assert drawdown(_totals([100])) is None


def test_the_high_is_the_past_year_only():
    rows = [(date(2025, 1, 2), 200.0), (date(2026, 11, 1), 100.0), (date(2026, 11, 2), 95.0)]
    dd = _dd(rows)
    assert dd.peak == 100.0 and dd.band == 0


def test_attribution_and_the_alert_body():
    series = {
        "NVDA": [(date(2026, 10, 21), 200.0), (date(2026, 11, 25), 150.0)],
        "GOOGL": [(date(2026, 10, 21), 100.0), (date(2026, 11, 25), 110.0)],
        "SPY": [(date(2026, 10, 21), 700.0), (date(2026, 11, 25), 650.0)],
    }
    movers = attribution({"NVDA": 100, "GOOGL": 10}, series, date(2026, 10, 21), END)
    assert movers[0] == {"ticker": "NVDA", "pct": -25.0, "usd": -5000.0}
    spy = change_between(series["SPY"], date(2026, 10, 21), END)
    dd = _dd([(date(2026, 10, 21), 469_587.0), (END, 368_704.0)])
    subject, body = build_alert(
        dd,
        spy_pct=spy,
        movers=movers,
        concentration={
            "max_group_pct": 50,
            "groups": [{"tickers": ["BE", "NVDA"], "pct": 81, "over": True, "worst_drop_pct": 37}],
            "vol_caps": [],
        },
        market={"status": "Market in correction", "detail": "QQQ 16% off the high"},
        core_fund="VOO",
    )
    assert subject == "Portfolio down 21% from its high (-$100,883)"
    assert "SPY over the same days: -7.1%" in body and "NVDA</b> -25.0%" in body
    assert "81% of the portfolio" in body and "Market in correction" in body
    assert "No sale is suggested" in body
    assert "GOOGL" not in body.split("Where it went")[1].split("</ul>")[0]


def test_the_weekly_email_names_the_concentration_decision():
    h = PortfolioHealth()
    h.units_by_account = {"NVDA": {"Traditional IRA": 10.0}, "BE": {"Robinhood Individual": 5.0}}
    h.concentration = {
        "max_group_pct": 50,
        "core_fund": "VOO",
        "groups": [
            {
                "tickers": ["BE", "NVDA"],
                "pct": 81.0,
                "over": True,
                "worst_drop_pct": 37.0,
                "loss_at_worst_usd": 110_000.0,
                "trim_usd": 114_000.0,
            }
        ],
        "vol_caps": [
            {
                "ticker": "BE",
                "vol_pct": 112.0,
                "cap_pct": 14.0,
                "pct": 19.6,
                "over": True,
                "trim_usd": 20_000.0,
            }
        ],
    }
    h.portfolio_drawdown = {
        "pct": 21.5,
        "loss": 101_000.0,
        "peak": 469_587.0,
        "peak_day": "2026-10-21",
        "spy_pct": -7.3,
    }
    h.market = {"status": "Market in correction", "detail": "QQQ 16% off the high"}
    # Default: no sale for concentration — information (priority 4) only.
    items = {i["label"]: i for i in decision_items(h)}
    group = items["CONCENTRATED"]
    assert group["priority"] == 4 and "No sale suggested" in group["text"]
    assert "BE, NVDA move together and are 81%" in group["text"] and "VOO" in group["text"]
    assert items["TOO VOLATILE"]["priority"] == 3
    assert items["TOO VOLATILE"]["text"].startswith("BE swings 112% a year")
    # Trims on: an action, from the tax-free account first.
    h.concentration = {**h.concentration, "group_trims": True}
    group = next(i for i in decision_items(h) if i["label"] == "CONCENTRATED")
    assert group["priority"] == 3 and "Traditional IRA first" in group["text"]
    h.concentration = {**h.concentration, "group_trims": False}
    page = render_health_html(h)
    assert "21.5% below" in page and "SPY -7.3%" in page and "CONCENTRATED" in page
    assert "poor time to sell" in page
