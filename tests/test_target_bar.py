from stock_analyzer.cli.portfolio import build_email
from stock_analyzer.reporting.health import _stock_detail_html
from stock_analyzer.reporting.target_bar import analyst_target_html, target_summary


def _data(price=100.0, low=80.0, mean=120.0, high=150.0, **extra):
    return {
        "price_value": price,
        "analyst_target": f"${mean:,.2f}",
        "analyst_targets": {
            "low": low,
            "mean": mean,
            "high": high,
            "count": 31,
            "rating": "strong_buy",
        },
        **extra,
    }


def test_bar_marks_price_and_average_inside_the_range():
    out = analyst_target_html(_data())
    assert "31 analysts" in out and "rated strong buy" in out
    assert "Low $80.00 · Average $120.00 · High $150.00" in out
    assert "+20% to the average target" in out
    # Price at 20/70 of the way along: label left-aligned, starting at 29%.
    assert 'width="29%"' in out and "▼ now $100.00" in out
    # Average at 40/70 = 57%: label right-aligned, ending there.
    assert 'width="57%"' in out and "avg $120.00 ▲" in out


def test_price_outside_the_range_stretches_the_scale_and_says_so():
    above = analyst_target_html(_data(price=200.0))
    assert "above every analyst&#x27;s target" in above
    assert "now $200.00 ▼" in above  # at the right end of the bar
    below = analyst_target_html(_data(price=50.0))
    assert "below every analyst&#x27;s target" in below


def test_no_bar_without_a_full_set_of_targets():
    assert analyst_target_html({"price_value": 10.0}) == ""
    assert analyst_target_html(_data(high=None)) == ""
    assert analyst_target_html(_data(price=None)) == ""
    assert target_summary(_data(low=150.0, high=150.0)) is None


def test_idea_block_shows_the_bar_instead_of_the_bare_average():
    html = _stock_detail_html("XYZ", _data(name="Xyz Corp"))
    assert "Analyst 12-month targets" in html and "Analyst target $" not in html
    plain = _stock_detail_html("XYZ", {"name": "Xyz", "analyst_target": "$5.00"})
    assert "Analyst target $5.00" in plain


def test_holding_blocks_get_the_bar_under_their_chart():
    report = (
        "Social/Economic Sentiment:\nCalm.\n"
        "----------------------------------------\n"
        "OK - Okay Corp\nPrice: $100\n"
        "----------------------------------------\n"
        "NONE - No Targets Inc\nPrice: $5\n"
    )
    _, body = build_email(
        report, None, {"OK": "chart-OK"}, {"OK": _data(), "NONE": {"price_value": 5.0}}
    )
    assert body.count("Analyst 12-month targets") == 1
    assert body.index('src="cid:chart-OK"') < body.index("Analyst 12-month targets")
    assert body.index("Analyst 12-month targets") < body.index("<h2>NONE")


def test_rebalance_report_puts_the_bar_under_picks_and_holding_reviews():
    from stock_analyzer.discover.rebalance_sections import append_holding_review_sections
    from stock_analyzer.discover.report_html import render_html_email
    from stock_analyzer.discover.report_pdf import render_pdf
    from stock_analyzer.discover.report_sections import append_pick_cards

    fund = {
        "quote_price": 100.0,
        "analyst_target_low": 80.0,
        "analyst_target_mean": 120.0,
        "analyst_target_high": 150.0,
        "analyst_count": 12,
        "analyst_recommendation": "buy",
    }
    s = []
    append_pick_cards(
        s,
        pick_order=["PICK", "BARE"],
        structured_ranker=None,
        structured_redteam=None,
        structured_sizer=None,
        ranker_text="",
        redteam_text="",
        sizer_text="",
        pick_catalysts=None,
        fundamentals={"PICK": fund, "BARE": {"quote_price": 5.0}},
    )
    append_holding_review_sections(s, {"HELD": "Hold it."}, {"HELD": fund})
    html = render_html_email(s, {"PICK": "chart-PICK"})
    assert html.count("Analyst 12-month targets") == 2
    assert (
        html.index("cid:chart-PICK") < html.index("Analyst 12-month targets") < html.index("BARE")
    )
    assert "12 analysts · rated buy" in html
    assert render_pdf(s, {})  # the PDF takes the section as a text line


def test_yahoo_none_rating_is_left_out():
    data = _data()
    data["analyst_targets"]["rating"] = "none"
    assert "rated" not in analyst_target_html(data)
