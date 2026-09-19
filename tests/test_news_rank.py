"""Deterministic news ranking — what the daily email shows on the mornings
it doesn't pay for an LLM rerank."""

from __future__ import annotations

from types import SimpleNamespace

from stock_analyzer.agents.news_reranker import NewsReranker
from stock_analyzer.data.news_rank import company_terms, mentions_company, rank_news


def _item(title, *, snippet="", publisher="Yahoo Finance"):
    return {
        "title": title,
        "link": f"https://x/{hash(title) % 1000}",
        "snippet": snippet,
        "publisher": publisher,
    }


# Feed order is deliberately worst-first: the listicle is what the email
# would have shown for a reused view before this ranking existed.
CANDIDATES = [
    _item("3 Reasons to Buy Broadcom Stock Like There's No Tomorrow", publisher="Motley Fool"),
    _item("Prediction: Where Will Broadcom Stock Be in 5 Years?", publisher="Motley Fool"),
    _item("Chip sector drifts as traders await the Fed"),
    _item("Analyst raises Broadcom price target to $420", publisher="Barron's"),
    _item("Broadcom unveils new Tomahawk networking chip"),
    _item(
        "Broadcom beats Q3 earnings estimates, raises AI revenue guidance",
        snippet="The company reported EPS of $2.44 against a $2.40 estimate.",
        publisher="Reuters",
    ),
    _item("Broadcom named in antitrust probe over VMware licensing", publisher="Bloomberg"),
]


def test_most_material_first():
    top = rank_news(CANDIDATES, "AVGO", "Broadcom Inc.")
    assert [t["title"].split(" ")[0] for t in top][:1] == ["Broadcom"]
    titles = [t["title"] for t in top]
    assert "earnings" in titles[0] and "guidance" in titles[0]
    assert "antitrust probe" in titles[1]
    # Listicles and generic sector commentary rank below real news.
    assert titles.index("Broadcom unveils new Tomahawk networking chip") < titles.index(
        "Analyst raises Broadcom price target to $420"
    )
    assert "3 Reasons to Buy" not in " ".join(titles[:3])


def test_items_that_do_not_name_the_company_are_dropped():
    top = rank_news(CANDIDATES, "AVGO", "Broadcom Inc.")
    assert "Chip sector drifts as traders await the Fed" not in [t["title"] for t in top]


def test_nothing_relevant_returns_nothing():
    """An empty news section is the honest answer: on a live NVDA feed
    (2026-09-19) all ten items were about other companies."""
    assert rank_news([_item(f"Sector note {i}") for i in range(8)], "NVDA", "NVIDIA") == []


def test_naming_the_company_is_not_enough():
    """Filler that happens to spell the name loses its slot too."""
    filler = [
        _item("Are Oils-Energy Stocks Lagging Bloom Energy (BE) This Year?", publisher="Zacks"),
        _item("Bloom Energy wins 200MW fuel-cell order from a hyperscaler"),
    ]
    assert [i["title"] for i in rank_news(filler, "BE", "Bloom Energy Corporation")] == [
        "Bloom Energy wins 200MW fuel-cell order from a hyperscaler"
    ]


def test_two_letter_tickers_need_full_caps():
    """`\bBe\b` against a sentence-cased headline made every "Could Be
    Bigger" story news about BE (Bloom Energy)."""
    assert not mentions_company(
        "The Anthropic IPO Could Be Bigger Than SpaceX", "BE", "Bloom Energy"
    )
    assert mentions_company("Why Are BE, FCEL, GEV Stocks Rising Overnight?", "BE", "Bloom Energy")


def test_title_cased_names_still_count():
    """ARM's best story on 2026-09-19 was headlined "Arm", not "ARM"."""
    assert mentions_company(
        "SoftBank Raises Arm Margin Loan to $25 Billion as AI Bets Grow", "ARM", "Arm Holdings plc"
    )


def test_the_body_alone_does_not_qualify_an_item():
    """Every "10 stocks to buy" piece names ten companies in its body."""
    assert not mentions_company("10 Chip Stocks for the Next Decade", "AVGO", "Broadcom Inc.")


def test_syndication_ranks_below_real_reporting():
    mill = _item("Broadcom announces new networking chip", publisher="Motley Fool")
    wire = _item("Broadcom announces new networking chip family", publisher="Reuters")
    assert rank_news([mill, wire], "AVGO", "Broadcom Inc.")[0]["publisher"] == "Reuters"


def test_company_terms_skip_legal_suffixes():
    assert company_terms("AVGO", "Broadcom Inc.") == ["avgo", "broadcom"]
    assert "group" not in company_terms("ARM", "Arm Holdings plc Group")


def test_reranker_falls_back_to_the_ranking_not_feed_order():
    """A model reply with no JSON array (seen for MRVL on 2026-09-18) used
    to hand the email whatever order the feed returned."""
    reranker = NewsReranker.__new__(NewsReranker)
    reranker.agent = SimpleNamespace(run=lambda prompt: SimpleNamespace(content="sorry, no idea"))
    top = reranker.rerank(CANDIDATES, "AVGO", "Broadcom Inc.")
    assert "earnings" in top[0]["title"]
    assert top != CANDIDATES[:5]


# --- one rerank call for the whole portfolio --------------------------------


def _batch_reranker(reply):
    r = NewsReranker.__new__(NewsReranker)
    r.batch_agent = SimpleNamespace(run=lambda prompt: SimpleNamespace(content=reply))
    return r


def test_batch_ranks_every_holding_in_one_call():
    candidates = {"AVGO": CANDIDATES, "TSLA": [_item(f"Tesla item {i}") for i in range(6)]}
    names = {"AVGO": "Broadcom Inc.", "TSLA": "Tesla, Inc."}
    out = _batch_reranker('{"AVGO": [5, 6], "TSLA": [2]}').rerank_batch(candidates, names)
    assert [i["title"] for i in out["AVGO"]] == [CANDIDATES[5]["title"], CANDIDATES[6]["title"]]
    assert [i["title"] for i in out["TSLA"]] == ["Tesla item 2"]


def test_batch_keeps_an_empty_answer():
    """ "None of these are about NVDA" is the right answer on some days, not
    a parse failure to fall back from."""
    out = _batch_reranker('{"NVDA": []}').rerank_batch(
        {"NVDA": [_item(f"NVIDIA item {i}") for i in range(6)]}, {"NVDA": "NVIDIA"}
    )
    assert out["NVDA"] == []


def test_batch_falls_back_per_ticker_not_for_everyone():
    """A reply that covers one stock and garbles another leaves the second
    on the deterministic ranking rather than dropping its news."""
    candidates = {"AVGO": CANDIDATES, "TSLA": [_item(f"Tesla item {i}") for i in range(6)]}
    out = _batch_reranker('{"AVGO": [5], "TSLA": "oops"}').rerank_batch(
        candidates, {"AVGO": "Broadcom Inc.", "TSLA": "Tesla, Inc."}
    )
    assert [i["title"] for i in out["AVGO"]] == [CANDIDATES[5]["title"]]
    assert len(out["TSLA"]) == 5  # ranked in code


def test_batch_falls_back_when_the_model_fails():
    r = NewsReranker.__new__(NewsReranker)
    r.batch_agent = SimpleNamespace(
        run=lambda prompt: (_ for _ in ()).throw(RuntimeError("overloaded"))
    )
    out = r.rerank_batch({"AVGO": CANDIDATES}, {"AVGO": "Broadcom Inc."})
    assert "earnings" in out["AVGO"][0]["title"]
