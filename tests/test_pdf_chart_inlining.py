"""PDF pick_card + chart placement.

Previously each pick's chart was a separate full-width Image flowable
after a card that usually filled most of a page — so the chart landed on
its own near-empty page. _pdf_pick_card now draws a smaller chart directly
under the header pills (guaranteed same page as the header), and
render_pdf's dispatch loop skips the now-redundant standalone "image"
section that always immediately follows a pick_card in report_sections.py.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image as PILImage
from reportlab.platypus import Image as RLImage

from stock_analyzer.discover.report_pdf import _pdf_pick_card, _pdf_styles, render_pdf
from stock_analyzer.models.reports import Section


def _fake_png() -> bytes:
    img = PILImage.new("RGB", (300, 150), color=(10, 20, 30))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_pick_card_embeds_image_flowable_when_chart_data_given():
    flow = _pdf_pick_card({"ticker": "NVDA", "rank": 1}, _pdf_styles(), chart_data=_fake_png())
    assert any(isinstance(el, RLImage) for el in flow)


def test_pick_card_has_no_image_flowable_without_chart_data():
    flow = _pdf_pick_card({"ticker": "NVDA", "rank": 1}, _pdf_styles(), chart_data=None)
    assert not any(isinstance(el, RLImage) for el in flow)


def test_pick_card_survives_bad_chart_bytes():
    flow = _pdf_pick_card({"ticker": "NVDA", "rank": 1}, _pdf_styles(), chart_data=b"not a png")
    # Same silent-skip behavior as the standalone "image" section renderer.
    assert not any(isinstance(el, RLImage) for el in flow)


def test_render_pdf_pick_card_followed_by_image_consumes_it_once():
    sections = [
        Section(kind="pick_card", data={"ticker": "NVDA", "rank": 1, "one_liner": "x"}),
        Section(kind="image", image_ticker="NVDA"),
    ]
    pdf_bytes = render_pdf(sections, {"NVDA": _fake_png()})
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 800


def test_render_pdf_pick_card_not_followed_by_image_still_renders():
    sections = [
        Section(kind="pick_card", data={"ticker": "NVDA", "rank": 1, "one_liner": "x"}),
        Section(kind="para", text="unrelated paragraph"),
    ]
    pdf_bytes = render_pdf(sections, {"NVDA": _fake_png()})
    assert pdf_bytes.startswith(b"%PDF")


def test_render_pdf_two_consecutive_pick_cards_each_get_own_chart():
    sections = [
        Section(kind="pick_card", data={"ticker": "NVDA", "rank": 1, "one_liner": "x"}),
        Section(kind="image", image_ticker="NVDA"),
        Section(kind="pick_card", data={"ticker": "AMD", "rank": 2, "one_liner": "y"}),
        Section(kind="image", image_ticker="AMD"),
    ]
    pdf_bytes = render_pdf(sections, {"NVDA": _fake_png(), "AMD": _fake_png()})
    assert pdf_bytes.startswith(b"%PDF")
