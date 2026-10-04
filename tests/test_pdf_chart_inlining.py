"""Charts in the PDF (report_pdf prints the email's HTML with WeasyPrint).

The email references each chart by `cid:`; the PDF embeds the same PNG as
a data URI, keeps it on the page of the pick card above it, and drops a
chart that isn't a usable image rather than failing the whole report.
"""

from __future__ import annotations

from io import BytesIO

from PIL import Image as PILImage

from stock_analyzer.discover.report_html import render_html, render_html_email
from stock_analyzer.discover.report_pdf import _PRINT_CSS, render_pdf
from stock_analyzer.models.reports import Section


def _fake_png(color: tuple[int, int, int] = (10, 20, 30)) -> bytes:
    img = PILImage.new("RGB", (300, 150), color=color)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _card_and_chart(ticker: str) -> list[Section]:
    return [
        Section(kind="pick_card", data={"ticker": ticker, "rank": 1}),
        Section(kind="image", image_ticker=ticker),
    ]


def test_the_email_references_the_attachment_and_the_pdf_embeds_the_png():
    sections = _card_and_chart("NVDA")
    assert "src='cid:chart-NVDA'" in render_html_email(sections, {"NVDA": "chart-NVDA"})
    assert "src='data:image/png;base64," in render_html(
        sections, {"NVDA": "data:image/png;base64,AA"}
    )
    pdf = render_pdf(sections, {"NVDA": _fake_png()})
    assert pdf.startswith(b"%PDF") and b"/Image" in pdf


def test_a_chart_is_kept_on_the_page_of_the_card_above_it():
    assert "break-before: avoid" in _PRINT_CSS.split("img {")[1].split("}")[0]


def test_a_missing_or_broken_chart_never_fails_the_report():
    for charts in ({}, {"NVDA": b""}, {"NVDA": b"not a png"}):
        pdf = render_pdf(_card_and_chart("NVDA"), charts)
        assert pdf.startswith(b"%PDF")


def test_two_cards_each_get_their_own_chart():
    sections = _card_and_chart("NVDA") + _card_and_chart("AMD")
    # Distinct images: WeasyPrint stores identical ones once.
    pdf = render_pdf(sections, {"NVDA": _fake_png(), "AMD": _fake_png((200, 30, 30))})
    assert pdf.count(b"/Subtype /Image") >= 2
