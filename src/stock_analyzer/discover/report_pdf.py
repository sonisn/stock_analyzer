"""PDF renderer for the report Section IR: the email's own HTML, printed.

The PDF used to be a second renderer (ReportLab) that had to be kept in
step with the HTML one section kind by section kind. It is now the same
document `report_html.render_html` builds for the email, with the charts
embedded as data URIs and a print stylesheet on top, laid out by
WeasyPrint — so the attachment shows exactly what the email shows.
"""

from __future__ import annotations

import base64
import logging

from ..logging import get_logger
from ..models.reports import Section
from .report_html import render_html

logger = get_logger(__name__)

# WeasyPrint reports every CSS property it does not support at WARNING;
# the email stylesheet has a few on purpose (box-shadow on old clients).
logging.getLogger("weasyprint").setLevel(logging.ERROR)
logging.getLogger("fontTools").setLevel(logging.ERROR)

_TITLE = "Stock Discovery"

# On top of the email stylesheet: a Letter page with the old margins, a
# page number, the email's width cap lifted, and the things that read
# badly split across a page kept whole.
_PRINT_CSS = """
@page {
  size: letter;
  margin: 0.7in 0.6in;
  @bottom-right { content: counter(page) " / " counter(pages);
                  font-size: 8pt; color: #9ca3af; }
}
body { max-width: none; margin: 0; padding: 0; background: #fff; font-size: 10pt; }
hr.page-break { break-after: page; border: none; margin: 0; }
h1, h2, h3 { break-after: avoid; }
tr, .banner, .metrics, .metric, .badge, .pie-wrap, .chart, svg, blockquote {
  break-inside: avoid;
}
img { max-height: 3.4in; break-inside: avoid; break-before: avoid; }
pre { white-space: pre-wrap; overflow-x: visible; }
/* The email's tile minimum wraps a 4th metric; a Letter page fits five. */
.metric { min-width: 0; }
/* WeasyPrint stacks inline-flex legend entries; keep them on one line. */
.chart span { display: inline-block !important; margin-right: 14px; }
.chart span svg { vertical-align: middle; margin-right: 4px; }
"""


def render_pdf(sections: list[Section], chart_bytes: dict[str, bytes]) -> bytes:
    """The report as PDF bytes. A chart that is not a usable image is left
    out of the page rather than failing it."""
    from weasyprint import HTML

    image_src = {
        ticker: "data:image/png;base64," + base64.b64encode(png).decode()
        for ticker, png in chart_bytes.items()
        if png
    }
    # In the document, after the email's own <style>, so it wins (a
    # stylesheet handed to write_pdf ranks below the document's styles).
    document = render_html(sections, image_src).replace(
        "</style></head>", f"</style><style>{_PRINT_CSS}</style><title>{_TITLE}</title></head>", 1
    )
    return HTML(string=document).write_pdf()
