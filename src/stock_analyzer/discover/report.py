"""Output renderers — HTML email body, PDF report, terminal summary.

This module is a thin public-surface shim. The actual implementation
lives in three siblings:

  - `report_sections`: Section IR + LLM-output parsers + shared palettes
  - `report_html`:     HTML renderer (email body)
  - `report_pdf`:      PDF renderer (email attachment)

Both renderers consume the same `Section` list so layout stays in sync,
and they share palettes from `report_sections` so colors stay identical.

Existing callers (cli/discover.py, cli/rebalance.py, tests) import from
this module — keep the re-exports stable so the split is invisible to them.
"""
from __future__ import annotations

from .report_html import render_html_email
from .report_pdf import render_pdf
from .report_sections import (
    Section,
    SectionKind,
    build_sections,
    parse_actions,
    parse_confidence,
    parse_picks,
    parse_rebalance_status,
    parse_verdict,
    print_terminal_summary,
)

__all__ = [
    "Section",
    "SectionKind",
    "build_sections",
    "parse_actions",
    "parse_confidence",
    "parse_picks",
    "parse_rebalance_status",
    "parse_verdict",
    "print_terminal_summary",
    "render_html_email",
    "render_pdf",
]
