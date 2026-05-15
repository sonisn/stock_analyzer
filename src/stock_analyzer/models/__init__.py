"""Pydantic domain models for the stock analyzer pipeline.

This package holds the shared data contracts — LLM I/O schemas,
market-data records, portfolio/tax-lot models, track-record models,
and report-section IR — separated from the I/O / orchestration code
that produces and consumes them.

Callsites import the specific submodule they need, e.g.
``from stock_analyzer.models.llm import RankerOutput``. There are no
re-exports at the package root so the dependency graph stays explicit.
"""
