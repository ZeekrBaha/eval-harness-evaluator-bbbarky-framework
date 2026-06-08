"""Build a report object from result rows."""

from __future__ import annotations

from .aggregate import summarize


def build_report(suite: str, rows: list[dict]) -> dict:
    """Assemble a report: suite name, summary, and the raw rows."""
    return {
        "suite": suite,
        "summary": summarize(rows),
        "rows": rows,
    }
