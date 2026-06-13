"""Build a report object from result rows."""

from __future__ import annotations

import datetime
import uuid

from .aggregate import summarize

SCHEMA_VERSION = "1.1"


def build_report(suite: str, rows: list[dict]) -> dict:
    """Assemble a report: suite name, summary, and the raw rows."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "run_id": str(uuid.uuid4()),
        "suite": suite,
        "summary": summarize(rows),
        "rows": rows,
    }
