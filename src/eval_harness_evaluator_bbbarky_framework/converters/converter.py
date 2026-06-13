"""Convert legacy row data and CSV files into an EvalSet."""

from __future__ import annotations

import csv

from ..models.core import EvalCase, EvalSet, Invocation

# Columns consumed directly; anything else becomes case metadata.
_CORE_COLUMNS = {"id", "input", "response", "expected", "latency_ms", "trace_id"}


def _str_or_empty(value: object) -> str:
    return "" if value is None else str(value)


def rows_to_evalset(suite: str, rows: list[dict]) -> EvalSet:
    """Build an EvalSet from a list of flat row dicts.

    Recognized keys: ``id``, ``input``, ``response``, ``expected``,
    ``latency_ms``, ``trace_id``. Any other keys (e.g. ``kind``, ``lang``)
    are stored as case metadata.
    """
    cases: list[EvalCase] = []
    for index, row in enumerate(rows):
        metadata = {k: v for k, v in row.items() if k not in _CORE_COLUMNS and v not in (None, "")}
        latency_raw = row.get("latency_ms")
        trace_id_raw = row.get("trace_id")
        cases.append(
            EvalCase(
                id=str(row.get("id") or f"case-{index}"),
                invocations=[
                    Invocation(
                        user_input=_str_or_empty(row.get("input")),
                        final_response=_str_or_empty(row.get("response")),
                        latency_ms=float(latency_raw) if latency_raw not in (None, "") else None,
                        trace_id=_str_or_empty(trace_id_raw) if trace_id_raw is not None else None,
                    )
                ],
                expected=row.get("expected"),
                metadata=metadata,
            )
        )
    return EvalSet(name=suite, cases=cases)


def csv_to_evalset(path: str, suite: str) -> EvalSet:
    """Read a CSV with id/input/response/expected columns into an EvalSet."""
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows_to_evalset(suite, rows)
