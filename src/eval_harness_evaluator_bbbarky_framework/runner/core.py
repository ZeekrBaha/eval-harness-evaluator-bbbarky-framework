"""Suite runner: evaluate every case with every evaluator into flat rows.

Rows are intentionally plain dicts so the report layer can aggregate them
without importing evaluator types. Each row is one (case, evaluator) pair.
"""

from __future__ import annotations

from ..evaluators.base import Evaluator
from ..models.core import EvalSet


async def run_suite(evalset: EvalSet, evaluators: dict[str, Evaluator]) -> list[dict]:
    """Run all evaluators over all cases; return one result row per pair."""
    rows: list[dict] = []
    for case in evalset.cases:
        kind = case.metadata.get("kind", "all")
        lang = case.metadata.get("lang", "all")
        for metric_name, evaluator in evaluators.items():
            result = await evaluator.evaluate_invocations(case.invocations, case.expected)
            rows.append(
                {
                    "id": case.id,
                    "metric": metric_name,
                    "kind": kind,
                    "lang": lang,
                    "success": result.passed,
                    "score": result.score,
                }
            )
    return rows
