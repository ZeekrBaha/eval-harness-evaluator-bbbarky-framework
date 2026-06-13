"""Suite runner: evaluate every case with every evaluator into flat rows.

Rows are intentionally plain dicts so the report layer can aggregate them
without importing evaluator types. Each row is one (case, evaluator) pair.
"""

from __future__ import annotations

import asyncio

from ..evaluators.base import Evaluator
from ..models.core import EvalSet


async def run_suite(
    evalset: EvalSet,
    evaluators: dict[str, Evaluator],
    max_concurrency: int = 20,
) -> list[dict]:
    """Run all evaluators over all cases; return one result row per pair."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _evaluate_pair(case, metric_name, evaluator):
        async with semaphore:
            result = await evaluator.evaluate_invocations(case.invocations, case.expected)
        return {
            "id": case.id,
            "metric": metric_name,
            "kind": case.metadata.get("kind", "all"),
            "lang": case.metadata.get("lang", "all"),
            "success": result.passed,
            "score": result.score,
            "details": _representative_details(result),
            "per_invocation": [
                {"score": p.score, "passed": p.passed, **p.details}
                for p in result.per_invocation
            ],
        }

    tasks = [
        _evaluate_pair(case, metric_name, evaluator)
        for case in evalset.cases
        for metric_name, evaluator in evaluators.items()
    ]
    return list(await asyncio.gather(*tasks))


def _representative_details(result) -> dict:
    """Pick the most informative per-invocation details for the row summary.

    Prefer the first failing invocation (the useful one for auditing), else the
    first. Empty for evaluators that record no per-invocation details. The full
    trail is preserved separately under the row's ``per_invocation`` key.
    """
    per = result.per_invocation
    if not per:
        return {}
    chosen = next((p for p in per if not p.passed), per[0])
    return dict(chosen.details)
