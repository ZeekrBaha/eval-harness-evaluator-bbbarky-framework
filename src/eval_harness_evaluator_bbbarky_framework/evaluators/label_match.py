"""Deterministic label-classification evaluator."""

from __future__ import annotations

from ..models.core import EvalResult, Invocation, PerInvocationResult
from .base import Evaluator


def _expected_set(expected: object | None) -> set[str]:
    if expected is None:
        return set()
    if isinstance(expected, (list, tuple, set)):
        return {str(e).strip().lower() for e in expected}
    return {str(expected).strip().lower()}


class LabelMatchEvaluator(Evaluator):
    """Pass when the final response matches one of the expected labels."""

    metric_name = "label_match"

    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None
    ) -> EvalResult:
        wanted = _expected_set(expected)
        per: list[PerInvocationResult] = []
        for invocation in invocations:
            actual = invocation.final_response.strip().lower()
            hit = actual in wanted
            per.append(
                PerInvocationResult(
                    score=1.0 if hit else 0.0,
                    passed=hit,
                    details={"actual": invocation.final_response},
                )
            )
        return self._aggregate(per)
