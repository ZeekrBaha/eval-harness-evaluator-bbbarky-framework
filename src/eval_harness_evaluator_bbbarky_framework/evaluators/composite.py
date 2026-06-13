"""Composite evaluator: combine several evaluators (logical AND)."""

from __future__ import annotations

import asyncio

from ..models.core import EvalResult, Invocation
from .base import Evaluator


class CompositeEvaluator(Evaluator):
    """Pass only when every sub-evaluator passes; score is their mean."""

    metric_name = "composite"

    def __init__(self, evaluators: list[Evaluator]) -> None:
        if not evaluators:
            raise ValueError("CompositeEvaluator requires at least one evaluator")
        self.evaluators = evaluators

    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None
    ) -> EvalResult:
        sub_results = list(await asyncio.gather(
            *[ev.evaluate_invocations(invocations, expected) for ev in self.evaluators]
        ))
        score = sum(r.score for r in sub_results) / len(sub_results)
        passed = all(r.passed for r in sub_results)
        return EvalResult(
            score=score,
            passed=passed,
            details={"sub_results": [r.model_dump() for r in sub_results]},
        )
