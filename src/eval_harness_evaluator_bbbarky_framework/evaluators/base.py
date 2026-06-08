"""Evaluator base class and per-invocation aggregation.

All evaluators are async so the runner can treat deterministic and
LLM-backed evaluators uniformly. A deterministic evaluator simply never awaits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models.core import EvalResult, Invocation, PerInvocationResult


class Evaluator(ABC):
    """Score a sequence of invocations against an expected outcome."""

    metric_name: str = "evaluator"
    threshold: float = 0.5

    @abstractmethod
    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None
    ) -> EvalResult:
        """Return an :class:`EvalResult` for the given invocations."""
        raise NotImplementedError

    def _aggregate(self, per_invocation: list[PerInvocationResult]) -> EvalResult:
        """Mean per-invocation score; pass when the mean meets the threshold."""
        if not per_invocation:
            return EvalResult(score=0.0, passed=False, per_invocation=[])
        score = sum(p.score for p in per_invocation) / len(per_invocation)
        return EvalResult(
            score=score,
            passed=score >= self.threshold,
            per_invocation=per_invocation,
        )
