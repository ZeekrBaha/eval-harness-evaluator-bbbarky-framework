"""LLM-as-judge evaluator: bridges a BaseJudge to a ModelClient."""

from __future__ import annotations

from collections.abc import Callable

from ..judges.base_judge import BaseJudge
from ..models.core import EvalResult, Invocation, PerInvocationResult
from .base import Evaluator


def _render_context(context: object | None) -> str:
    if context is None:
        return ""
    if isinstance(context, (list, tuple)):
        return "\n".join(str(c) for c in context)
    return str(context)


def _default_vars(invocation: Invocation) -> dict:
    return {
        "question": invocation.user_input,
        "answer": invocation.final_response,
        "context": _render_context(invocation.context),
    }


class LlmJudgeEvaluator(Evaluator):
    """Run a judge over each invocation using an injected model client.

    ``var_mapper`` adapts an invocation to the judge's template variables; it
    defaults to ``{"question": user_input, "answer": final_response}``.
    """

    metric_name = "llm_judge"

    def __init__(
        self,
        judge: BaseJudge,
        model_client,
        var_mapper: Callable[[Invocation], dict] | None = None,
        threshold: float = 0.5,
    ) -> None:
        self.judge = judge
        self.model_client = model_client
        self.var_mapper = var_mapper or _default_vars
        self.threshold = threshold

    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None
    ) -> EvalResult:
        per: list[PerInvocationResult] = []
        for invocation in invocations:
            verdict = await self.judge.evaluate(self.model_client, **self.var_mapper(invocation))
            per.append(
                PerInvocationResult(
                    score=verdict.score,
                    passed=verdict.score >= self.threshold,
                    details={
                        "label": verdict.label,
                        "rationale": verdict.rationale,
                        "issues": verdict.issues,
                    },
                )
            )
        return self._aggregate(per)
