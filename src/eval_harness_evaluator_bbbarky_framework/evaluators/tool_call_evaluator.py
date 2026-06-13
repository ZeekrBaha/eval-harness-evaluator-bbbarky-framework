"""Evaluator that checks invocations include a specific tool call with required parameters."""

from __future__ import annotations

from eval_harness_evaluator_bbbarky_framework.models.core import (
    EvalResult,
    Invocation,
    PerInvocationResult,
)
from eval_harness_evaluator_bbbarky_framework.evaluators.base import Evaluator


class ToolCallEvaluator:
    """Checks that invocations include a specific tool call with required parameters."""

    metric_name = "tool_call"

    def __init__(
        self,
        expected_tool: str,
        required_params: list[str] | None = None,
        threshold: float = 1.0,
    ) -> None:
        self.expected_tool = expected_tool
        self.required_params = required_params or []
        self.threshold = threshold

    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None = None
    ) -> EvalResult:
        per: list[PerInvocationResult] = []
        for inv in invocations:
            calls = inv.tool_calls or []
            matched = [c for c in calls if c.get("name") == self.expected_tool]
            if not matched:
                per.append(PerInvocationResult(
                    score=0.0,
                    passed=False,
                    details={
                        "error": f"tool '{self.expected_tool}' not called",
                        "actual_tools": [c.get("name") for c in calls],
                    },
                ))
                continue
            call = matched[0]
            params = call.get("parameters") or call.get("arguments") or {}
            missing = [p for p in self.required_params if p not in params]
            score = 1.0 if not missing else 0.5
            per.append(PerInvocationResult(
                score=score,
                passed=len(missing) == 0,
                details={"missing_params": missing, "matched_call": call},
            ))

        if not per:
            return EvalResult(score=0.0, passed=False, per_invocation=[])

        avg_score = sum(p.score for p in per) / len(per)
        return EvalResult(
            score=avg_score,
            passed=avg_score >= self.threshold,
            per_invocation=per,
        )
