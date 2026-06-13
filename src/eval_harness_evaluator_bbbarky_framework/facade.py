"""HarnessEvaluator: a one-call facade for running named judges.

Mirrors the ergonomic "evaluate a session with a list of judges" entry point,
but provider-agnostic: it takes any :class:`ModelClient` and returns one
:class:`EvalResult` per requested judge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .evaluators.llm_judge import LlmJudgeEvaluator
from .judges import configs as _configs  # noqa: F401  (registers bundled judges on import)
from .judges.registry import get_judge
from .models.core import EvalResult, Invocation

if TYPE_CHECKING:
    from .evaluators.base import Evaluator


class HarnessEvaluator:
    """Run one or more named judges against a list of invocations."""

    def __init__(self, model_client, threshold: float = 0.5) -> None:
        self.model_client = model_client
        self.threshold = threshold

    async def evaluate(
        self,
        invocations: list[Invocation],
        judge_names: list[str] | None = None,
        evaluators: dict[str, "Evaluator"] | None = None,
        expected: object | None = None,
    ) -> dict[str, EvalResult]:
        """Evaluate invocations with judges and/or pre-built evaluators.

        judge_names: names of judges to look up in the registry
        evaluators: pre-built {metric_name: Evaluator} instances
        expected: passed to all evaluators as the expected value
        """
        all_evaluators: dict[str, Any] = {}

        # Add registry-sourced judge evaluators
        for name in (judge_names or []):
            judge = get_judge(name)
            all_evaluators[name] = LlmJudgeEvaluator(
                judge=judge,
                model_client=self.model_client,
                threshold=self.threshold,
            )

        # Add pre-built evaluators
        for name, ev in (evaluators or {}).items():
            all_evaluators[name] = ev

        results: dict[str, EvalResult] = {}
        for name, ev in all_evaluators.items():
            results[name] = await ev.evaluate_invocations(invocations, expected)
        return results
