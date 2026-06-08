"""HarnessEvaluator: a one-call facade for running named judges.

Mirrors the ergonomic "evaluate a session with a list of judges" entry point,
but provider-agnostic: it takes any :class:`ModelClient` and returns one
:class:`EvalResult` per requested judge.
"""

from __future__ import annotations

from .evaluators.llm_judge import LlmJudgeEvaluator
from .judges import configs as _configs  # noqa: F401  (registers bundled judges on import)
from .judges.registry import get_judge
from .models.core import EvalResult, Invocation


class HarnessEvaluator:
    """Run one or more named judges against a list of invocations."""

    def __init__(self, model_client, threshold: float = 0.5) -> None:
        self.model_client = model_client
        self.threshold = threshold

    async def evaluate(
        self, invocations: list[Invocation], judge_names: list[str]
    ) -> dict[str, EvalResult]:
        results: dict[str, EvalResult] = {}
        for name in judge_names:
            judge = get_judge(name)
            evaluator = LlmJudgeEvaluator(
                judge=judge, model_client=self.model_client, threshold=self.threshold
            )
            results[name] = await evaluator.evaluate_invocations(invocations, expected=None)
        return results
