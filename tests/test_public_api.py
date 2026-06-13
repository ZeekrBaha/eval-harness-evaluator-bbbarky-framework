"""The package's public surface and the HarnessEvaluator facade."""

import eval_harness_evaluator_bbbarky_framework as pkg
from eval_harness_evaluator_bbbarky_framework import (
    HarnessEvaluator,
    Invocation,
    LabelMatchEvaluator,
    EvalResult,
    Evaluator,
)
from eval_harness_evaluator_bbbarky_framework.models.core import PerInvocationResult


def test_public_exports_present():
    for name in ("HarnessEvaluator", "Evaluator", "run_suite", "EvalSet", "LabelMatchEvaluator"):
        assert hasattr(pkg, name), f"missing public export: {name}"


async def test_harness_evaluator_runs_named_bundled_judges():
    class FakeClient:
        async def generate(self, messages, **opts):
            return '{"label": "5", "rationale": "ok"}'

    harness = HarnessEvaluator(FakeClient())
    results = await harness.evaluate(
        [Invocation(user_input="why", final_response="because")],
        ["relevance", "coherence"],
    )
    assert set(results) == {"relevance", "coherence"}
    assert results["relevance"].passed is True
    assert results["coherence"].score == 1.0


async def test_evaluate_accepts_pre_built_evaluator():
    """evaluate() should accept pre-built evaluator instances via evaluators=."""
    harness = HarnessEvaluator(model_client=None)
    ev = LabelMatchEvaluator()
    invocations = [Invocation(user_input="q", final_response="correct response")]

    results = await harness.evaluate(
        invocations,
        evaluators={"label": ev},
        expected="correct response",
    )

    assert "label" in results
    assert results["label"].passed is True


async def test_evaluate_accepts_evaluators_and_judge_names_together():
    """evaluate() should accept both evaluators= and judge_names= in the same call."""

    class FakeClient:
        async def generate(self, messages, **opts):
            return '{"label": "5", "rationale": "ok"}'

    harness = HarnessEvaluator(FakeClient())
    ev = LabelMatchEvaluator()
    invocations = [Invocation(user_input="q", final_response="correct response")]

    results = await harness.evaluate(
        invocations,
        judge_names=["relevance"],
        evaluators={"label": ev},
        expected="correct response",
    )

    assert "relevance" in results
    assert "label" in results


async def test_evaluate_passes_expected_to_evaluators():
    """When expected is provided, it must reach the evaluator unchanged."""

    recorded: list[object] = []

    class RecordingEvaluator(Evaluator):
        metric_name = "recording"

        async def evaluate_invocations(
            self, invocations: list[Invocation], expected: object | None
        ) -> EvalResult:
            recorded.append(expected)
            return EvalResult(
                score=1.0,
                passed=True,
                per_invocation=[
                    PerInvocationResult(score=1.0, passed=True) for _ in invocations
                ],
            )

    harness = HarnessEvaluator(model_client=None)
    invocations = [Invocation(user_input="q", final_response="r")]

    await harness.evaluate(
        invocations,
        evaluators={"rec": RecordingEvaluator()},
        expected="sentinel_value",
    )

    assert len(recorded) == 1
    assert recorded[0] == "sentinel_value"
