"""The package's public surface and the HarnessEvaluator facade."""

import eval_harness_evaluator_bbbarky_framework as pkg
from eval_harness_evaluator_bbbarky_framework import HarnessEvaluator, Invocation


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
