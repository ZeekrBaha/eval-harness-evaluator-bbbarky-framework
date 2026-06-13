"""Judge auditability: confidence/rubric/issues surface into rows and reports."""

from eval_harness_evaluator_bbbarky_framework.evaluators.llm_judge import LlmJudgeEvaluator
from eval_harness_evaluator_bbbarky_framework.judges import configs  # noqa: F401
from eval_harness_evaluator_bbbarky_framework.judges.registry import get_judge
from eval_harness_evaluator_bbbarky_framework.models.core import EvalSet, Invocation
from eval_harness_evaluator_bbbarky_framework.report.reporters import to_markdown
from eval_harness_evaluator_bbbarky_framework.runner.core import run_suite


class GoodClient:
    async def generate(self, messages, **opts):
        return '{"label": "5", "rationale": "supported", "confidence": 0.9}'


class BadClient:
    async def generate(self, messages, **opts):
        return '{"label": "B - Bad", "issues": ["unsupported claim"], "rationale": "hallucinated"}'


async def test_llm_judge_details_include_confidence_and_rubric_version():
    ev = LlmJudgeEvaluator(judge=get_judge("groundedness"), model_client=GoodClient())
    result = await ev.evaluate_invocations(
        [Invocation(user_input="q", final_response="a", context=["a"])], expected=None
    )
    details = result.per_invocation[0].details
    assert details["confidence"] == 0.9
    assert details["rubric_version"] == "v2"


async def test_run_suite_row_surfaces_judge_details():
    evalset = EvalSet.from_dict(
        {
            "name": "rag",
            "cases": [
                {
                    "id": "c1",
                    "invocations": [{"user_input": "q", "final_response": "a", "context": ["a"]}],
                    "expected": None,
                },
            ],
        }
    )
    ev = LlmJudgeEvaluator(judge=get_judge("groundedness"), model_client=BadClient())
    rows = await run_suite(evalset, {"groundedness": ev})
    assert rows[0]["success"] is False
    assert rows[0]["details"]["rationale"] == "hallucinated"
    assert rows[0]["details"]["issues"] == ["unsupported claim"]


def test_markdown_includes_failure_examples_with_rationale():
    report = {
        "suite": "rag",
        "summary": {
            "n": 1,
            "passed": 0,
            "failed": 1,
            "pass_rate": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.793,
            "by_metric": {
                "groundedness": {
                    "n": 1,
                    "passed": 0,
                    "failed": 1,
                    "pass_rate": 0.0,
                    "ci_low": 0.0,
                    "ci_high": 0.793,
                }
            },
        },
        "rows": [
            {
                "id": "c1",
                "metric": "groundedness",
                "kind": "rag",
                "lang": "all",
                "success": False,
                "score": 0.0,
                "details": {"rationale": "hallucinated", "issues": ["unsupported claim"]},
            },
        ],
    }
    md = to_markdown(report)
    assert "Failure" in md
    assert "hallucinated" in md
    assert "c1" in md
