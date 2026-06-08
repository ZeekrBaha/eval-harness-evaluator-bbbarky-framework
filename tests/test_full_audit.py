"""Rows keep the FULL per-invocation audit trail, not only a representative."""

from eval_harness_evaluator_bbbarky_framework.evaluators.llm_judge import LlmJudgeEvaluator
from eval_harness_evaluator_bbbarky_framework.judges import configs  # noqa: F401
from eval_harness_evaluator_bbbarky_framework.judges.registry import get_judge
from eval_harness_evaluator_bbbarky_framework.models.core import EvalSet
from eval_harness_evaluator_bbbarky_framework.runner.core import run_suite


class Client:
    async def generate(self, messages, **opts):
        return '{"label": "A - Good", "rationale": "ok", "confidence": 0.7}'


async def test_row_preserves_all_per_invocation_details():
    evalset = EvalSet.from_dict(
        {
            "name": "multi",
            "cases": [
                {
                    "id": "c1",
                    "invocations": [
                        {"user_input": "q1", "final_response": "a1", "context": ["x"]},
                        {"user_input": "q2", "final_response": "a2", "context": ["y"]},
                    ],
                    "expected": None,
                }
            ],
        }
    )
    ev = LlmJudgeEvaluator(judge=get_judge("groundedness"), model_client=Client())
    rows = await run_suite(evalset, {"groundedness": ev})

    row = rows[0]
    # full audit trail: one details dict per invocation
    assert len(row["per_invocation"]) == 2
    assert all(pi["confidence"] == 0.7 for pi in row["per_invocation"])
    assert all(pi["rubric_version"] == "v1" for pi in row["per_invocation"])
    # representative summary still present for human-readable reports
    assert "details" in row


async def test_deterministic_row_has_per_invocation_list_too():
    from eval_harness_evaluator_bbbarky_framework.evaluators.label_match import LabelMatchEvaluator

    evalset = EvalSet.from_dict(
        {
            "name": "det",
            "cases": [
                {"id": "c1", "invocations": [{"user_input": "q", "final_response": "Sales"}],
                 "expected": "Sales"}
            ],
        }
    )
    rows = await run_suite(evalset, {"label_match": LabelMatchEvaluator()})
    assert rows[0]["per_invocation"][0]["actual"] == "Sales"
