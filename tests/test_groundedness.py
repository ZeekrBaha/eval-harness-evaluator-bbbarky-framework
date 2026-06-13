"""Context-aware judging: the groundedness judge sees retrieved context."""

from eval_harness_evaluator_bbbarky_framework.evaluators.llm_judge import LlmJudgeEvaluator
from eval_harness_evaluator_bbbarky_framework.judges import configs  # noqa: F401
from eval_harness_evaluator_bbbarky_framework.judges.registry import get_judge
from eval_harness_evaluator_bbbarky_framework.models.core import Invocation


def test_groundedness_judge_is_registered():
    judge = get_judge("groundedness")
    assert judge.name == "groundedness"


async def test_context_reaches_the_judge_prompt():
    captured = {}

    class FakeClient:
        async def generate(self, messages, **opts):
            captured["user"] = messages[1]["content"]
            return '{"label": "5", "rationale": "supported"}'

    judge = get_judge("groundedness")
    ev = LlmJudgeEvaluator(judge=judge, model_client=FakeClient())
    inv = Invocation(
        user_input="What is the refund window?",
        final_response="30 days.",
        context=["Refunds are accepted within 30 days of purchase."],
    )
    result = await ev.evaluate_invocations([inv], expected=None)
    assert result.passed is True
    assert "30 days of purchase" in captured["user"]


async def test_missing_context_renders_empty_not_error():
    class FakeClient:
        async def generate(self, messages, **opts):
            return '{"label": "B - Bad", "rationale": "no support"}'

    judge = get_judge("groundedness")
    ev = LlmJudgeEvaluator(judge=judge, model_client=FakeClient())
    inv = Invocation(user_input="q", final_response="a")  # no context
    result = await ev.evaluate_invocations([inv], expected=None)
    assert result.passed is False
