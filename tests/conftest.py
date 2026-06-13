import pytest
from eval_harness_evaluator_bbbarky_framework.judges.base_judge import BaseJudge
from eval_harness_evaluator_bbbarky_framework.models.core import Invocation


class _FakeClient:
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list | None = None
        self.call_count: int = 0

    async def generate(self, messages: list, **opts) -> str:
        self.last_messages = messages
        self.call_count += 1
        return self._response


@pytest.fixture
def fake_good_client():
    return _FakeClient('{"score": 5, "label": "5", "rationale": "supported", "confidence": 0.9, "issues": []}')


@pytest.fixture
def fake_bad_client():
    return _FakeClient('{"score": 1, "label": "1", "issues": ["hallucinated"], "rationale": "wrong", "confidence": 0.8}')


@pytest.fixture
def make_judge():
    def _make(**overrides):
        defaults = dict(
            name="test_judge",
            system_prompt="You evaluate test quality.",
            user_prompt_template="Input: {question}\nOutput: {answer}",
            passing_labels=["4", "5"],
        )
        defaults.update(overrides)
        return BaseJudge(**defaults)
    return _make


@pytest.fixture
def make_inv():
    def _make(user: str = "test question", response: str = "test answer", **kwargs) -> list[Invocation]:
        return [Invocation(user_input=user, final_response=response, **kwargs)]
    return _make
