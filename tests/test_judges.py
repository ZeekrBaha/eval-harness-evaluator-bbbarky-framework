"""Tests for the LLM-as-judge library: BaseJudge, parsing, registry."""

import pytest

from eval_harness_evaluator_bbbarky_framework.judges.base_judge import BaseJudge
from eval_harness_evaluator_bbbarky_framework.judges.registry import (
    JudgeRegistry,
    get_judge,
    register_judge,
)


def make_judge(**overrides):
    defaults = dict(
        name="relevance",
        system_prompt="You evaluate relevance.",
        user_prompt_template="Question: {question}\nAnswer: {answer}",
        passing_labels=["A - Good"],
    )
    defaults.update(overrides)
    return BaseJudge(**defaults)


def test_build_messages_formats_template_and_prepends_system():
    judge = make_judge()
    messages = judge.build_messages(question="why sky blue", answer="rayleigh")
    assert messages[0] == {"role": "system", "content": "You evaluate relevance."}
    assert messages[1]["role"] == "user"
    assert "why sky blue" in messages[1]["content"]
    assert "rayleigh" in messages[1]["content"]


def test_parse_extracts_fields_and_scores_passing_label():
    judge = make_judge()
    raw = '{"label": "A - Good", "issues": [], "rationale": "on topic"}'
    result = judge.parse(raw)
    assert result.label == "A - Good"
    assert result.rationale == "on topic"
    assert result.score == 1.0


def test_parse_scores_non_passing_label_zero():
    judge = make_judge()
    raw = '{"label": "B - Bad", "issues": ["off topic"], "rationale": "wrong"}'
    result = judge.parse(raw)
    assert result.label == "B - Bad"
    assert result.score == 0.0
    assert result.issues == ["off topic"]


def test_parse_handles_fenced_json_with_surrounding_text():
    judge = make_judge()
    raw = 'Sure!\n```json\n{"label": "A - Good", "rationale": "ok"}\n```\nDone.'
    result = judge.parse(raw)
    assert result.label == "A - Good"
    assert result.score == 1.0
    assert result.raw_response == raw


async def test_evaluate_calls_model_client_and_parses():
    class FakeClient:
        async def generate(self, messages, **opts):
            self.seen = messages
            return '{"label": "A - Good", "rationale": "great"}'

    judge = make_judge()
    client = FakeClient()
    result = await judge.evaluate(client, question="q", answer="a")
    assert result.label == "A - Good"
    assert result.score == 1.0
    assert client.seen[0]["role"] == "system"


def test_registry_register_and_get():
    registry = JudgeRegistry()
    judge = make_judge(name="coherence")
    registry.register(judge)
    assert registry.get("coherence") is judge


def test_registry_unknown_name_raises():
    registry = JudgeRegistry()
    with pytest.raises(KeyError):
        registry.get("does-not-exist")


def test_module_level_register_and_get_roundtrip():
    judge = make_judge(name="query_quality")
    register_judge(judge)
    assert get_judge("query_quality") is judge
