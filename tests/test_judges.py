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


def test_parse_ignores_trailing_brace_tokens_after_json():
    """Greedy DOTALL regex fails when prose after JSON contains '}' (e.g. '{v1}').

    The response '{"label": "A - Good", "rationale": "ok"} See rubric {v1}' has
    two closing braces. The old r'\\{.*\\}' regex (DOTALL, greedy) captures the
    whole string — from the first '{' to the LAST '}' — producing invalid JSON
    and silently returning score=0.0. The fix must return score=1.0.
    """
    judge = make_judge()
    raw = '{"label": "A - Good", "rationale": "ok"} See rubric {v1}'
    result = judge.parse(raw)
    assert result.label == "A - Good", f"Expected 'A - Good' but got {result.label!r}"
    assert result.score == 1.0, f"Expected score 1.0 but got {result.score}"
    assert result.rationale == "ok"


# --- Numeric label scoring (1-5 rubric) ---


def test_numeric_label_5_gives_score_1_0():
    """Label '5' must produce score 1.0 (5 / 5)."""
    judge = make_judge(passing_labels=["4", "5"])
    raw = '{"label": "5", "rationale": "perfect", "score": 5}'
    result = judge.parse(raw)
    assert result.label == "5"
    assert result.score == 1.0


def test_numeric_label_1_gives_score_0_2():
    """Label '1' must produce score 0.2 (1 / 5)."""
    judge = make_judge(passing_labels=["4", "5"])
    raw = '{"label": "1", "rationale": "irrelevant"}'
    result = judge.parse(raw)
    assert result.label == "1"
    assert result.score == pytest.approx(0.2)


def test_numeric_label_3_gives_score_0_6():
    """Label '3' must produce score 0.6 (3 / 5)."""
    judge = make_judge(passing_labels=["4", "5"])
    raw = '{"label": "3", "rationale": "partial"}'
    result = judge.parse(raw)
    assert result.label == "3"
    assert result.score == pytest.approx(0.6)


def test_relevance_judge_passing_labels_are_4_and_5():
    """Upgraded relevance judge must have passing_labels=['4', '5']."""
    from eval_harness_evaluator_bbbarky_framework.judges.configs.relevance import (
        make_relevance_judge,
    )

    judge = make_relevance_judge()
    assert judge.passing_labels == ["4", "5"]


def test_relevance_judge_has_rubric_version_v2():
    """Upgraded relevance judge must carry rubric_version='v2'."""
    from eval_harness_evaluator_bbbarky_framework.judges.configs.relevance import (
        make_relevance_judge,
    )

    judge = make_relevance_judge()
    assert judge.rubric_version == "v2"
