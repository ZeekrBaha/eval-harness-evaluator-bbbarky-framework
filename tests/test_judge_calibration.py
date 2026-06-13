"""Judge calibration: confidence parsing and rubric versioning."""

from eval_harness_evaluator_bbbarky_framework.judges.base_judge import BaseJudge
from eval_harness_evaluator_bbbarky_framework.judges.registry import get_judge
from eval_harness_evaluator_bbbarky_framework.models.core import JudgeResult


def _judge(**over):
    base = dict(
        name="relevance",
        system_prompt="judge",
        user_prompt_template="Q: {question}\nA: {answer}",
        passing_labels=["A - Good"],
    )
    base.update(over)
    return BaseJudge(**base)


def test_judge_result_has_optional_confidence_and_rubric_version():
    jr = JudgeResult(label="A - Good", score=1.0)
    assert jr.confidence is None
    assert jr.rubric_version is None


def test_parse_extracts_confidence_when_present():
    judge = _judge()
    result = judge.parse('{"label": "A - Good", "rationale": "ok", "confidence": 0.82}')
    assert result.confidence == 0.82


def test_parse_confidence_absent_is_none():
    judge = _judge()
    result = judge.parse('{"label": "A - Good", "rationale": "ok"}')
    assert result.confidence is None


def test_rubric_version_is_stamped_into_result():
    judge = _judge(rubric_version="v3")
    result = judge.parse('{"label": "A - Good"}')
    assert result.rubric_version == "v3"


def test_groundedness_judge_carries_rubric_version():
    from eval_harness_evaluator_bbbarky_framework.judges import configs  # noqa: F401

    judge = get_judge("groundedness")
    assert judge.rubric_version == "v2"
