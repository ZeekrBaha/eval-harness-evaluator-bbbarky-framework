"""The bundled generic judges should self-register on import."""

from eval_harness_evaluator_bbbarky_framework.judges import configs  # noqa: F401
from eval_harness_evaluator_bbbarky_framework.judges.registry import get_judge


def test_relevance_judge_is_registered():
    judge = get_judge("relevance")
    assert judge.name == "relevance"
    assert judge.passing_labels  # has at least one passing label


def test_coherence_judge_is_registered():
    judge = get_judge("coherence")
    assert judge.name == "coherence"


def test_bundled_judge_parses_a_passing_verdict():
    # The top numeric label "5" must produce the maximum score of 1.0.
    judge = get_judge("relevance")
    result = judge.parse('{"label": "5", "rationale": "ok"}')
    assert result.score == 1.0
