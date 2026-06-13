"""Tests for the hallucination judge."""

from eval_harness_evaluator_bbbarky_framework.judges.configs.hallucination import make_hallucination_judge
from eval_harness_evaluator_bbbarky_framework.judges.registry import JudgeRegistry, seed_default_judges


def test_hallucination_judge_name_is_hallucination():
    judge = make_hallucination_judge()
    assert judge.name == "hallucination"


def test_hallucination_judge_passing_labels_are_4_and_5():
    judge = make_hallucination_judge()
    assert judge.passing_labels == ["4", "5"]


def test_hallucination_judge_in_seed_default_judges():
    registry = seed_default_judges(JudgeRegistry())
    judge = registry.get("hallucination")
    assert judge.name == "hallucination"
