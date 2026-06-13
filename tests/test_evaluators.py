"""Tests for the evaluator hierarchy."""

import pytest

from eval_harness_evaluator_bbbarky_framework.evaluators.base import Evaluator
from eval_harness_evaluator_bbbarky_framework.evaluators.composite import CompositeEvaluator
from eval_harness_evaluator_bbbarky_framework.evaluators.json_schema import JsonSchemaEvaluator
from eval_harness_evaluator_bbbarky_framework.evaluators.label_match import LabelMatchEvaluator
from eval_harness_evaluator_bbbarky_framework.evaluators.llm_judge import LlmJudgeEvaluator
from eval_harness_evaluator_bbbarky_framework.evaluators.response_scorers import (
    ContainsKeywordsScorer,
    ExactMatchScorer,
    FuzzyF1Scorer,
    ScorerEvaluator,
)
from eval_harness_evaluator_bbbarky_framework.judges.base_judge import BaseJudge
from eval_harness_evaluator_bbbarky_framework.models.core import EvalResult, Invocation


def inv(user, response):
    return [Invocation(user_input=user, final_response=response)]


# --- base contract ------------------------------------------------------


def test_evaluator_is_abstract():
    with pytest.raises(TypeError):
        Evaluator()  # type: ignore[abstract]


# --- label match --------------------------------------------------------


@pytest.mark.parametrize(
    "response, expected, passed, score",
    [
        ("Refund Agent", "Refund Agent", True, 1.0),
        ("refund agent", "Refund Agent", True, None),
        ("Sales", ["Refund Agent", "Sales"], True, None),
        ("Sales", "Refund Agent", False, 0.0),
    ],
    ids=[
        "passes_on_exact_label",
        "is_case_insensitive",
        "accepts_list_of_expected",
        "fails_on_mismatch",
    ],
)
async def test_label_match(response, expected, passed, score):
    ev = LabelMatchEvaluator()
    result = await ev.evaluate_invocations(inv("q", response), expected=expected)
    assert isinstance(result, EvalResult)
    assert result.passed is passed
    if score is not None:
        assert result.score == score


# --- scorers ------------------------------------------------------------


def test_exact_match_scorer():
    s = ExactMatchScorer()
    assert s.score("Hello", "hello") == 1.0
    assert s.score("Hello", "world") == 0.0


def test_contains_keywords_scorer_fraction():
    s = ContainsKeywordsScorer(["alpha", "beta"])
    assert s.score("alpha only", None) == 0.5
    assert s.score("alpha and beta", None) == 1.0


def test_fuzzy_f1_scorer_identical_is_one():
    s = FuzzyF1Scorer()
    assert s.score("the quick brown fox", "the quick brown fox") == pytest.approx(1.0)


def test_fuzzy_f1_scorer_disjoint_is_zero():
    s = FuzzyF1Scorer()
    assert s.score("alpha", "omega") == 0.0


def test_fuzzy_f1_handles_repeated_tokens_correctly():
    # "cat cat cat" vs "cat dog": with multiset F1
    # pred tokens = [cat, cat, cat], ref tokens = [cat, dog]
    # overlap = min(3,1) for cat = 1; precision = 1/3, recall = 1/2
    # F1 = 2 * (1/3) * (1/2) / (1/3 + 1/2) = (1/3) / (5/6) = 2/5 = 0.40
    s = FuzzyF1Scorer()
    assert s.score("cat cat cat", "cat dog") == pytest.approx(0.40, abs=1e-6)


def test_contains_keywords_no_substring_false_positive():
    # keyword "is" should NOT match inside "this" or "crisis"
    s = ContainsKeywordsScorer(["is"])
    assert s.score("this crisis", None) == 0.0


async def test_scorer_evaluator_uses_scorer_and_threshold():
    ev = ScorerEvaluator(ExactMatchScorer(), threshold=0.5)
    passed = await ev.evaluate_invocations(inv("q", "yes"), expected="yes")
    failed = await ev.evaluate_invocations(inv("q", "no"), expected="yes")
    assert passed.passed is True
    assert failed.passed is False


# --- llm judge bridge ---------------------------------------------------


async def test_llm_judge_evaluator_bridges_judge_and_client():
    class FakeClient:
        async def generate(self, messages, **opts):
            return '{"label": "A - Good", "rationale": "ok"}'

    judge = BaseJudge(
        name="relevance",
        system_prompt="judge",
        user_prompt_template="Q: {question}\nA: {answer}",
        passing_labels=["A - Good"],
    )
    ev = LlmJudgeEvaluator(judge=judge, model_client=FakeClient())
    result = await ev.evaluate_invocations(inv("why", "because"), expected=None)
    assert result.passed is True
    assert result.score == 1.0


# --- json schema --------------------------------------------------------


async def test_json_schema_evaluator_requires_keys():
    ev = JsonSchemaEvaluator(required_keys=["intent", "confidence"])
    good = await ev.evaluate_invocations(
        inv("q", '{"intent": "buy", "confidence": 0.9}'), expected=None
    )
    bad = await ev.evaluate_invocations(inv("q", '{"intent": "buy"}'), expected=None)
    not_json = await ev.evaluate_invocations(inv("q", "not json"), expected=None)
    assert good.passed is True
    assert bad.passed is False
    assert not_json.passed is False


# --- composite ----------------------------------------------------------


async def test_composite_passes_only_when_all_pass():
    both = CompositeEvaluator([LabelMatchEvaluator(), JsonSchemaEvaluator(["x"])])
    # final_response can't be both a bare label and valid JSON with key x -> fails
    result = await both.evaluate_invocations(inv("q", "Sales"), expected="Sales")
    assert result.passed is False


async def test_composite_passes_when_all_subevaluators_pass():
    comp = CompositeEvaluator([LabelMatchEvaluator(), LabelMatchEvaluator()])
    result = await comp.evaluate_invocations(inv("q", "Sales"), expected="Sales")
    assert result.passed is True
    assert result.score == 1.0
