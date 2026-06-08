"""Tests for core data models: Invocation, EvalCase, EvalSet, EvalResult, JudgeResult."""

from eval_harness_evaluator_bbbarky_framework.models.core import (
    EvalCase,
    EvalResult,
    EvalSet,
    Invocation,
    JudgeResult,
    PerInvocationResult,
)


def test_invocation_holds_user_input_and_final_response():
    inv = Invocation(user_input="hello", final_response="hi there")
    assert inv.user_input == "hello"
    assert inv.final_response == "hi there"


def test_eval_case_has_id_invocations_and_expected_label():
    case = EvalCase(
        id="case-1",
        invocations=[Invocation(user_input="hi", final_response="hello")],
        expected="A - Good",
    )
    assert case.id == "case-1"
    assert len(case.invocations) == 1
    assert case.expected == "A - Good"


def test_eval_set_round_trips_through_dict():
    data = {
        "name": "regression",
        "cases": [
            {
                "id": "case-1",
                "invocations": [{"user_input": "hi", "final_response": "hello"}],
                "expected": "A - Good",
            }
        ],
    }
    evalset = EvalSet.from_dict(data)
    assert evalset.name == "regression"
    assert evalset.cases[0].id == "case-1"
    assert evalset.cases[0].invocations[0].user_input == "hi"
    # round-trip back out
    assert evalset.to_dict() == data


def test_eval_result_reports_score_and_passed():
    result = EvalResult(
        score=1.0,
        passed=True,
        per_invocation=[PerInvocationResult(score=1.0, passed=True)],
    )
    assert result.score == 1.0
    assert result.passed is True
    assert result.per_invocation[0].passed is True


def test_judge_result_carries_label_rationale_and_score():
    jr = JudgeResult(
        label="A - Good",
        issues=[],
        rationale="title is concise and relevant",
        raw_response='{"label": "A - Good"}',
        score=1.0,
    )
    assert jr.label == "A - Good"
    assert jr.score == 1.0
    assert jr.issues == []
