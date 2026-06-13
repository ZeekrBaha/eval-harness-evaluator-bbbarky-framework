"""Tests for core data models: Invocation, EvalCase, EvalSet, EvalResult, JudgeResult."""

from datetime import datetime

import pytest
from pydantic import ValidationError

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


# --- new tests for token/cost/timestamp fields and validators ---


def test_invocation_accepts_token_fields():
    now = datetime.now()
    inv = Invocation(
        user_input="hello",
        final_response="hi",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        started_at=now,
    )
    assert inv.input_tokens == 10
    assert inv.output_tokens == 5
    assert inv.cost_usd == 0.001
    assert inv.started_at == now


def test_judge_result_accepts_token_fields():
    jr = JudgeResult(
        label="A",
        input_tokens=50,
        output_tokens=20,
        cost_usd=0.002,
    )
    assert jr.input_tokens == 50
    assert jr.output_tokens == 20
    assert jr.cost_usd == 0.002


def test_eval_result_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        EvalResult(score=1.5, passed=True)


def test_per_invocation_score_rejects_out_of_range():
    with pytest.raises(ValidationError):
        PerInvocationResult(score=-0.1, passed=False)


def test_judge_result_confidence_rejects_out_of_range():
    with pytest.raises(ValidationError):
        JudgeResult(label="A", confidence=1.5)
