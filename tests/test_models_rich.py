"""Optional structured fields on Invocation / EvalCase (back-compat preserved)."""

from eval_harness_evaluator_bbbarky_framework.models.core import EvalCase, EvalSet, Invocation


def test_invocation_optional_fields_default_none():
    inv = Invocation(user_input="hi", final_response="hello")
    assert inv.context is None
    assert inv.tool_calls is None
    assert inv.trace_id is None
    assert inv.latency_ms is None
    assert inv.turn_index is None


def test_invocation_accepts_structured_fields():
    inv = Invocation(
        user_input="what is our refund policy?",
        final_response="30 days.",
        context=["Refunds are accepted within 30 days."],
        tool_calls=[{"name": "kb_search", "args": {"q": "refund"}}],
        trace_id="trace-abc",
        latency_ms=812.5,
        turn_index=0,
    )
    assert inv.context == ["Refunds are accepted within 30 days."]
    assert inv.tool_calls[0]["name"] == "kb_search"
    assert inv.trace_id == "trace-abc"
    assert inv.latency_ms == 812.5
    assert inv.turn_index == 0


def test_eval_case_optional_per_turn_and_artifacts():
    case = EvalCase(
        id="c1",
        invocations=[Invocation(user_input="a", final_response="b")],
        expected="x",
        expected_per_turn=["x"],
        artifacts={"source_doc": "kb/refunds.md"},
    )
    assert case.expected_per_turn == ["x"]
    assert case.artifacts["source_doc"] == "kb/refunds.md"


def test_round_trip_still_omits_unset_optionals():
    # Back-compat: a minimal case must round-trip without the new fields appearing.
    data = {
        "name": "regression",
        "cases": [
            {
                "id": "c1",
                "invocations": [{"user_input": "hi", "final_response": "hello"}],
                "expected": "x",
            }
        ],
    }
    evalset = EvalSet.from_dict(data)
    assert evalset.to_dict() == data


def test_round_trip_preserves_structured_fields():
    data = {
        "name": "rag",
        "cases": [
            {
                "id": "c1",
                "invocations": [
                    {
                        "user_input": "q",
                        "final_response": "a",
                        "context": ["doc1"],
                        "trace_id": "t1",
                    }
                ],
                "expected": "x",
            }
        ],
    }
    evalset = EvalSet.from_dict(data)
    assert evalset.to_dict() == data
