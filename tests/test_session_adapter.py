"""Generic agent-session -> EvalSet adapter (vendor-neutral integration pattern)."""

from eval_harness_evaluator_bbbarky_framework.formats.session import (
    session_to_evalcase,
    sessions_to_evalset,
)


def _session():
    return {
        "id": "sess-1",
        "turns": [
            {"user": "hi", "response": "hello", "context": ["greeting policy"]},
            {
                "user": "refund?",
                "response": "30 days",
                "tool_calls": [{"name": "kb_search", "args": {"q": "refund"}}],
            },
        ],
        "expected": "handled",
        "metadata": {"kind": "support"},
    }


def test_session_to_evalcase_builds_one_invocation_per_turn():
    case = session_to_evalcase(_session())
    assert case.id == "sess-1"
    assert len(case.invocations) == 2
    assert case.invocations[0].user_input == "hi"
    assert case.invocations[0].context == ["greeting policy"]
    assert case.invocations[0].turn_index == 0
    assert case.invocations[1].tool_calls[0]["name"] == "kb_search"
    assert case.invocations[1].turn_index == 1


def test_session_to_evalcase_carries_expected_and_metadata():
    case = session_to_evalcase(_session())
    assert case.expected == "handled"
    assert case.metadata["kind"] == "support"


def test_session_handles_missing_optional_fields():
    case = session_to_evalcase({"id": "s2", "turns": [{"user": "q", "response": "a"}]})
    assert case.invocations[0].context is None
    assert case.invocations[0].tool_calls is None
    assert case.expected is None


def test_sessions_to_evalset_collects_cases():
    evalset = sessions_to_evalset("support_suite", [_session(), {"id": "s2", "turns": []}])
    assert evalset.name == "support_suite"
    assert [c.id for c in evalset.cases] == ["sess-1", "s2"]
