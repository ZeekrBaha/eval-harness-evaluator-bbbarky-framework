"""Thin adapter for a richer external agent-session shape (vendor-neutral)."""

import json
from pathlib import Path

from eval_harness_evaluator_bbbarky_framework.formats.session import (
    agent_session_to_evalcase,
    agent_sessions_to_evalset,
)

EXPORT_DIR = Path(__file__).parents[1] / "examples" / "agent_session_export"


def _rich_session():
    return {
        "id": "conv-1",
        "user_query": "What's the refund window?",
        "agent_response": "Refunds within 30 days.",
        "history": [
            {"role": "user", "content": "hi"},
            {"role": "agent", "content": "hello, how can I help?"},
        ],
        "retrieved_context": ["Policy: refunds accepted within 30 days."],
        "tool_calls": [{"name": "kb_search", "args": {"q": "refund"}}],
        "judge_configs": {"groundedness": {"rubric_version": "v1", "threshold": 0.5}},
        "expected": "A - Good",
        "metadata": {"kind": "rag"},
    }


def test_adapter_maps_query_response_and_context():
    case = agent_session_to_evalcase(_rich_session())
    inv = case.invocations[0]
    assert inv.user_input == "What's the refund window?"
    assert inv.final_response == "Refunds within 30 days."
    assert inv.context == ["Policy: refunds accepted within 30 days."]
    assert inv.tool_calls[0]["name"] == "kb_search"


def test_adapter_stores_history_and_judge_configs_in_artifacts():
    case = agent_session_to_evalcase(_rich_session())
    assert case.artifacts["judge_configs"]["groundedness"]["rubric_version"] == "v1"
    assert len(case.artifacts["history"]) == 2
    assert case.expected == "A - Good"
    assert case.metadata["kind"] == "rag"


def test_adapter_handles_minimal_session():
    case = agent_session_to_evalcase({"id": "c0", "user_query": "q", "agent_response": "a"})
    assert case.invocations[0].context is None
    assert case.artifacts == {}


def test_sessions_to_evalset_collects_cases():
    evalset = agent_sessions_to_evalset("rag_suite", [_rich_session(), _rich_session()])
    assert evalset.name == "rag_suite"
    assert len(evalset.cases) == 2


def test_shipped_export_example_converts():
    sessions = json.loads((EXPORT_DIR / "sessions.json").read_text())
    evalset = agent_sessions_to_evalset("exported", sessions)
    assert len(evalset.cases) >= 1
    assert evalset.cases[0].invocations[0].user_input
