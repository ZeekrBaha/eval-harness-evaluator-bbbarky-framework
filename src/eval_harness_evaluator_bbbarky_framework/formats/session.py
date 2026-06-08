"""Adapt a generic agent *session* into the framework's EvalSet/EvalCase model.

This is the vendor-neutral integration seam: capture whatever your agent
produced as a plain session dict, and map it here. No coupling to any specific
agent runtime — adapt your own transcript shape into this one.

Session shape (all turn fields except ``user``/``response`` are optional)::

    {
      "id": "session-123",
      "turns": [
        {"user": "...", "response": "...", "context": [...], "tool_calls": [...],
         "trace_id": "...", "latency_ms": 12.3}
      ],
      "expected": <case-level expected outcome>,   # optional
      "metadata": {"kind": "...", "lang": "..."}    # optional
    }
"""

from __future__ import annotations

from ..models.core import EvalCase, EvalSet, Invocation


def _turn_to_invocation(turn: dict, index: int) -> Invocation:
    return Invocation(
        user_input=str(turn.get("user", "")),
        final_response=str(turn.get("response", "")),
        context=turn.get("context"),
        tool_calls=turn.get("tool_calls"),
        trace_id=turn.get("trace_id"),
        latency_ms=turn.get("latency_ms"),
        turn_index=index,
    )


def session_to_evalcase(session: dict) -> EvalCase:
    """Convert one session dict into an EvalCase (one invocation per turn)."""
    turns = session.get("turns", [])
    return EvalCase(
        id=str(session.get("id", "session")),
        invocations=[_turn_to_invocation(t, i) for i, t in enumerate(turns)],
        expected=session.get("expected"),
        metadata=session.get("metadata", {}) or {},
    )


def sessions_to_evalset(name: str, sessions: list[dict]) -> EvalSet:
    """Convert many session dicts into a named EvalSet."""
    return EvalSet(name=name, cases=[session_to_evalcase(s) for s in sessions])


# --- richer single-turn agent export shape ------------------------------------
#
# Some agents export one record per conversation with a richer shape than the
# multi-turn ``turns`` format above: a user query, the agent's response, a
# retrieved-context list, tool calls, a prior-history transcript, and the judge
# configuration that produced any in-house labels. This thin adapter maps that
# shape in without coupling the core to any specific agent runtime.
#
#     {
#       "id": "...",
#       "user_query": "...",
#       "agent_response": "...",
#       "retrieved_context": [...],         # -> Invocation.context
#       "tool_calls": [...],                # -> Invocation.tool_calls
#       "history": [{"role": ..., "content": ...}, ...],   # -> artifacts["history"]
#       "judge_configs": {...},             # -> artifacts["judge_configs"]
#       "expected": <label>,                # optional
#       "metadata": {...}                   # optional
#     }


def agent_session_to_evalcase(session: dict) -> EvalCase:
    """Map a richer single-turn agent-export record into an EvalCase."""
    invocation = Invocation(
        user_input=str(session.get("user_query", "")),
        final_response=str(session.get("agent_response", "")),
        context=session.get("retrieved_context"),
        tool_calls=session.get("tool_calls"),
        trace_id=session.get("trace_id"),
        latency_ms=session.get("latency_ms"),
        turn_index=0,
    )
    artifacts: dict = {}
    if session.get("history") is not None:
        artifacts["history"] = session["history"]
    if session.get("judge_configs") is not None:
        artifacts["judge_configs"] = session["judge_configs"]
    return EvalCase(
        id=str(session.get("id", "session")),
        invocations=[invocation],
        expected=session.get("expected"),
        artifacts=artifacts,
        metadata=session.get("metadata", {}) or {},
    )


def agent_sessions_to_evalset(name: str, sessions: list[dict]) -> EvalSet:
    """Convert many richer agent-export records into a named EvalSet."""
    return EvalSet(name=name, cases=[agent_session_to_evalcase(s) for s in sessions])
