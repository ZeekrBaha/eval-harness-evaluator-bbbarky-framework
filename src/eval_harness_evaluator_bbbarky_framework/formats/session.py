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
