"""Convert a richer agent-session export into a BBBarky EvalSet.

This is the vendor-neutral integration path: an upstream live-session/ADK harness
captures conversations (user query, agent response, retrieved context, tool
calls, prior history, and the judge configs that produced any in-house labels);
you export them as plain records and map them here for offline scoring.

No proprietary dependencies — only the generic adapter in formats/session.py.

    uv run python examples/agent_session_export/export_to_evalset.py
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_harness_evaluator_bbbarky_framework.formats.session import agent_sessions_to_evalset

HERE = Path(__file__).parent


def main() -> None:
    sessions = json.loads((HERE / "sessions.json").read_text())
    evalset = agent_sessions_to_evalset("exported_rag_suite", sessions)
    out = HERE / "exported_rag_suite.evalset.json"
    out.write_text(json.dumps(evalset.to_dict(), indent=2))
    print(f"wrote {len(evalset.cases)} cases to {out}")
    # Judge configs and prior history are preserved on each case under artifacts.
    print("artifacts on case 0:", sorted(evalset.cases[0].artifacts))


if __name__ == "__main__":
    main()
