"""Convert a captured agent-session log into an EvalSet JSON file.

Demonstrates the vendor-neutral integration seam: bring your own agent's
transcript (as generic session dicts) into the framework.

    uv run python examples/session_conversion/convert_sessions.py
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_harness_evaluator_bbbarky_framework.formats.session import sessions_to_evalset

HERE = Path(__file__).parent


def main() -> None:
    sessions = json.loads((HERE / "sessions.json").read_text())
    evalset = sessions_to_evalset("support_suite", sessions)
    out = HERE / "support_suite.evalset.json"
    out.write_text(json.dumps(evalset.to_dict(), indent=2))
    print(f"wrote {len(evalset.cases)} cases to {out}")


if __name__ == "__main__":
    main()
