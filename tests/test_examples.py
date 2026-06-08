"""The shipped examples must stay valid and runnable."""

import json
from pathlib import Path

import pytest

from eval_harness_evaluator_bbbarky_framework.cli.main import main
from eval_harness_evaluator_bbbarky_framework.formats.session import sessions_to_evalset
from eval_harness_evaluator_bbbarky_framework.models.core import EvalSet

EXAMPLES = Path(__file__).parents[1] / "examples"


@pytest.mark.parametrize(
    "rel",
    [
        "faq_router/evalsets/regression.evalset.json",
        "intent/evalset.json",
        "rag/evalset.json",
        "multiturn/evalset.json",
    ],
)
def test_example_evalsets_load(rel):
    evalset = EvalSet.from_dict(json.loads((EXAMPLES / rel).read_text()))
    assert len(evalset.cases) >= 1


def test_intent_example_runs_offline(tmp_path, monkeypatch):
    monkeypatch.chdir(Path(__file__).parents[1])
    exit_code = main(
        ["run", "--config", "examples/config/intent.yaml", "--output-dir", str(tmp_path)]
    )
    assert exit_code == 0
    report = json.loads((tmp_path / "intent_extraction_report.json").read_text())
    # 2 of 3 responses carry both intent + confidence.
    assert report["summary"]["by_metric"]["intent_schema"]["passed"] == 2


def test_session_conversion_example_builds_multiturn_evalset():
    sessions = json.loads((EXAMPLES / "session_conversion" / "sessions.json").read_text())
    evalset = sessions_to_evalset("support_suite", sessions)
    assert [c.id for c in evalset.cases] == ["sess-support-1", "sess-support-2"]
    # first session has two turns -> two invocations with ordered turn_index
    first = evalset.cases[0]
    assert len(first.invocations) == 2
    assert first.invocations[1].turn_index == 1
    assert first.invocations[1].tool_calls[0]["name"] == "kb_search"
