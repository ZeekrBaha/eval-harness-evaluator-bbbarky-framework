"""End-to-end tests for the eval-harness CLI."""

import json

import pytest

from eval_harness_evaluator_bbbarky_framework.cli.main import main


def _write_evalset(path):
    evalset = {
        "name": "demo",
        "cases": [
            {
                "id": "c1",
                "invocations": [{"user_input": "q", "final_response": "Sales"}],
                "expected": "Sales",
            },
            {
                "id": "c2",
                "invocations": [{"user_input": "q", "final_response": "Refund"}],
                "expected": "Sales",
            },
        ],
    }
    path.write_text(json.dumps(evalset))


def test_cli_run_writes_json_and_markdown_reports(tmp_path):
    evalset_path = tmp_path / "regression.evalset.json"
    _write_evalset(evalset_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "suite: demo\n"
        f"evalsets:\n  - {evalset_path}\n"
        "evaluators:\n"
        "  - eval_harness_evaluator_bbbarky_framework.evaluators.label_match:LabelMatchEvaluator\n"
    )

    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--config", str(config_path), "--output-dir", str(tmp_path)])

    assert exc_info.value.code == 0
    report_json = tmp_path / "demo_report.json"
    report_md = tmp_path / "demo_report.md"
    assert report_json.exists()
    assert report_md.exists()
    data = json.loads(report_json.read_text())
    assert data["summary"]["n"] == 2
    assert data["summary"]["passed"] == 1


def test_cli_convert_csv_to_evalset(tmp_path):
    csv_path = tmp_path / "cases.csv"
    csv_path.write_text("id,input,response,expected\nc1,hi,Sales,Sales\n")
    out_path = tmp_path / "out.evalset.json"

    with pytest.raises(SystemExit) as exc_info:
        main(
            ["convert", "--input", str(csv_path), "--output", str(out_path), "--suite", "demo"]
        )

    assert exc_info.value.code == 0
    data = json.loads(out_path.read_text())
    assert data["name"] == "demo"
    assert data["cases"][0]["id"] == "c1"


def test_run_nonexistent_config_exits_1():
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--config", "nonexistent.yaml"])
    assert exc_info.value.code == 1


def test_run_bad_yaml_exits_1(tmp_path):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("key: [unclosed\n")
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--config", str(bad_yaml)])
    assert exc_info.value.code == 1
