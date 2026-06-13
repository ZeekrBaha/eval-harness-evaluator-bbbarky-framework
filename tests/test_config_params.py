"""Config evaluators can be constructed with arguments (EvaluatorSpec)."""

import json

import pytest

from eval_harness_evaluator_bbbarky_framework.cli.main import main
from eval_harness_evaluator_bbbarky_framework.runner.config import (
    RunConfig,
    instantiate_evaluators,
)


def test_run_config_accepts_string_and_dict_evaluators():
    label_path = (
        "eval_harness_evaluator_bbbarky_framework.evaluators.label_match:LabelMatchEvaluator"
    )
    schema_path = (
        "eval_harness_evaluator_bbbarky_framework.evaluators.json_schema:JsonSchemaEvaluator"
    )
    config = RunConfig.model_validate(
        {
            "suite": "demo",
            "evalsets": ["x.json"],
            "evaluators": [
                label_path,
                {"type": schema_path, "name": "schema", "params": {"required_keys": ["a"]}},
            ],
        }
    )
    assert config.evaluators[0] == label_path
    assert config.evaluators[1]["name"] == "schema"


def test_instantiate_string_backcompat():
    evaluators = instantiate_evaluators(
        ["eval_harness_evaluator_bbbarky_framework.evaluators.label_match:LabelMatchEvaluator"]
    )
    assert "label_match" in evaluators


def test_instantiate_passes_params_to_constructor():
    evaluators = instantiate_evaluators(
        [
            {
                "type": "eval_harness_evaluator_bbbarky_framework.evaluators.json_schema:JsonSchemaEvaluator",
                "name": "intent_schema",
                "params": {"required_keys": ["intent", "confidence"], "threshold": 1.0},
            }
        ]
    )
    ev = evaluators["intent_schema"]
    assert ev.required_keys == ["intent", "confidence"]
    assert ev.threshold == 1.0


def test_cli_run_with_parameterized_evaluator(tmp_path):
    evalset = {
        "name": "intent",
        "cases": [
            {
                "id": "c1",
                "invocations": [
                    {"user_input": "q", "final_response": '{"intent": "buy", "confidence": 0.9}'}
                ],
                "expected": None,
            },
            {
                "id": "c2",
                "invocations": [{"user_input": "q", "final_response": '{"intent": "buy"}'}],
                "expected": None,
            },
        ],
    }
    evalset_path = tmp_path / "intent.evalset.json"
    evalset_path.write_text(json.dumps(evalset))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "suite: intent\n"
        f"evalsets:\n  - {evalset_path}\n"
        "evaluators:\n"
        "  - type: eval_harness_evaluator_bbbarky_framework.evaluators.json_schema:JsonSchemaEvaluator\n"
        "    name: intent_schema\n"
        "    params:\n"
        "      required_keys: [intent, confidence]\n"
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--config", str(config_path), "--output-dir", str(tmp_path)])
    assert exc_info.value.code == 0
    report = json.loads((tmp_path / "intent_report.json").read_text())
    assert report["summary"]["by_metric"]["intent_schema"]["passed"] == 1
