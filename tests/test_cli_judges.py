"""CLI support for LLM-judge evaluators via a model_client config section."""

import json

import pytest

from eval_harness_evaluator_bbbarky_framework.cli.main import main
from eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client import LiteLLMClient
from eval_harness_evaluator_bbbarky_framework.models.core import Invocation
from eval_harness_evaluator_bbbarky_framework.runner.config import instantiate_evaluators


class FakeClient:
    async def generate(self, messages, **opts):
        return '{"label": "5", "rationale": "supported", "confidence": 0.9}'


async def test_judge_spec_builds_llm_judge_evaluator_with_client():
    evaluators = instantiate_evaluators(
        [{"judge": "groundedness", "name": "grounded", "threshold": 0.5}],
        model_client=FakeClient(),
    )
    result = await evaluators["grounded"].evaluate_invocations(
        [Invocation(user_input="q", final_response="a", context=["a"])], expected=None
    )
    assert result.passed is True


def test_judge_spec_without_model_client_raises_clear_error():
    with pytest.raises(ValueError, match="model client"):
        instantiate_evaluators([{"judge": "groundedness"}], model_client=None)


def test_cli_run_with_model_client_and_judge(tmp_path, monkeypatch):
    async def fake_generate(self, messages, **opts):
        return '{"label": "5", "rationale": "ok", "confidence": 0.8}'

    monkeypatch.setattr(LiteLLMClient, "generate", fake_generate)

    evalset = {
        "name": "rag",
        "cases": [
            {
                "id": "c1",
                "invocations": [{"user_input": "q", "final_response": "a", "context": ["a"]}],
                "expected": None,
            }
        ],
    }
    (tmp_path / "rag.evalset.json").write_text(json.dumps(evalset))
    (tmp_path / "config.yaml").write_text(
        "suite: rag\n"
        f"evalsets:\n  - {tmp_path / 'rag.evalset.json'}\n"
        "model_client:\n"
        "  type: eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client:LiteLLMClient\n"
        "  params:\n"
        "    model: gpt-4o-mini\n"
        "    temperature: 0.0\n"
        "evaluators:\n"
        "  - judge: groundedness\n"
        "    name: groundedness\n"
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["run", "--config", str(tmp_path / "config.yaml"), "--output-dir", str(tmp_path)])
    assert exc_info.value.code == 0
    report = json.loads((tmp_path / "rag_report.json").read_text())
    assert report["summary"]["by_metric"]["groundedness"]["passed"] == 1
