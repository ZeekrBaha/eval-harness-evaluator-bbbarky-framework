"""Tests for ToolCallEvaluator."""

from eval_harness_evaluator_bbbarky_framework.evaluators.tool_call_evaluator import ToolCallEvaluator
from eval_harness_evaluator_bbbarky_framework.models.core import Invocation


def _inv(tool_calls: list[dict] | None) -> list[Invocation]:
    return [Invocation(user_input="q", final_response="a", tool_calls=tool_calls)]


async def test_tool_call_evaluator_passes_when_tool_called():
    ev = ToolCallEvaluator(expected_tool="search")
    result = await ev.evaluate_invocations(_inv([{"name": "search", "parameters": {}}]))
    assert result.passed is True
    assert result.score == 1.0


async def test_tool_call_evaluator_fails_when_tool_not_called():
    ev = ToolCallEvaluator(expected_tool="search")
    result = await ev.evaluate_invocations(_inv([{"name": "other_tool"}]))
    assert result.passed is False
    assert result.score == 0.0


async def test_tool_call_evaluator_fails_when_required_param_missing():
    ev = ToolCallEvaluator(expected_tool="search", required_params=["query"])
    result = await ev.evaluate_invocations(_inv([{"name": "search", "parameters": {}}]))
    assert result.passed is False


async def test_tool_call_evaluator_partial_score_when_params_missing():
    ev = ToolCallEvaluator(expected_tool="search", required_params=["query"])
    result = await ev.evaluate_invocations(_inv([{"name": "search", "parameters": {}}]))
    assert result.score == 0.5
