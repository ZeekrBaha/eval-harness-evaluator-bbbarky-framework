"""Evaluator asserting the final response is JSON with required keys."""

from __future__ import annotations

import json

from ..models.core import EvalResult, Invocation, PerInvocationResult
from .base import Evaluator


class JsonSchemaEvaluator(Evaluator):
    """Pass when the final response parses as a JSON object containing all keys."""

    metric_name = "json_schema"

    def __init__(self, required_keys: list[str], threshold: float = 0.5) -> None:
        self.required_keys = required_keys
        self.threshold = threshold

    async def evaluate_invocations(
        self, invocations: list[Invocation], expected: object | None
    ) -> EvalResult:
        per: list[PerInvocationResult] = []
        for invocation in invocations:
            ok, detail = self._check(invocation.final_response)
            per.append(PerInvocationResult(score=1.0 if ok else 0.0, passed=ok, details=detail))
        return self._aggregate(per)

    def _check(self, text: str) -> tuple[bool, dict]:
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return False, {"error": "not valid JSON"}
        if not isinstance(data, dict):
            return False, {"error": "JSON is not an object"}
        missing = [k for k in self.required_keys if k not in data]
        return (not missing), {"missing": missing}
