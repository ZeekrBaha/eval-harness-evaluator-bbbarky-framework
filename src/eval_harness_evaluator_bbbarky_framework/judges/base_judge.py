"""BaseJudge: render a prompt, call a model client, parse a structured verdict.

A judge is provider-agnostic — it talks to any :class:`ModelClient`. Parsing is
deliberately lenient: models often wrap JSON in prose or fenced code blocks.
"""

from __future__ import annotations

import json

from ..models.core import JudgeResult


def _extract_first_json_object(raw: str) -> str | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(raw[start:], start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return raw[start: i + 1]
    return None


class BaseJudge:
    """An LLM-as-judge with a fixed prompt and a set of passing labels."""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        user_prompt_template: str,
        passing_labels: list[str],
        rubric_version: str | None = None,
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.passing_labels = list(passing_labels)
        self.rubric_version = rubric_version

    def build_messages(self, **template_vars) -> list[dict]:
        """Return chat messages with the system prompt and rendered user prompt."""
        user = self.user_prompt_template.format(**template_vars)
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user},
        ]

    def parse(self, raw: str) -> JudgeResult:
        """Extract the JSON verdict from a raw model response."""
        extracted = _extract_first_json_object(raw)
        if extracted is None:
            return self._error_result(raw, "unparseable judge response")
        try:
            data = json.loads(extracted)
        except json.JSONDecodeError:
            return self._error_result(raw, "invalid JSON in judge response")
        label = str(data.get("label", ""))
        confidence = data.get("confidence")
        return JudgeResult(
            label=label,
            issues=list(data.get("issues", [])),
            rationale=str(data.get("rationale", "")),
            raw_response=raw,
            score=1.0 if label in self.passing_labels else 0.0,
            confidence=float(confidence) if confidence is not None else None,
            rubric_version=self.rubric_version,
        )

    def _error_result(self, raw: str, issue: str) -> JudgeResult:
        return JudgeResult(
            label="",
            issues=[issue],
            rationale="",
            raw_response=raw,
            score=0.0,
            rubric_version=self.rubric_version,
        )

    async def evaluate(self, model_client, **template_vars) -> JudgeResult:
        """Render, call the model client, and parse the verdict."""
        messages = self.build_messages(**template_vars)
        raw = await model_client.generate(messages)
        return self.parse(raw)
