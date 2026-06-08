"""BaseJudge: render a prompt, call a model client, parse a structured verdict.

A judge is provider-agnostic — it talks to any :class:`ModelClient`. Parsing is
deliberately lenient: models often wrap JSON in prose or fenced code blocks.
"""

from __future__ import annotations

import json
import re

from ..models.core import JudgeResult

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


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
        match = _JSON_OBJECT.search(raw)
        if not match:
            return self._error_result(raw, "unparseable judge response")
        try:
            data = json.loads(match.group(0))
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
