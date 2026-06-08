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
    ) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.passing_labels = list(passing_labels)

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
            return JudgeResult(
                label="",
                issues=["unparseable judge response"],
                rationale="",
                raw_response=raw,
                score=0.0,
            )
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return JudgeResult(
                label="",
                issues=["invalid JSON in judge response"],
                rationale="",
                raw_response=raw,
                score=0.0,
            )
        label = str(data.get("label", ""))
        return JudgeResult(
            label=label,
            issues=list(data.get("issues", [])),
            rationale=str(data.get("rationale", "")),
            raw_response=raw,
            score=1.0 if label in self.passing_labels else 0.0,
        )

    async def evaluate(self, model_client, **template_vars) -> JudgeResult:
        """Render, call the model client, and parse the verdict."""
        messages = self.build_messages(**template_vars)
        raw = await model_client.generate(messages)
        return self.parse(raw)
