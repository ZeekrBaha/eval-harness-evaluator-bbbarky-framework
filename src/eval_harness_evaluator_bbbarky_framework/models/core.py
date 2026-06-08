"""Core data models for the evaluation framework.

Provider-agnostic, decoupled from any agent runtime. These are the canonical
types that flow through the runner, evaluators, judges, and reporters.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Invocation(BaseModel):
    """A single user turn and the agent's final response to it."""

    user_input: str
    final_response: str


class EvalCase(BaseModel):
    """One test case: a sequence of invocations plus the expected outcome.

    ``expected`` is intentionally loose (a label, list of labels, or arbitrary
    expected payload) so different evaluators can interpret it.
    """

    id: str
    invocations: list[Invocation] = Field(default_factory=list)
    expected: object | None = None
    metadata: dict = Field(default_factory=dict)


class EvalSet(BaseModel):
    """A named collection of evaluation cases."""

    name: str
    cases: list[EvalCase] = Field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "EvalSet":
        return cls.model_validate(data)

    def to_dict(self) -> dict:
        return self.model_dump(exclude_defaults=True)


class PerInvocationResult(BaseModel):
    """Result for a single invocation within a case."""

    score: float
    passed: bool
    details: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregate result for one case from one evaluator."""

    score: float
    passed: bool
    per_invocation: list[PerInvocationResult] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """Structured output from an LLM-as-judge evaluation."""

    label: str
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""
    raw_response: str = ""
    score: float = 0.0
