"""Core data models for the evaluation framework.

Provider-agnostic, decoupled from any agent runtime. These are the canonical
types that flow through the runner, evaluators, judges, and reporters.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Invocation(BaseModel):
    """A single user turn and the agent's final response to it.

    The first two fields are the lightweight core. The rest are optional
    structured context for richer evals (grounding, tool-calling, latency,
    tracing, multi-turn ordering) and stay absent when unused.
    """

    user_input: str
    final_response: str
    context: object | None = None
    tool_calls: list[dict] | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    turn_index: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    started_at: datetime | None = None


class EvalCase(BaseModel):
    """One test case: a sequence of invocations plus the expected outcome.

    ``expected`` is intentionally loose (a label, list of labels, or arbitrary
    expected payload) so different evaluators can interpret it.
    """

    id: str
    invocations: list[Invocation] = Field(default_factory=list)
    expected: object | None = None
    expected_per_turn: list | None = None
    artifacts: dict = Field(default_factory=dict)
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

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    details: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Aggregate result for one case from one evaluator."""

    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    per_invocation: list[PerInvocationResult] = Field(default_factory=list)
    details: dict = Field(default_factory=dict)


class JudgeResult(BaseModel):
    """Structured output from an LLM-as-judge evaluation."""

    label: str
    issues: list[str] = Field(default_factory=list)
    rationale: str = ""
    raw_response: str = ""
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rubric_version: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
