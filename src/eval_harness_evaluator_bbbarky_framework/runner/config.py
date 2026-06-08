"""YAML run configuration and dynamic class loading."""

from __future__ import annotations

import importlib

import yaml
from pydantic import BaseModel, Field, field_validator


class RunConfig(BaseModel):
    """Parsed evaluation run configuration.

    ``evaluators`` entries are either ``"module.path:ClassName"`` references or
    names of evaluators the caller has pre-registered.
    """

    suite: str
    evalsets: list[str]
    evaluators: list[str | dict] = Field(default_factory=list)
    model_client: dict | None = None
    criteria: str | None = None

    @field_validator("evalsets")
    @classmethod
    def _non_empty_evalsets(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("config must list at least one evalset")
        return value


def load_config(path: str) -> RunConfig:
    """Load and validate a YAML run configuration."""
    with open(path) as handle:
        data = yaml.safe_load(handle) or {}
    return RunConfig.model_validate(data)


def load_class_from_path(path: str):
    """Import a class from a ``"module.path:ClassName"`` reference."""
    if ":" not in path:
        raise ValueError(f"expected 'module:Class', got {path!r}")
    module_path, class_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def _resolve_value(value):
    """Resolve a param value, recursing into nested ``{type, params}`` specs.

    Lists are mapped element-wise (so a list of plain strings like
    ``required_keys`` passes through untouched, while a list of specs is built).
    """
    if isinstance(value, dict) and "type" in value:
        return build_object(value)
    if isinstance(value, list):
        return [_resolve_value(item) for item in value]
    return value


def build_object(spec: dict):
    """Construct an object from a ``{"type": "module:Class", "params": {...}}`` spec.

    ``params`` values are resolved recursively, so an evaluator can take other
    evaluators or scorers as arguments (e.g. CompositeEvaluator, ScorerEvaluator).
    """
    cls = load_class_from_path(spec["type"])
    params = {k: _resolve_value(v) for k, v in (spec.get("params") or {}).items()}
    return cls(**params)


def instantiate_evaluators(specs: list, model_client=None) -> dict:
    """Build a name -> Evaluator mapping from config evaluator specs.

    Each spec is one of:
      - a ``"module:Class"`` string (constructed with no arguments);
      - a dict ``{"type": "module:Class", "name": <key>, "params": {...}}`` where
        ``params`` are keyword arguments, resolved recursively for nested specs;
      - a dict ``{"judge": "<registered judge>", "name": <key>, "threshold": ...}``
        which builds an LlmJudgeEvaluator (requires ``model_client``).

    The mapping key is the explicit ``name`` when given, else ``metric_name``.
    """
    evaluators: dict = {}
    for spec in specs:
        if isinstance(spec, str):
            instance = load_class_from_path(spec)()
            evaluators[instance.metric_name] = instance
        elif "judge" in spec:
            instance = _build_judge_evaluator(spec, model_client)
            evaluators[spec.get("name") or instance.metric_name] = instance
        else:
            instance = build_object(spec)
            evaluators[spec.get("name") or instance.metric_name] = instance
    return evaluators


def _build_judge_evaluator(spec: dict, model_client):
    from ..evaluators.llm_judge import LlmJudgeEvaluator
    from ..judges import configs as _configs  # noqa: F401  (registers bundled judges)
    from ..judges.registry import get_judge

    if model_client is None:
        raise ValueError(
            f"judge evaluator {spec.get('judge')!r} needs a model client; add a "
            f"top-level 'model_client' section to the config"
        )
    judge = get_judge(spec["judge"])
    return LlmJudgeEvaluator(
        judge=judge, model_client=model_client, threshold=spec.get("threshold", 0.5)
    )
