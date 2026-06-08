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


def instantiate_evaluators(specs: list) -> dict:
    """Build a name -> Evaluator mapping from config evaluator specs.

    Each spec is either:
      - a ``"module:Class"`` string (constructed with no arguments), or
      - a dict ``{"type": "module:Class", "name": <key>, "params": {...}}``
        where ``params`` are passed as keyword arguments to the constructor.

    The mapping key is the explicit ``name`` when given, else the instance's
    ``metric_name``.
    """
    evaluators: dict = {}
    for spec in specs:
        params: dict = {}
        name: str | None = None
        if isinstance(spec, str):
            type_path = spec
        else:
            type_path = spec["type"]
            params = spec.get("params", {}) or {}
            name = spec.get("name")
        cls = load_class_from_path(type_path)
        instance = cls(**params)
        evaluators[name or instance.metric_name] = instance
    return evaluators
