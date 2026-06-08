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
    evaluators: list[str] = Field(default_factory=list)
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
