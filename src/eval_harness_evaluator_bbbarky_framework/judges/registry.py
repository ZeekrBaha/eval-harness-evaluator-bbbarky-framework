"""Name -> judge lookup, with a shared module-level default registry."""

from __future__ import annotations

from .base_judge import BaseJudge


class JudgeRegistry:
    """An in-memory registry mapping judge names to judge instances."""

    def __init__(self) -> None:
        self._judges: dict[str, BaseJudge] = {}

    def register(self, judge: BaseJudge) -> BaseJudge:
        self._judges[judge.name] = judge
        return judge

    def get(self, name: str) -> BaseJudge:
        if name not in self._judges:
            raise KeyError(f"no judge registered under name {name!r}")
        return self._judges[name]

    def names(self) -> list[str]:
        return sorted(self._judges)


# Shared default registry used by config modules and convenience helpers.
default_registry = JudgeRegistry()


def register_judge(judge: BaseJudge) -> BaseJudge:
    return default_registry.register(judge)


def get_judge(name: str) -> BaseJudge:
    return default_registry.get(name)


def seed_default_judges(registry: JudgeRegistry | None = None) -> JudgeRegistry:
    """Seed a registry with bundled judges. Uses default_registry if None passed."""
    from .configs.relevance import make_relevance_judge
    from .configs.coherence import make_coherence_judge
    from .configs.groundedness import make_groundedness_judge

    r = registry or default_registry
    r.register(make_relevance_judge())
    r.register(make_coherence_judge())
    r.register(make_groundedness_judge())
    return r
