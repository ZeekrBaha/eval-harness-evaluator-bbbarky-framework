"""Bundled generic judges. Importing this package registers them all."""

from __future__ import annotations

from ..registry import register_judge
from .coherence import make_coherence_judge
from .groundedness import make_groundedness_judge
from .relevance import make_relevance_judge

register_judge(make_relevance_judge())
register_judge(make_coherence_judge())
register_judge(make_groundedness_judge())

__all__ = ["make_relevance_judge", "make_coherence_judge", "make_groundedness_judge"]
