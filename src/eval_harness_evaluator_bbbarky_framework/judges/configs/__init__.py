"""Bundled generic judges. Importing this package registers them all."""

from __future__ import annotations

from . import coherence, groundedness, relevance  # noqa: F401

__all__ = ["relevance", "coherence", "groundedness"]
