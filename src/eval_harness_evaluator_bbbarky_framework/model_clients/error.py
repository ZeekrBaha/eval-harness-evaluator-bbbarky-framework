"""Errors for the model client layer."""

from __future__ import annotations


class ModelClientError(RuntimeError):
    """Raised when a model client fails to produce a completion."""
