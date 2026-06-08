"""The provider-agnostic model client contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelClient(Protocol):
    """Minimal async chat-completion interface.

    Any object with this shape can drive an LLM-as-judge evaluator, so users
    may register native SDK clients in place of the default litellm client.
    """

    async def generate(self, messages: list[dict], **opts) -> str:
        """Return the assistant's text reply for ``messages``."""
        ...
