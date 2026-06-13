"""Default provider-agnostic model client backed by litellm.

litellm gives one interface over OpenAI, Anthropic, Gemini, and local models,
so the framework core needs no provider-specific code. The network call is
injectable (``completion_fn``) to keep tests offline.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .error import ModelClientError
from .retry import _TRANSIENT_EXCEPTIONS, with_retry

CompletionFn = Callable[..., Awaitable[Any]]

try:
    from litellm.exceptions import RateLimitError, ServiceUnavailableError

    _LITELLM_TRANSIENT: tuple[type[BaseException], ...] = (RateLimitError, ServiceUnavailableError)
except ImportError:
    _LITELLM_TRANSIENT = ()


class LiteLLMClient:
    """A :class:`ModelClient` that calls litellm's async completion."""

    def __init__(
        self,
        model: str,
        completion_fn: CompletionFn | None = None,
        max_attempts: int = 3,
        retry_delay: float = 0.0,
        timeout: float = 60.0,
        **default_opts,
    ) -> None:
        self.model = model
        self._default_opts = default_opts
        self._max_attempts = max_attempts
        self._retry_delay = retry_delay
        self._timeout = timeout
        if completion_fn is None:
            import litellm

            completion_fn = litellm.acompletion
        self._completion_fn = completion_fn

    async def generate(self, messages: list[dict], **opts) -> str:
        call_opts = {**self._default_opts, **opts}

        @with_retry(
            max_attempts=self._max_attempts,
            delay=self._retry_delay,
            exceptions=(_TRANSIENT_EXCEPTIONS + _LITELLM_TRANSIENT),
        )
        async def _call() -> str:
            try:
                response = await asyncio.wait_for(
                    self._completion_fn(model=self.model, messages=messages, **call_opts),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError as exc:
                raise ModelClientError(
                    f"LiteLLM call timed out after {self._timeout}s"
                ) from exc
            return str(response.choices[0].message.content)

        try:
            return await _call()
        except ModelClientError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider errors
            raise ModelClientError(
                f"litellm completion failed for model {self.model!r}: {exc}"
            ) from exc
