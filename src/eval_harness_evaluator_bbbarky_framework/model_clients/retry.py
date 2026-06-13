"""Async retry helper for transient model-provider failures."""

from __future__ import annotations

import asyncio
import functools
import random
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

T = TypeVar("T")

# Only retry exceptions that indicate a transient infrastructure problem.
# Programming errors (TypeError, AttributeError, …) propagate immediately.
_TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError, OSError)


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[type[BaseException], ...] = _TRANSIENT_EXCEPTIONS,
    *,
    delay: float | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorate an async function to retry on transient exceptions.

    Uses exponential back-off with full jitter to avoid thundering herd.
    Only exception types listed in *exceptions* are retried; all others
    propagate immediately.

    Args:
        max_attempts: Total number of tries (including the first).
        base_delay: Base delay in seconds for the exponential back-off.
        max_delay: Upper cap (seconds) for any single sleep.
        exceptions: Exception types that trigger a retry.
        delay: Backward-compat alias for ``base_delay``.
    """
    if delay is not None:
        base_delay = delay

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts - 1:
                        raise
                    cap = min(max_delay, base_delay * (2**attempt))
                    await asyncio.sleep(random.uniform(0.0, cap))
            raise AssertionError("unreachable")

        return wrapper

    return decorator
