"""Async retry helper for transient model-provider failures."""

from __future__ import annotations

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    delay: float = 0.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorate an async function to retry on the given exceptions.

    Retries up to ``max_attempts`` total tries, sleeping ``delay`` seconds
    between attempts. Re-raises the last exception once attempts are exhausted.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return await fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        raise
                    if delay:
                        await asyncio.sleep(delay)
            # Unreachable: loop either returns or raises.
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
