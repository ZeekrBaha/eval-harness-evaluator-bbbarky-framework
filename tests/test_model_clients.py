"""Tests for the provider-agnostic model client layer."""

import asyncio
import time

import pytest

from eval_harness_evaluator_bbbarky_framework.model_clients.error import ModelClientError
from eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client import LiteLLMClient
from eval_harness_evaluator_bbbarky_framework.model_clients.retry import with_retry

# --- retry --------------------------------------------------------------


async def test_with_retry_succeeds_after_transient_failures():
    attempts = 0

    @with_retry(max_attempts=3, delay=0, exceptions=(ValueError,))
    async def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("transient")
        return "ok"

    result = await flaky()
    assert result == "ok"
    assert attempts == 3


async def test_with_retry_raises_after_exhausting_attempts():
    attempts = 0

    @with_retry(max_attempts=2, delay=0, exceptions=(ValueError,))
    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await always_fails()
    assert attempts == 2


async def test_retry_uses_exponential_backoff(monkeypatch):
    """Sleep durations should reflect exponential caps; TypeError is NOT retried."""
    sleep_calls: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    attempt = 0

    @with_retry(max_attempts=3, base_delay=1.0, max_delay=60.0, exceptions=(ConnectionError,))
    async def flaky():
        nonlocal attempt
        attempt += 1
        if attempt < 3:
            raise ConnectionError("transient")
        return "ok"

    result = await flaky()
    assert result == "ok"
    # 3 attempts → 2 sleeps
    assert len(sleep_calls) == 2
    # attempt 0: cap = min(60, 1.0 * 2**0) = 1.0  → sleep in [0, 1.0]
    assert 0.0 <= sleep_calls[0] <= 1.0 + 1e-9
    # attempt 1: cap = min(60, 1.0 * 2**1) = 2.0  → sleep in [0, 2.0]
    assert 0.0 <= sleep_calls[1] <= 2.0 + 1e-9


async def test_retry_does_not_catch_type_error():
    """TypeError (programming bug) must bypass the retry loop immediately."""
    calls = 0

    @with_retry(max_attempts=3)
    async def buggy():
        nonlocal calls
        calls += 1
        raise TypeError("bug!")

    with pytest.raises(TypeError):
        await buggy()

    assert calls == 1  # fired once, not retried


# --- LiteLLMClient ------------------------------------------------------


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


async def test_litellm_client_returns_message_content():
    async def fake_completion(**kwargs):
        return _FakeResponse("the answer")

    client = LiteLLMClient(model="gpt-4o-mini", completion_fn=fake_completion)
    text = await client.generate([{"role": "user", "content": "hi"}])
    assert text == "the answer"


async def test_litellm_client_passes_model_and_messages():
    captured = {}

    async def fake_completion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse("x")

    client = LiteLLMClient(model="claude-3", completion_fn=fake_completion, temperature=0.0)
    await client.generate([{"role": "user", "content": "hi"}])
    assert captured["model"] == "claude-3"
    assert captured["messages"] == [{"role": "user", "content": "hi"}]
    assert captured["temperature"] == 0.0


async def test_litellm_client_wraps_provider_errors():
    async def boom(**kwargs):
        raise RuntimeError("provider down")

    client = LiteLLMClient(model="gpt-4o-mini", completion_fn=boom, max_attempts=1)
    with pytest.raises(ModelClientError):
        await client.generate([{"role": "user", "content": "hi"}])


async def test_litellm_client_timeout():
    """A slow completion_fn must raise ModelClientError well within the timeout."""

    async def slow_completion(**kwargs):
        await asyncio.sleep(10)  # far longer than the configured timeout
        return _FakeResponse("never")

    client = LiteLLMClient(
        model="gpt-4o-mini",
        completion_fn=slow_completion,
        timeout=0.05,
        max_attempts=1,
    )

    start = time.monotonic()
    with pytest.raises(ModelClientError, match="timed out"):
        await client.generate([{"role": "user", "content": "hi"}])
    elapsed = time.monotonic() - start
    assert elapsed < 0.5  # must abort long before 10 s
