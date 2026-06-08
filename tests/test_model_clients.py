"""Tests for the provider-agnostic model client layer."""

import pytest

from eval_harness_evaluator_bbbarky_framework.model_clients.error import ModelClientError
from eval_harness_evaluator_bbbarky_framework.model_clients.litellm_client import LiteLLMClient
from eval_harness_evaluator_bbbarky_framework.model_clients.retry import with_retry

# --- retry --------------------------------------------------------------


async def test_with_retry_succeeds_after_transient_failures():
    attempts = 0

    @with_retry(max_attempts=3, delay=0)
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

    @with_retry(max_attempts=2, delay=0)
    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await always_fails()
    assert attempts == 2


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
