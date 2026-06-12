"""Tests for OpenRouterBackend."""

import json

import httpx
import pytest
import respx

from ollama_mcp.backends.openrouter import OpenRouterBackend
from ollama_mcp.errors import (
    OpenRouterAuthError,
    OpenRouterConnectionError,
    OpenRouterModelNotFound,
    OpenRouterRateLimitError,
    OpenRouterServerError,
    OpenRouterTimeout,
)

BASE = "https://openrouter.ai/api/v1"
CHAT_URL = f"{BASE}/chat/completions"
MODELS_URL = f"{BASE}/models"


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-123")
    return OpenRouterBackend(
        api_key_env="OPENROUTER_API_KEY",
        default_model="google/gemma-3-27b-it:free",
    )


def _chat_response(content="hello", prompt_tokens=10, completion_tokens=5, cost=None):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    if cost is not None:
        usage["cost"] = cost
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}}],
        "usage": usage,
    })


# --- generate ---


@respx.mock
async def test_generate_success(backend):
    respx.post(CHAT_URL).mock(return_value=_chat_response("hello world"))

    text, meta = await backend.generate("say hello")

    assert text == "hello world"
    assert meta["backend"] == "openrouter"
    assert meta["model"] == "google/gemma-3-27b-it:free"
    assert meta["prompt_tokens"] == 10
    assert meta["output_tokens"] == 5
    assert meta["eval_ms"] == 0
    assert "wall_ms" in meta


@respx.mock
async def test_generate_sends_correct_format(backend):
    route = respx.post(CHAT_URL).mock(return_value=_chat_response())

    await backend.generate("test prompt", system="be concise")

    payload = json.loads(route.calls[0].request.read())
    assert payload["model"] == "google/gemma-3-27b-it:free"
    assert payload["messages"] == [
        {"role": "system", "content": "be concise"},
        {"role": "user", "content": "test prompt"},
    ]

    headers = route.calls[0].request.headers
    assert headers["authorization"] == "Bearer test-key-123"


@respx.mock
async def test_generate_no_system_prompt(backend):
    route = respx.post(CHAT_URL).mock(return_value=_chat_response())

    await backend.generate("test prompt")

    payload = json.loads(route.calls[0].request.read())
    assert len(payload["messages"]) == 1
    assert payload["messages"][0]["role"] == "user"


@respx.mock
async def test_generate_model_override(backend):
    route = respx.post(CHAT_URL).mock(return_value=_chat_response())

    text, meta = await backend.generate("test", model="meta-llama/llama-3-70b")

    payload = json.loads(route.calls[0].request.read())
    assert payload["model"] == "meta-llama/llama-3-70b"
    assert meta["model"] == "meta-llama/llama-3-70b"


# --- error handling ---


@respx.mock
async def test_generate_connection_error(backend):
    respx.post(CHAT_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OpenRouterConnectionError, match="openrouter.ai"):
        await backend.generate("hello")


@respx.mock
async def test_generate_timeout(backend):
    respx.post(CHAT_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(OpenRouterTimeout, match="timed out"):
        await backend.generate("hello")


@respx.mock
async def test_generate_auth_error_401(backend):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(401, text="unauthorized"))

    with pytest.raises(OpenRouterAuthError, match="Invalid API key"):
        await backend.generate("hello")


@respx.mock
async def test_generate_rate_limit_429(backend):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(429, text="too many"))

    with pytest.raises(OpenRouterRateLimitError, match="rate limit"):
        await backend.generate("hello")


@respx.mock
async def test_generate_model_not_found_404(backend):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(404, text="not found"))

    with pytest.raises(OpenRouterModelNotFound, match="not available"):
        await backend.generate("hello")


@respx.mock
async def test_generate_server_error_500(backend):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(500, text="internal"))

    with pytest.raises(OpenRouterServerError, match="500"):
        await backend.generate("hello")


@respx.mock
async def test_generate_malformed_response(backend):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"bad": "shape"}))

    with pytest.raises(OpenRouterServerError, match="unexpected response"):
        await backend.generate("hello")


def test_missing_api_key():
    backend = OpenRouterBackend(api_key_env="NONEXISTENT_KEY_VAR")

    with pytest.raises(OpenRouterAuthError, match="NONEXISTENT_KEY_VAR"):
        backend._get_api_key()


def test_direct_api_key():
    backend = OpenRouterBackend(api_key="sk-or-direct-key")
    assert backend._get_api_key() == "sk-or-direct-key"


def test_direct_api_key_takes_priority(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    backend = OpenRouterBackend(api_key="direct-key")
    assert backend._get_api_key() == "direct-key"


# --- cost capture ---


@respx.mock
async def test_generate_captures_cost_from_usage(backend):
    respx.post(CHAT_URL).mock(return_value=_chat_response(cost=0.000456))

    text, meta = await backend.generate("test")

    assert meta["cost"] == pytest.approx(0.000456)


@respx.mock
async def test_generate_captures_cost_from_total_cost(backend):
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        "total_cost": 0.00123,
    }))

    text, meta = await backend.generate("test")

    assert meta["cost"] == pytest.approx(0.00123)


@respx.mock
async def test_generate_no_cost_when_absent(backend):
    respx.post(CHAT_URL).mock(return_value=_chat_response())

    text, meta = await backend.generate("test")

    assert "cost" not in meta


# --- list_models ---


@respx.mock
async def test_list_models(backend):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={
        "data": [
            {"id": "google/gemma-3-27b-it:free"},
            {"id": "meta-llama/llama-3-70b"},
            {"id": "anthropic/claude-3-haiku"},
        ],
    }))

    models = await backend.list_models()

    assert models == [
        "anthropic/claude-3-haiku",
        "google/gemma-3-27b-it:free",
        "meta-llama/llama-3-70b",
    ]


@respx.mock
async def test_list_models_empty(backend):
    respx.get(MODELS_URL).mock(return_value=httpx.Response(200, json={"data": []}))

    assert await backend.list_models() == []


@respx.mock
async def test_list_models_connection_error(backend):
    respx.get(MODELS_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OpenRouterConnectionError):
        await backend.list_models()
