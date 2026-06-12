"""Tests for OllamaBackend."""

import json

import httpx
import pytest
import respx

from ollama_mcp.backends.ollama import OllamaBackend
from ollama_mcp.errors import (
    OllamaConnectionError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)

URL = "http://localhost:11434"
API_URL = f"{URL}/api/generate"
TAGS_URL = f"{URL}/api/tags"


@pytest.fixture
def backend():
    return OllamaBackend(url=URL, default_model="gemma4-32k")


# --- generate ---


@respx.mock
async def test_generate_success(backend):
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "hello",
        "prompt_eval_count": 10,
        "eval_count": 5,
        "eval_duration": 100_000_000,
    }))

    text, meta = await backend.generate("say hello")

    assert text == "hello"
    assert meta["model"] == "gemma4-32k"
    assert meta["backend"] == "ollama"
    assert meta["prompt_tokens"] == 10
    assert meta["output_tokens"] == 5
    assert "wall_ms" in meta
    assert "eval_ms" in meta


@respx.mock
async def test_generate_with_system(backend):
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "ok",
    }))

    await backend.generate("test", system="be concise")

    payload = json.loads(route.calls[0].request.read())
    assert payload["system"] == "be concise"


@respx.mock
async def test_generate_with_model_override(backend):
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "ok",
    }))

    text, meta = await backend.generate("test", model="custom-model")

    payload = json.loads(route.calls[0].request.read())
    assert payload["model"] == "custom-model"
    assert meta["model"] == "custom-model"


@respx.mock
async def test_generate_connection_error(backend):
    respx.post(API_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        await backend.generate("hello")


@respx.mock
async def test_generate_timeout(backend):
    respx.post(API_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(OllamaTimeout, match="timed out"):
        await backend.generate("hello")


@respx.mock
async def test_generate_model_not_found_404(backend):
    respx.post(API_URL).mock(return_value=httpx.Response(404, text="not found"))

    with pytest.raises(OllamaModelNotFound, match="ollama pull"):
        await backend.generate("hello")


@respx.mock
async def test_generate_model_not_found_via_error_field(backend):
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "error": "model 'foo' not found",
    }))

    with pytest.raises(OllamaModelNotFound):
        await backend.generate("hello")


@respx.mock
async def test_generate_server_error(backend):
    respx.post(API_URL).mock(return_value=httpx.Response(500, text="internal"))

    with pytest.raises(OllamaServerError, match="500"):
        await backend.generate("hello")


@respx.mock
async def test_generate_malformed_json(backend):
    respx.post(API_URL).mock(return_value=httpx.Response(200, text="not json"))

    with pytest.raises(OllamaMalformedResponse, match="not valid JSON"):
        await backend.generate("hello")


@respx.mock
async def test_generate_missing_response_field(backend):
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={"other": "data"}))

    with pytest.raises(OllamaMalformedResponse, match="response"):
        await backend.generate("hello")


# --- list_models ---


@respx.mock
async def test_list_models(backend):
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={
        "models": [{"name": "llama3.1"}, {"name": "gemma4-32k"}],
    }))

    models = await backend.list_models()
    assert models == ["gemma4-32k", "llama3.1"]


@respx.mock
async def test_list_models_empty(backend):
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))

    assert await backend.list_models() == []


@respx.mock
async def test_list_models_connection_error(backend):
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OllamaConnectionError):
        await backend.list_models()
