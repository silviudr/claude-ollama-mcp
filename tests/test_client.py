import httpx
import pytest
import respx

from ollama_mcp.client import generate
from ollama_mcp.config import MODEL, OLLAMA_URL
from ollama_mcp.errors import (
    OllamaConnectionError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)

API_URL = f"{OLLAMA_URL}/api/generate"


@respx.mock
async def test_success():
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "hello world",
        "prompt_eval_count": 10,
        "eval_count": 5,
        "eval_duration": 100_000_000,
    }))

    text, meta = await generate("say hello")

    assert text == "hello world"
    assert meta["model"] == MODEL
    assert meta["prompt_tokens"] == 10
    assert meta["output_tokens"] == 5
    assert "wall_ms" in meta


@respx.mock
async def test_system_prompt_included():
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "ok",
    }))

    await generate("test", system="be concise")

    sent = route.calls[0].request
    body = sent.read()
    import json
    payload = json.loads(body)
    assert payload["system"] == "be concise"


@respx.mock
async def test_connection_error():
    respx.post(API_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OllamaConnectionError, match="ollama serve"):
        await generate("hello")


@respx.mock
async def test_timeout():
    respx.post(API_URL).mock(side_effect=httpx.ReadTimeout("timed out"))

    with pytest.raises(OllamaTimeout, match="timed out"):
        await generate("hello")


@respx.mock
async def test_model_not_found_via_404():
    respx.post(API_URL).mock(return_value=httpx.Response(404, text="not found"))

    with pytest.raises(OllamaModelNotFound, match="ollama pull"):
        await generate("hello")


@respx.mock
async def test_model_not_found_via_error_field():
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "error": "model 'foo' not found",
    }))

    with pytest.raises(OllamaModelNotFound, match="ollama pull"):
        await generate("hello")


@respx.mock
async def test_server_error_500():
    respx.post(API_URL).mock(return_value=httpx.Response(500, text="internal error"))

    with pytest.raises(OllamaServerError, match="500"):
        await generate("hello")


@respx.mock
async def test_malformed_json():
    respx.post(API_URL).mock(return_value=httpx.Response(200, text="not json"))

    with pytest.raises(OllamaMalformedResponse, match="not valid JSON"):
        await generate("hello")


@respx.mock
async def test_missing_response_field():
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "something_else": "data",
    }))

    with pytest.raises(OllamaMalformedResponse, match="response"):
        await generate("hello")


@respx.mock
async def test_generic_error_in_body():
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "error": "GPU out of memory",
    }))

    with pytest.raises(OllamaServerError, match="GPU out of memory"):
        await generate("hello")
