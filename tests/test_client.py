import json

import httpx
import pytest
import respx

from ollama_mcp.client import generate, generate_json, list_models
from ollama_mcp.config import MODEL, OLLAMA_URL
from ollama_mcp.errors import (
    OllamaConnectionError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)
from ollama_mcp.schemas import ReviewResult

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


# --- generate_json tests ---


@respx.mock
async def test_generate_json_valid():
    valid_json = json.dumps({
        "findings": [
            {"severity": "HIGH", "category": "BUG", "message": "null deref",
             "file": "app.py", "line": 42}
        ],
        "summary": "1 finding (1 high)",
    })
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": valid_json,
    }))

    parsed, raw, meta = await generate_json("review this", ReviewResult)

    assert parsed is not None
    assert len(parsed.findings) == 1
    assert parsed.findings[0].severity == "HIGH"
    assert parsed.summary == "1 finding (1 high)"


@respx.mock
async def test_generate_json_with_markdown_fences():
    valid_json = json.dumps({
        "findings": [],
        "summary": "No issues found.",
    })
    fenced = f"```json\n{valid_json}\n```"
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": fenced,
    }))

    parsed, raw, meta = await generate_json("review this", ReviewResult)

    assert parsed is not None
    assert parsed.findings == []


@respx.mock
async def test_generate_json_invalid_json_returns_none():
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "This is not JSON at all, just plain text review.",
    }))

    parsed, raw, meta = await generate_json("review this", ReviewResult)

    assert parsed is None
    assert "not JSON" in raw


@respx.mock
async def test_generate_json_wrong_schema_returns_none():
    wrong_shape = json.dumps({"wrong_field": "data"})
    respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": wrong_shape,
    }))

    parsed, raw, meta = await generate_json("review this", ReviewResult)

    assert parsed is None
    assert raw == wrong_shape


@respx.mock
async def test_generate_json_includes_schema_in_system():
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": json.dumps({"findings": [], "summary": "clean"}),
    }))

    await generate_json("test", ReviewResult, system="Be strict.")

    payload = json.loads(route.calls[0].request.read())
    assert "Be strict." in payload["system"]
    assert "JSON" in payload["system"]
    assert "schema" in payload["system"].lower()


# --- model override tests ---


TAGS_URL = f"{OLLAMA_URL}/api/tags"


@respx.mock
async def test_generate_with_model_override():
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "ok",
    }))

    text, meta = await generate("test", model="custom-model")

    payload = json.loads(route.calls[0].request.read())
    assert payload["model"] == "custom-model"
    assert meta["model"] == "custom-model"


@respx.mock
async def test_generate_uses_default_model_when_none():
    route = respx.post(API_URL).mock(return_value=httpx.Response(200, json={
        "response": "ok",
    }))

    text, meta = await generate("test")

    payload = json.loads(route.calls[0].request.read())
    assert payload["model"] == MODEL


# --- list_models tests ---


@respx.mock
async def test_list_models():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={
        "models": [
            {"name": "llama3.1"},
            {"name": "gemma4-32k"},
            {"name": "deepseek-coder"},
        ],
    }))

    models = await list_models()

    assert models == ["deepseek-coder", "gemma4-32k", "llama3.1"]


@respx.mock
async def test_list_models_empty():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={
        "models": [],
    }))

    models = await list_models()
    assert models == []


@respx.mock
async def test_list_models_connection_error():
    respx.get(TAGS_URL).mock(side_effect=httpx.ConnectError("refused"))

    with pytest.raises(OllamaConnectionError):
        await list_models()
