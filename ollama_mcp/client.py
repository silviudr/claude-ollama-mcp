"""Ollama HTTP client."""

import json
import re
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import MODEL, OLLAMA_URL
from .errors import (
    OllamaConnectionError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)

T = TypeVar("T", bound=BaseModel)

TIMEOUT_S = 180


async def generate(
    prompt: str,
    system: str | None = None,
    model: str | None = None,
) -> tuple[str, dict]:
    use_model = model or MODEL
    payload: dict = {"model": use_model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
            r = await c.post(f"{OLLAMA_URL}/api/generate", json=payload)
    except httpx.ConnectError:
        raise OllamaConnectionError()
    except httpx.TimeoutException:
        raise OllamaTimeout(TIMEOUT_S)

    if r.status_code == 404:
        raise OllamaModelNotFound(use_model)
    if r.status_code >= 400:
        raise OllamaServerError(r.status_code, r.text[:200])

    try:
        data = r.json()
    except ValueError:
        raise OllamaMalformedResponse("response is not valid JSON")

    if "error" in data:
        msg = data["error"]
        if "not found" in msg.lower():
            raise OllamaModelNotFound(use_model)
        raise OllamaServerError(r.status_code, msg)

    if "response" not in data:
        raise OllamaMalformedResponse("missing 'response' field")

    return data["response"], {
        "wall_ms": int((time.perf_counter() - t0) * 1000),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "eval_ms": (data.get("eval_duration") or 0) // 1_000_000,
        "model": use_model,
    }


async def list_models() -> list[str]:
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        raise OllamaConnectionError()
    except httpx.TimeoutException:
        raise OllamaTimeout(10)
    return sorted(m["name"] for m in data.get("models", []))


_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


async def generate_json(
    prompt: str,
    schema: type[T],
    system: str | None = None,
) -> tuple[T | None, str, dict]:
    """Call generate() requesting JSON, validate against a Pydantic model.

    Returns (parsed_model, raw_text, meta).  If the model's output fails
    JSON parsing or schema validation, parsed_model is None and raw_text
    contains the original response so callers can fall back gracefully.
    """
    schema_hint = json.dumps(schema.model_json_schema(), indent=2)
    json_instruction = (
        "\n\nYou MUST respond with valid JSON matching this schema — "
        "no markdown fences, no prose before or after:\n"
        f"{schema_hint}"
    )
    full_system = (system or "") + json_instruction

    raw, meta = await generate(prompt, system=full_system)

    cleaned = _strip_fences(raw)
    try:
        parsed = schema.model_validate_json(cleaned)
        return parsed, raw, meta
    except (json.JSONDecodeError, ValidationError):
        return None, raw, meta
