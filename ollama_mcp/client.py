"""Ollama HTTP client."""

import time

import httpx

from .config import MODEL, OLLAMA_URL
from .errors import (
    OllamaConnectionError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)

TIMEOUT_S = 180


async def generate(prompt: str, system: str | None = None) -> tuple[str, dict]:
    payload: dict = {"model": MODEL, "prompt": prompt, "stream": False}
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
        raise OllamaModelNotFound(MODEL)
    if r.status_code >= 400:
        raise OllamaServerError(r.status_code, r.text[:200])

    try:
        data = r.json()
    except ValueError:
        raise OllamaMalformedResponse("response is not valid JSON")

    if "error" in data:
        msg = data["error"]
        if "not found" in msg.lower():
            raise OllamaModelNotFound(MODEL)
        raise OllamaServerError(r.status_code, msg)

    if "response" not in data:
        raise OllamaMalformedResponse("missing 'response' field")

    return data["response"], {
        "wall_ms": int((time.perf_counter() - t0) * 1000),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "eval_ms": (data.get("eval_duration") or 0) // 1_000_000,
        "model": MODEL,
    }
