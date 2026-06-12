"""Ollama backend — local inference via the Ollama HTTP API."""

from __future__ import annotations

import time

import httpx

from ..errors import (
    OllamaConnectionError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)
from .base import Backend

TIMEOUT_S = 180


class OllamaBackend(Backend):
    def __init__(self, url: str, default_model: str):
        self.name = "ollama"
        self.url = url
        self.default_model = default_model

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> tuple[str, dict]:
        use_model = model or self.default_model
        payload: dict = {"model": use_model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
                r = await c.post(f"{self.url}/api/generate", json=payload)
        except httpx.ConnectError:
            raise OllamaConnectionError(self.url)
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
            "backend": "ollama",
        }

    async def list_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self.url}/api/tags")
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError:
            raise OllamaConnectionError(self.url)
        except httpx.TimeoutException:
            raise OllamaTimeout(10)
        return sorted(m["name"] for m in data.get("models", []))
