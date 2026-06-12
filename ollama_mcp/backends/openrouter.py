"""OpenRouter backend — cloud inference via the OpenAI-compatible API."""

from __future__ import annotations

import os
import time

import httpx

from ..errors import (
    OpenRouterAuthError,
    OpenRouterConnectionError,
    OpenRouterModelNotFound,
    OpenRouterRateLimitError,
    OpenRouterServerError,
    OpenRouterTimeout,
)
from .base import Backend

DEFAULT_URL = "https://openrouter.ai/api/v1"
TIMEOUT_S = 120


class OpenRouterBackend(Backend):
    def __init__(
        self,
        api_key_env: str = "OPENROUTER_API_KEY",
        default_model: str = "google/gemma-3-27b-it:free",
        url: str = DEFAULT_URL,
        api_key: str | None = None,
    ):
        self.name = "openrouter"
        self.api_key_env = api_key_env
        self.default_model = default_model
        self.url = url
        self._api_key = api_key

    def _get_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        key = os.environ.get(self.api_key_env)
        if not key:
            raise OpenRouterAuthError(
                f"Environment variable {self.api_key_env} is not set"
            )
        return key

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> tuple[str, dict]:
        use_model = model or self.default_model

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {"model": use_model, "messages": messages}
        headers = {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_S) as c:
                r = await c.post(
                    f"{self.url}/chat/completions",
                    json=payload,
                    headers=headers,
                )
        except httpx.ConnectError:
            raise OpenRouterConnectionError()
        except httpx.TimeoutException:
            raise OpenRouterTimeout(TIMEOUT_S)

        if r.status_code == 401:
            raise OpenRouterAuthError("Invalid API key")
        if r.status_code == 429:
            raise OpenRouterRateLimitError()
        if r.status_code == 404:
            raise OpenRouterModelNotFound(use_model)
        if r.status_code >= 400:
            raise OpenRouterServerError(r.status_code, r.text[:200])

        try:
            data = r.json()
        except ValueError:
            raise OpenRouterServerError(r.status_code, "response is not valid JSON")

        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise OpenRouterServerError(
                r.status_code, "unexpected response structure"
            )

        usage = data.get("usage", {})

        meta = {
            "wall_ms": int((time.perf_counter() - t0) * 1000),
            "prompt_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "eval_ms": 0,
            "model": use_model,
            "backend": "openrouter",
        }

        cost = usage.get("cost") or data.get("total_cost")
        if cost is not None:
            try:
                meta["cost"] = float(cost)
            except (TypeError, ValueError):
                pass

        return text, meta

    async def list_models(self) -> list[str]:
        headers = {"Authorization": f"Bearer {self._get_api_key()}"}
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.get(f"{self.url}/models", headers=headers)
                r.raise_for_status()
                data = r.json()
        except httpx.ConnectError:
            raise OpenRouterConnectionError()
        except httpx.TimeoutException:
            raise OpenRouterTimeout(10)
        return sorted(m["id"] for m in data.get("data", []))
