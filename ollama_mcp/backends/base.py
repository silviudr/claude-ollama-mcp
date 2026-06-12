"""Abstract base class for LLM backends."""

from __future__ import annotations

import abc
import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


class Backend(abc.ABC):
    """Abstract backend for LLM inference."""

    name: str
    default_model: str

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
    ) -> tuple[str, dict]:
        """Generate text completion.

        Returns (text, meta_dict) where meta_dict includes at minimum:
        wall_ms, prompt_tokens, output_tokens, eval_ms, model, backend.
        """
        ...

    @abc.abstractmethod
    async def list_models(self) -> list[str]:
        """List available models for this backend."""
        ...

    async def generate_json(
        self,
        prompt: str,
        schema: type[T],
        system: str | None = None,
        model: str | None = None,
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

        raw, meta = await self.generate(prompt, system=full_system, model=model)

        cleaned = self._strip_fences(raw)
        try:
            parsed = schema.model_validate_json(cleaned)
            return parsed, raw, meta
        except (json.JSONDecodeError, ValidationError):
            return None, raw, meta

    @staticmethod
    def _strip_fences(text: str) -> str:
        m = _FENCE_RE.search(text)
        return m.group(1).strip() if m else text.strip()
