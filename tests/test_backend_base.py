"""Tests for the Backend ABC and its concrete generate_json / _strip_fences."""

import json

import pytest
from pydantic import BaseModel

from ollama_mcp.backends.base import Backend


class DummyResult(BaseModel):
    summary: str
    count: int = 0


class FakeBackend(Backend):
    """Minimal concrete backend for testing the base class methods."""

    def __init__(self, response: str = "ok", meta: dict | None = None):
        self.name = "fake"
        self.default_model = "fake-model"
        self._response = response
        self._meta = meta or {"wall_ms": 1, "prompt_tokens": 5, "output_tokens": 3,
                               "eval_ms": 0, "model": "fake-model", "backend": "fake"}

    async def generate(self, prompt, system=None, model=None):
        return self._response, self._meta

    async def list_models(self):
        return ["fake-model"]


# --- _strip_fences ---


def test_strip_fences_json_block():
    text = '```json\n{"a": 1}\n```'
    assert Backend._strip_fences(text) == '{"a": 1}'


def test_strip_fences_plain_block():
    text = '```\nhello world\n```'
    assert Backend._strip_fences(text) == "hello world"


def test_strip_fences_no_fences():
    text = '{"a": 1}'
    assert Backend._strip_fences(text) == '{"a": 1}'


def test_strip_fences_whitespace():
    text = '  \n{"a": 1}\n  '
    assert Backend._strip_fences(text) == '{"a": 1}'


# --- generate_json ---


async def test_generate_json_valid():
    valid = json.dumps({"summary": "looks good", "count": 42})
    backend = FakeBackend(response=valid)

    parsed, raw, meta = await backend.generate_json("test", DummyResult)

    assert parsed is not None
    assert parsed.summary == "looks good"
    assert parsed.count == 42
    assert raw == valid


async def test_generate_json_with_fences():
    valid = json.dumps({"summary": "ok", "count": 1})
    fenced = f"```json\n{valid}\n```"
    backend = FakeBackend(response=fenced)

    parsed, raw, meta = await backend.generate_json("test", DummyResult)

    assert parsed is not None
    assert parsed.summary == "ok"


async def test_generate_json_invalid_json_returns_none():
    backend = FakeBackend(response="this is not JSON at all")

    parsed, raw, meta = await backend.generate_json("test", DummyResult)

    assert parsed is None
    assert "not JSON" in raw


async def test_generate_json_wrong_schema_returns_none():
    wrong = json.dumps({"wrong_field": "data"})
    backend = FakeBackend(response=wrong)

    parsed, raw, meta = await backend.generate_json("test", DummyResult)

    assert parsed is None
    assert raw == wrong


async def test_generate_json_injects_schema_in_system():
    calls = []

    class SpyBackend(FakeBackend):
        async def generate(self, prompt, system=None, model=None):
            calls.append({"prompt": prompt, "system": system, "model": model})
            return json.dumps({"summary": "ok"}), self._meta

    backend = SpyBackend()
    await backend.generate_json("test", DummyResult, system="Be strict.")

    assert len(calls) == 1
    assert "Be strict." in calls[0]["system"]
    assert "JSON" in calls[0]["system"]


async def test_generate_json_passes_model():
    calls = []

    class SpyBackend(FakeBackend):
        async def generate(self, prompt, system=None, model=None):
            calls.append(model)
            return json.dumps({"summary": "ok"}), self._meta

    backend = SpyBackend()
    await backend.generate_json("test", DummyResult, model="custom-model")

    assert calls == ["custom-model"]
