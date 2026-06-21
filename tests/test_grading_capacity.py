"""Tests for Ollama capacity detection."""

import httpx
import pytest
import respx

from ollama_mcp.grading.capacity import (
    check_capacity,
    get_running_models,
    suggest_grading_config,
)

PS_URL = "http://localhost:11434/api/ps"


@respx.mock
async def test_get_running_models_with_models():
    respx.get(PS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "gemma4-32k", "size": 5_000_000_000, "size_vram": 4_500_000_000}
                ]
            },
        )
    )
    models = await get_running_models("http://localhost:11434")
    assert len(models) == 1
    assert models[0]["name"] == "gemma4-32k"


@respx.mock
async def test_get_running_models_empty():
    respx.get(PS_URL).mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    models = await get_running_models("http://localhost:11434")
    assert models == []


@respx.mock
async def test_get_running_models_connection_error():
    respx.get(PS_URL).mock(side_effect=httpx.ConnectError("refused"))
    models = await get_running_models("http://localhost:11434")
    assert models == []


@respx.mock
async def test_capacity_no_models_loaded():
    respx.get(PS_URL).mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    cap = await check_capacity("http://localhost:11434", "gemma4-32k")
    assert cap["can_grade_locally"] is True
    assert "freely" in cap["reason"]


@respx.mock
async def test_capacity_grading_model_already_loaded():
    respx.get(PS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "gemma4-32k:latest", "size_vram": 4_000_000_000}]},
        )
    )
    cap = await check_capacity("http://localhost:11434", "gemma4-32k")
    assert cap["can_grade_locally"] is True
    assert "already loaded" in cap["reason"]


@respx.mock
async def test_capacity_one_model_loaded():
    respx.get(PS_URL).mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "llama3:8b", "size_vram": 6_000_000_000}]},
        )
    )
    cap = await check_capacity("http://localhost:11434", "gemma4-32k")
    assert cap["can_grade_locally"] is True
    assert "warning" in cap


@respx.mock
async def test_capacity_two_models_loaded():
    respx.get(PS_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "models": [
                    {"name": "llama3:8b", "size_vram": 6_000_000_000},
                    {"name": "mistral:7b", "size_vram": 5_000_000_000},
                ]
            },
        )
    )
    cap = await check_capacity("http://localhost:11434", "gemma4-32k")
    assert cap["can_grade_locally"] is False
    assert "eviction" in cap["reason"]


def test_suggest_config_when_can_grade():
    cap = {"can_grade_locally": True, "grading_model": "x", "reason": "ok"}
    assert suggest_grading_config(cap) is None


def test_suggest_config_when_cannot_grade():
    cap = {
        "can_grade_locally": False,
        "grading_model": "gemma4-32k",
        "reason": "2 models loaded",
    }
    suggestion = suggest_grading_config(cap)
    assert suggestion is not None
    assert "openrouter" in suggestion.lower()
    assert "OLLAMA_MCP_GRADING=0" in suggestion
