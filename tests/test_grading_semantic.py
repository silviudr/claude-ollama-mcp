"""Tests for semantic LLM grading."""

import json

import httpx
import pytest
import respx

from ollama_mcp.backends.openrouter import OpenRouterBackend
from ollama_mcp.grading.semantic import SemanticGrade, _parse_grade, grade_output

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def test_parse_grade_valid():
    raw = json.dumps({
        "correctness": 4,
        "completeness": 3,
        "format": 5,
        "conciseness": 4,
        "overall": 0.8,
        "issues": [],
    })
    grade = _parse_grade(raw)
    assert grade is not None
    assert grade.overall == 0.8
    assert grade.correctness == 4


def test_parse_grade_with_fences():
    raw = '```json\n{"correctness": 3, "completeness": 3, "format": 4, "conciseness": 3, "overall": 0.65, "issues": ["minor"]}\n```'
    grade = _parse_grade(raw)
    assert grade is not None
    assert grade.overall == 0.65
    assert grade.issues == ["minor"]


def test_parse_grade_invalid():
    grade = _parse_grade("not json at all")
    assert grade is None


def test_parse_grade_out_of_range():
    raw = json.dumps({
        "correctness": 10,
        "completeness": 3,
        "format": 4,
        "conciseness": 3,
        "overall": 0.5,
        "issues": [],
    })
    grade = _parse_grade(raw)
    assert grade is None


def test_semantic_grade_model():
    g = SemanticGrade(
        correctness=4, completeness=3, format=5, conciseness=4,
        overall=0.8, issues=["minor style issue"],
    )
    assert g.overall == 0.8
    d = g.model_dump()
    assert "correctness" in d
    assert d["issues"] == ["minor style issue"]


@respx.mock
async def test_grade_output_success():
    grade_json = json.dumps({
        "correctness": 4, "completeness": 4, "format": 5,
        "conciseness": 4, "overall": 0.85, "issues": [],
    })
    respx.post(OR_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": grade_json}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )
    )
    backend = OpenRouterBackend(api_key="test-key")
    result = await grade_output(
        backend, "local_implement_small",
        "write a function", "def add(a, b): return a + b",
    )
    assert result["checker"] == "llm_judge"
    assert result["passed"] is True
    assert result["score"] == 0.85
    assert result["grader_backend"] == "openrouter"


@respx.mock
async def test_grade_output_unparseable():
    respx.post(OR_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "I cannot grade this."}}],
                "usage": {},
            },
        )
    )
    backend = OpenRouterBackend(api_key="test-key")
    result = await grade_output(
        backend, "local_summarize", "input", "output",
    )
    assert result["checker"] == "llm_judge"
    assert result["passed"] is None
    assert result["score"] is None
    assert "parse_error" in result["details"]


@respx.mock
async def test_grade_output_backend_error():
    respx.post(OR_URL).mock(
        return_value=httpx.Response(500, text="Internal Server Error")
    )
    backend = OpenRouterBackend(api_key="test-key")
    result = await grade_output(
        backend, "local_summarize", "input", "output",
    )
    assert result["checker"] == "llm_judge"
    assert result["passed"] is None
    assert "error" in result["details"]


@respx.mock
async def test_grade_output_low_score_fails():
    grade_json = json.dumps({
        "correctness": 1, "completeness": 1, "format": 2,
        "conciseness": 1, "overall": 0.25, "issues": ["very poor"],
    })
    respx.post(OR_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": grade_json}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )
    )
    backend = OpenRouterBackend(api_key="test-key")
    result = await grade_output(
        backend, "local_summarize", "input", "output",
    )
    assert result["passed"] is False
    assert result["score"] == 0.25
