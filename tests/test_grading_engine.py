"""Tests for the grading engine."""

import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from ollama_mcp.grading.engine import (
    _build_grading_backend,
    _grade,
    _load_grading_config,
    schedule_grading,
)

OR_URL = "https://openrouter.ai/api/v1/chat/completions"
PS_URL = "http://localhost:11434/api/ps"


class TestLoadGradingConfig:
    def test_no_config_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH",
            tmp_path / "nonexistent.json",
        )
        assert _load_grading_config() == {}

    def test_config_without_grading(self, tmp_path, monkeypatch):
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({"default_backend": "ollama"}))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        assert _load_grading_config() == {}

    def test_config_with_grading(self, tmp_path, monkeypatch):
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "backend": "openrouter",
                "model": "google/gemma-3-27b-it:free",
                "sample_rate": 0.1,
            }
        }))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        cfg = _load_grading_config()
        assert cfg["backend"] == "openrouter"
        assert cfg["sample_rate"] == 0.1


class TestBuildGradingBackend:
    def test_openrouter_backend(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        result = _build_grading_backend({"backend": "openrouter"})
        assert result is not None
        backend, model = result
        assert backend.name == "openrouter"

    def test_ollama_backend(self):
        result = _build_grading_backend({
            "backend": "ollama",
            "model": "gemma4-32k",
        })
        assert result is not None
        backend, model = result
        assert backend.name == "ollama"
        assert model == "gemma4-32k"

    def test_unknown_backend(self):
        result = _build_grading_backend({"backend": "unknown"})
        assert result is None


class TestGrade:
    @respx.mock
    async def test_heuristics_only_when_not_sampled(self, tmp_path, monkeypatch):
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {"enabled": True, "sample_rate": 0.0}
        }))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

        await _grade(
            "local_implement_small", "spec", "def add(a, b): return a + b", 1
        )

        assert len(grades_logged) > 0
        assert all(g["grade_type"] == "heuristic" for g in grades_logged)

    @respx.mock
    async def test_semantic_when_sampled(self, tmp_path, monkeypatch):
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "sample_rate": 1.0,
                "backend": "openrouter",
                "model": "test-model",
            }
        }))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

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

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        await _grade(
            "local_implement_small", "spec", "def add(a, b): return a + b", 1
        )

        semantic = [g for g in grades_logged if g["grade_type"] == "semantic"]
        assert len(semantic) == 1
        assert semantic[0]["score"] == 0.85

    @respx.mock
    async def test_disabled_grading(self, tmp_path, monkeypatch):
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({"grading": {"enabled": False}}))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

        await _grade("local_summarize", "input", "output", 1)
        assert len(grades_logged) == 0


    @respx.mock
    async def test_per_tool_sample_rate_overrides_global(self, tmp_path, monkeypatch):
        """A tool with sample_rate 0.0 in tool_sample_rates skips semantic
        even when global sample_rate is 1.0."""
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "sample_rate": 1.0,
                "tool_sample_rates": {
                    "local_summarize": 0.0,
                },
            }
        }))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

        await _grade("local_summarize", "input", "output summary", 1)

        assert all(g["grade_type"] == "heuristic" for g in grades_logged)

    @respx.mock
    async def test_per_tool_sample_rate_enables_semantic(self, tmp_path, monkeypatch):
        """A tool with sample_rate 1.0 in tool_sample_rates gets semantic
        even when global sample_rate is 0.0."""
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "sample_rate": 0.0,
                "backend": "openrouter",
                "model": "test-model",
                "tool_sample_rates": {
                    "local_implement_small": 1.0,
                },
            }
        }))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

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

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        await _grade(
            "local_implement_small", "spec", "def add(a, b): return a + b", 1
        )

        semantic = [g for g in grades_logged if g["grade_type"] == "semantic"]
        assert len(semantic) == 1

    @respx.mock
    async def test_tool_without_override_uses_global(self, tmp_path, monkeypatch):
        """A tool not listed in tool_sample_rates falls back to global rate."""
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "sample_rate": 0.0,
                "tool_sample_rates": {
                    "local_implement_small": 1.0,
                },
            }
        }))
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f
        )
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

        await _grade("local_summarize", "input", "output summary", 1)

        assert all(g["grade_type"] == "heuristic" for g in grades_logged)

    @respx.mock
    async def test_swarm_subtask_base_rate_enables_semantic(self, tmp_path, monkeypatch):
        """Base-tool rate 1.0 must reach a subtask even when global is 0.0 —
        the exact case that silently degraded before."""
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "sample_rate": 0.0,
                "backend": "openrouter",
                "model": "test-model",
                "tool_sample_rates": {"local_review_diff": 1.0},
            }
        }))
        monkeypatch.setattr("ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f)
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

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

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        await _grade("local_review_diff:security", "diff", "a finding", 1)

        semantic = [g for g in grades_logged if g["grade_type"] == "semantic"]
        assert len(semantic) == 1

    @respx.mock
    async def test_exact_subtask_rate_beats_base_tool_rate(self, tmp_path, monkeypatch):
        """An exact synthetic-name entry still wins over the base-tool one."""
        f = tmp_path / "routes.json"
        f.write_text(json.dumps({
            "grading": {
                "enabled": True,
                "sample_rate": 1.0,
                "tool_sample_rates": {
                    "local_review_diff": 1.0,
                    "local_review_diff:security": 0.0,
                },
            }
        }))
        monkeypatch.setattr("ollama_mcp.grading.engine.ROUTES_CONFIG_PATH", f)
        monkeypatch.setattr("ollama_mcp.grading.engine._capacity_checked", True)

        grades_logged = []
        monkeypatch.setattr(
            "ollama_mcp.grading.engine.log_grade",
            lambda g: grades_logged.append(g),
        )

        await _grade("local_review_diff:security", "diff", "a finding", 1)

        assert all(g["grade_type"] == "heuristic" for g in grades_logged)


class TestScheduleGrading:
    async def test_schedule_creates_task(self, monkeypatch):
        graded = []

        async def mock_grade(tool, inp, out, cid):
            graded.append(tool)

        monkeypatch.setattr("ollama_mcp.grading.engine._grade", mock_grade)

        schedule_grading("local_summarize", "input", "output", 1)

        import asyncio
        await asyncio.sleep(0.05)

        assert "local_summarize" in graded

    def test_schedule_no_loop(self, monkeypatch):
        """schedule_grading silently skips when no event loop is running."""
        monkeypatch.setattr(
            "ollama_mcp.grading.engine._grade",
            AsyncMock(),
        )
        # Outside an async context, this should not raise
        # Note: this test itself runs outside async since it's not async def
        # The function should handle RuntimeError gracefully
        # We test this by calling from a sync context
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            # Temporarily unset the running loop
            schedule_grading("local_summarize", "input", "output", 1)
        finally:
            loop.close()
