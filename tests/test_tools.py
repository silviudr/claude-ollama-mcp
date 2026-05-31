import json

import httpx
import pytest
import respx

from ollama_mcp import storage
from ollama_mcp.config import OLLAMA_URL
from ollama_mcp.tools import (
    local_commit_message,
    local_draft_boilerplate,
    local_generate_tests,
    local_implement_small,
    local_review_diff,
    local_summarize,
    local_usage_stats,
)

API_URL = f"{OLLAMA_URL}/api/generate"


def _mock_ollama(response_text: str = "ok"):
    return respx.post(API_URL).mock(
        return_value=httpx.Response(200, json={"response": response_text})
    )


@respx.mock
async def test_local_summarize():
    _mock_ollama("short summary")
    result = await local_summarize("a very long document...")
    assert result == "short summary"


@respx.mock
async def test_local_draft_boilerplate():
    _mock_ollama("FROM python:3.11")
    result = await local_draft_boilerplate("Dockerfile for Python 3.11")
    assert result == "FROM python:3.11"


@respx.mock
async def test_local_implement_small():
    _mock_ollama("def add(a, b): return a + b")
    result = await local_implement_small("a function that adds two numbers")
    assert result == "def add(a, b): return a + b"


@respx.mock
async def test_local_commit_message():
    _mock_ollama("fix: resolve null pointer in parser")
    result = await local_commit_message("--- a/foo.py\n+++ b/foo.py")
    assert result == "fix: resolve null pointer in parser"


@respx.mock
async def test_local_review_diff_basic():
    findings = "[HIGH] BUG: off-by-one error (foo.py:10)\nSummary: 1 findings (1 high, 0 medium, 0 low)"
    _mock_ollama(findings)
    result = await local_review_diff("--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-x[len(x)]\n+x[len(x)-1]")
    assert "BUG" in result
    assert "Summary" in result


@respx.mock
async def test_local_review_diff_with_focus():
    route = _mock_ollama("No issues found.")
    await local_review_diff("some diff", focus="security,performance")

    payload = json.loads(route.calls[0].request.read())
    assert "security" in payload["system"].lower()
    assert "performance" in payload["system"].lower()


@respx.mock
async def test_local_review_diff_empty_focus():
    route = _mock_ollama("No issues found.")
    await local_review_diff("some diff", focus="")

    payload = json.loads(route.calls[0].request.read())
    assert "Focus especially on" not in payload["system"]


@respx.mock
async def test_local_generate_tests_basic():
    test_code = "import pytest\nfrom foo import bar\n\ndef test_bar_returns_int():\n    assert isinstance(bar(1), int)"
    _mock_ollama(test_code)
    result = await local_generate_tests("def bar(x):\n    return x * 2")
    assert "def test_" in result
    assert "import" in result


@respx.mock
async def test_local_generate_tests_with_context():
    route = _mock_ollama("def test_something(): pass")
    await local_generate_tests("def foo(): pass", context="This is a math utility")

    payload = json.loads(route.calls[0].request.read())
    assert "math utility" in payload["system"]


@respx.mock
async def test_local_generate_tests_empty_context():
    route = _mock_ollama("def test_something(): pass")
    await local_generate_tests("def foo(): pass", context="")

    payload = json.loads(route.calls[0].request.read())
    assert "Additional context" not in payload["system"]


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    if hasattr(storage._local, "conn"):
        del storage._local.conn


async def test_local_usage_stats_empty():
    result = await local_usage_stats()
    assert result == "No local tool calls recorded yet."


async def test_local_usage_stats_with_data():
    import time
    storage.log_call({
        "ts": time.time(), "tool": "local_summarize", "ok": True,
        "model": "gemma4-32k", "input_chars": 100, "output_chars": 50,
        "prompt_tokens": 500, "output_tokens": 200,
        "total_ms": 1500, "wall_ms": 1400, "eval_ms": 1200,
    })
    storage.log_call({
        "ts": time.time(), "tool": "local_commit_message", "ok": True,
        "model": "gemma4-32k", "input_chars": 80, "output_chars": 30,
        "prompt_tokens": 300, "output_tokens": 100,
        "total_ms": 800, "wall_ms": 700, "eval_ms": 600,
    })
    result = await local_usage_stats()
    assert "Local calls:   2" in result
    assert "Successful:    2" in result
    assert "Prompt tokens:  800" in result
    assert "Output tokens:  300" in result
    assert "Opus:" in result
    assert "Sonnet:" in result
    assert "local_summarize" in result
    assert "local_commit_message" in result
