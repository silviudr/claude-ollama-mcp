import json

import httpx
import pytest
import respx

from ollama_mcp import storage
from ollama_mcp.config import OLLAMA_URL
from ollama_mcp.tools import (
    local_analyze_data,
    local_benchmark,
    local_classify_task,
    local_commit_message,
    local_draft_boilerplate,
    local_generate_tests,
    local_implement_small,
    local_list_models,
    local_review_diff,
    local_show_routes,
    local_summarize,
    local_usage_stats,
)

API_URL = f"{OLLAMA_URL}/api/generate"
TAGS_URL = f"{OLLAMA_URL}/api/tags"


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
async def test_local_review_diff_structured():
    review_json = json.dumps({
        "findings": [
            {"severity": "HIGH", "category": "BUG",
             "message": "off-by-one error", "file": "foo.py", "line": 10}
        ],
        "summary": "1 finding (1 high, 0 medium, 0 low)",
    })
    _mock_ollama(review_json)
    result = await local_review_diff("--- a/foo.py\n+++ b/foo.py")
    assert "[HIGH] BUG: off-by-one error (foo.py:10)" in result
    assert "1 finding" in result


@respx.mock
async def test_local_review_diff_fallback_on_invalid_json():
    _mock_ollama("Some plain text review with no JSON structure.")
    result = await local_review_diff("some diff")
    assert "plain text review" in result


@respx.mock
async def test_local_review_diff_with_focus():
    review_json = json.dumps({"findings": [], "summary": "clean"})
    route = _mock_ollama(review_json)
    await local_review_diff("some diff", focus="security,performance")

    payload = json.loads(route.calls[0].request.read())
    assert "security" in payload["system"].lower()
    assert "performance" in payload["system"].lower()


@respx.mock
async def test_local_review_diff_empty_focus():
    review_json = json.dumps({"findings": [], "summary": "clean"})
    route = _mock_ollama(review_json)
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


@respx.mock
async def test_local_benchmark_explicit_models():
    respx.post(API_URL).mock(side_effect=[
        httpx.Response(200, json={"response": "a1", "prompt_eval_count": 10, "eval_count": 5, "eval_duration": 50_000_000}),
        httpx.Response(200, json={"response": "a2", "prompt_eval_count": 12, "eval_count": 8, "eval_duration": 80_000_000}),
    ])
    result = await local_benchmark("say hello", models="model-a,model-b")
    assert "model-a" in result
    assert "model-b" in result


@respx.mock
async def test_local_benchmark_auto_discover():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={
        "models": [{"name": "alpha"}, {"name": "beta"}],
    }))
    respx.post(API_URL).mock(side_effect=[
        httpx.Response(200, json={"response": "ok", "prompt_eval_count": 10, "eval_count": 5, "eval_duration": 50_000_000}),
        httpx.Response(200, json={"response": "ok", "prompt_eval_count": 10, "eval_count": 5, "eval_duration": 50_000_000}),
    ])
    result = await local_benchmark("test")
    assert "alpha" in result
    assert "beta" in result


@respx.mock
async def test_local_benchmark_no_models():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))
    result = await local_benchmark("test")
    assert "No models available" in result


@respx.mock
async def test_local_list_models():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={
        "models": [{"name": "gemma4-32k"}, {"name": "llama3.1"}, {"name": "deepseek-coder"}],
    }))
    result = await local_list_models()
    assert "gemma4-32k" in result
    assert "llama3.1" in result
    assert "deepseek-coder" in result


@respx.mock
async def test_local_list_models_empty():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={"models": []}))
    result = await local_list_models()
    assert "No models found" in result
    assert "ollama pull" in result


async def test_local_show_routes(monkeypatch):
    import ollama_mcp.router as router
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", router.ROUTES_CONFIG_PATH.parent / "nonexistent")
    result = await local_show_routes()
    assert "No routing config" in result


async def test_local_show_routes_with_config(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default": "gemma4-32k",
        "routes": {"local_review_diff": "deepseek-coder"},
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)
    result = await local_show_routes()
    assert "deepseek-coder" in result
    assert "gemma4-32k" in result


@respx.mock
async def test_local_classify_task_structured():
    classification = json.dumps({
        "task_type": "review",
        "risk": "low",
        "recommended_tool": "local_review_diff",
        "recommended_model": "deepseek-coder",
        "should_use_local": True,
        "reasoning": "Simple code review",
    })
    _mock_ollama(classification)
    result = await local_classify_task("review this diff for bugs")
    assert "review" in result
    assert "local_review_diff" in result


@respx.mock
async def test_local_classify_task_fallback():
    _mock_ollama("I think this is a code review task.")
    result = await local_classify_task("review this diff")
    assert "code review" in result


# --- local_analyze_data tests ---


@respx.mock
async def test_local_analyze_data_simple(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,age\nAlice,30\nBob,25\n")
    analysis = json.dumps({
        "summary": "Small dataset with names and ages",
        "insights": [{"category": "distribution", "column": "age", "description": "ages range 25-30", "severity": "LOW"}],
        "row_count": 2,
        "col_count": 2,
    })
    _mock_ollama(analysis)
    result = await local_analyze_data(str(csv_path))
    assert "names and ages" in result


async def test_local_analyze_data_force_handoff(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,age\nAlice,30\nBob,25\n")
    result = await local_analyze_data(str(csv_path), force_handoff=True)
    assert "HANDOFF" in result
    assert "User requested" in result


async def test_local_analyze_data_auto_handoff(tmp_path):
    import csv as csv_mod
    csv_path = tmp_path / "wide.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.writer(f)
        headers = [f"col_{i}" for i in range(25)]
        writer.writerow(headers)
        for i in range(100):
            writer.writerow([f"unique_{i}_{j}" for j in range(25)])
    result = await local_analyze_data(str(csv_path))
    assert "HANDOFF" in result


async def test_local_analyze_data_file_not_found():
    result = await local_analyze_data("/nonexistent/data.csv")
    assert "not found" in result.lower()


async def test_local_analyze_data_non_csv(tmp_path):
    p = tmp_path / "data.xlsx"
    p.write_text("fake")
    result = await local_analyze_data(str(p))
    assert "CSV" in result


async def test_local_analyze_data_force_handoff_with_question(tmp_path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("name,age\nAlice,30\n")
    result = await local_analyze_data(
        str(csv_path), question="What is the average age?", force_handoff=True
    )
    assert "HANDOFF" in result
    assert "average age" in result
