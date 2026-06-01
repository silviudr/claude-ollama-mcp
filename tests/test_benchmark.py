import json

import httpx
import respx

from ollama_mcp.benchmark import ModelResult, format_results, run_benchmark
from ollama_mcp.config import OLLAMA_URL

API_URL = f"{OLLAMA_URL}/api/generate"
TAGS_URL = f"{OLLAMA_URL}/api/tags"


def _ollama_response(response_text="ok", prompt_tokens=10, output_tokens=5):
    return httpx.Response(200, json={
        "response": response_text,
        "prompt_eval_count": prompt_tokens,
        "eval_count": output_tokens,
        "eval_duration": 100_000_000,
    })


@respx.mock
async def test_run_benchmark_with_explicit_models():
    respx.post(API_URL).mock(side_effect=[
        _ollama_response("answer from alpha", 10, 5),
        _ollama_response("answer from beta", 12, 8),
    ])

    results = await run_benchmark("say hello", models=["alpha", "beta"])

    assert len(results) == 2
    assert results[0].model == "alpha"
    assert results[0].success is True
    assert results[0].response == "answer from alpha"
    assert results[1].model == "beta"
    assert results[1].success is True


@respx.mock
async def test_run_benchmark_auto_discovers_models():
    respx.get(TAGS_URL).mock(return_value=httpx.Response(200, json={
        "models": [
            {"name": "gemma4-32k"},
            {"name": "llama3.1"},
        ],
    }))
    respx.post(API_URL).mock(side_effect=[
        _ollama_response("gemma answer"),
        _ollama_response("llama answer"),
    ])

    results = await run_benchmark("test prompt")

    assert len(results) == 2
    models = {r.model for r in results}
    assert models == {"gemma4-32k", "llama3.1"}


@respx.mock
async def test_run_benchmark_handles_model_failure():
    respx.post(API_URL).mock(side_effect=[
        _ollama_response("works fine"),
        httpx.Response(404, text="model not found"),
    ])

    results = await run_benchmark("test", models=["good-model", "bad-model"])

    assert results[0].success is True
    assert results[1].success is False
    assert results[1].error is not None


@respx.mock
async def test_run_benchmark_collects_metrics():
    respx.post(API_URL).mock(return_value=_ollama_response("ok", 100, 50))

    results = await run_benchmark("test", models=["test-model"])

    r = results[0]
    assert r.prompt_tokens == 100
    assert r.output_tokens == 50
    assert r.wall_ms > 0


@respx.mock
async def test_run_benchmark_with_system_prompt():
    route = respx.post(API_URL).mock(return_value=_ollama_response())

    await run_benchmark("test", system="be concise", models=["m1"])

    payload = json.loads(route.calls[0].request.read())
    assert payload["system"] == "be concise"
    assert payload["model"] == "m1"


# --- format_results tests ---


def test_format_results_empty():
    assert format_results([]) == "No models available for benchmarking."


def test_format_results_with_data():
    results = [
        ModelResult("fast-model", True, "quick answer", 500, 10, 5, 400),
        ModelResult("slow-model", True, "verbose answer here", 2000, 10, 20, 1800),
    ]
    text = format_results(results)
    assert "fast-model" in text
    assert "slow-model" in text
    assert "Fastest:" in text
    assert "Most concise:" in text
    assert "fast-model" in text.split("Fastest:")[1]


def test_format_results_with_failure():
    results = [
        ModelResult("good", True, "ok", 500, 10, 5, 400),
        ModelResult("bad", False, "", 0, 0, 0, 0, error="model not found"),
    ]
    text = format_results(results)
    assert "FAIL" in text
    assert "model not found" in text


def test_format_results_sorted_by_latency():
    results = [
        ModelResult("slow", True, "a", 3000, 10, 5, 2500),
        ModelResult("fast", True, "b", 500, 10, 5, 400),
        ModelResult("mid", True, "c", 1500, 10, 5, 1200),
    ]
    text = format_results(results)
    lines = [l for l in text.split("\n") if l.strip() and not l.startswith("-")]
    model_order = [l.split()[0] for l in lines if l.split()[0] in ("fast", "mid", "slow")]
    assert model_order == ["fast", "mid", "slow"]


def test_format_results_single_model_no_comparison():
    results = [ModelResult("only-model", True, "ok", 500, 10, 5, 400)]
    text = format_results(results)
    assert "only-model" in text
    assert "Fastest:" not in text
