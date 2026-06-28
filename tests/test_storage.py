import threading
import time

import pytest

from ollama_mcp import storage


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    # Reset thread-local connection so each test gets a fresh DB
    if hasattr(storage._local, "conn"):
        del storage._local.conn


def _make_event(tool="local_summarize", ok=True, **overrides):
    event = {
        "ts": time.time(),
        "tool": tool,
        "ok": ok,
        "model": "gemma4-32k",
        "input_chars": 100,
        "output_chars": 50,
        "prompt_tokens": 200,
        "output_tokens": 80,
        "total_ms": 1500,
        "wall_ms": 1400,
        "eval_ms": 1200,
    }
    event.update(overrides)
    return event


def test_log_and_retrieve():
    storage.log_call(_make_event())
    stats = storage.get_stats()
    assert stats["total_calls"] == 1
    assert stats["successful"] == 1
    assert stats["failed"] == 0


def test_multiple_calls():
    storage.log_call(_make_event(tool="local_summarize"))
    storage.log_call(_make_event(tool="local_summarize"))
    storage.log_call(_make_event(tool="local_commit_message"))
    stats = storage.get_stats()
    assert stats["total_calls"] == 3
    assert len(stats["per_tool"]) == 2


def test_failed_call_tracked():
    storage.log_call(_make_event(ok=True))
    storage.log_call(_make_event(ok=False, error="timeout"))
    stats = storage.get_stats()
    assert stats["successful"] == 1
    assert stats["failed"] == 1


def test_token_aggregation():
    storage.log_call(_make_event(prompt_tokens=100, output_tokens=50))
    storage.log_call(_make_event(prompt_tokens=200, output_tokens=150))
    stats = storage.get_stats()
    assert stats["total_prompt_tokens"] == 300
    assert stats["total_output_tokens"] == 200


def test_cost_estimation():
    storage.log_call(_make_event(prompt_tokens=1_000_000, output_tokens=100_000))
    stats = storage.get_stats()
    costs = stats["estimated_cost_avoided"]
    # Opus: 1M * $15/M + 100K * $75/M = $15 + $7.50 = $22.50
    assert costs["opus"] == pytest.approx(22.50, abs=0.01)
    # Sonnet: 1M * $3/M + 100K * $15/M = $3 + $1.50 = $4.50
    assert costs["sonnet"] == pytest.approx(4.50, abs=0.01)


def test_empty_db_stats():
    stats = storage.get_stats()
    assert stats["total_calls"] == 0
    assert stats["total_prompt_tokens"] == 0
    assert stats["per_tool"] == []
    assert stats["estimated_cost_avoided"]["opus"] == 0
    assert stats["estimated_cost_avoided"]["sonnet"] == 0


def test_per_tool_breakdown():
    storage.log_call(_make_event(tool="local_summarize", prompt_tokens=100))
    storage.log_call(_make_event(tool="local_summarize", prompt_tokens=200))
    storage.log_call(_make_event(tool="local_review_diff", prompt_tokens=500))
    stats = storage.get_stats()
    by_tool = {t["tool"]: t for t in stats["per_tool"]}
    assert by_tool["local_summarize"]["calls"] == 2
    assert by_tool["local_summarize"]["prompt_tokens"] == 300
    assert by_tool["local_review_diff"]["calls"] == 1
    assert by_tool["local_review_diff"]["prompt_tokens"] == 500


def test_null_tokens_treated_as_zero():
    storage.log_call(_make_event(prompt_tokens=None, output_tokens=None))
    stats = storage.get_stats()
    assert stats["total_prompt_tokens"] == 0
    assert stats["total_output_tokens"] == 0


# --- Per-backend breakdown ---


def test_per_backend_single_backend():
    storage.log_call(_make_event(backend="ollama"))
    storage.log_call(_make_event(backend="ollama"))
    stats = storage.get_stats()
    assert "ollama" in stats["per_backend"]
    assert stats["per_backend"]["ollama"]["calls"] == 2
    assert "estimated_cost_avoided" in stats["per_backend"]["ollama"]


def test_per_backend_multiple_backends():
    storage.log_call(_make_event(backend="ollama", prompt_tokens=100, output_tokens=50))
    storage.log_call(_make_event(
        backend="openrouter", prompt_tokens=200, output_tokens=80,
        cost=0.001, model="google/gemma-3-27b-it",
    ))
    stats = storage.get_stats()
    assert "ollama" in stats["per_backend"]
    assert "openrouter" in stats["per_backend"]
    assert stats["per_backend"]["ollama"]["calls"] == 1
    assert stats["per_backend"]["ollama"]["prompt_tokens"] == 100
    assert stats["per_backend"]["openrouter"]["calls"] == 1
    assert stats["per_backend"]["openrouter"]["prompt_tokens"] == 200
    assert stats["per_backend"]["openrouter"]["total_cost"] == pytest.approx(0.001)


def test_per_backend_cost_only_on_ollama():
    storage.log_call(_make_event(backend="ollama"))
    storage.log_call(_make_event(backend="openrouter", cost=0.005))
    stats = storage.get_stats()
    assert "estimated_cost_avoided" in stats["per_backend"]["ollama"]
    assert "estimated_cost_avoided" not in stats["per_backend"]["openrouter"]


def test_per_backend_null_backend_treated_as_ollama():
    storage.log_call(_make_event())  # no backend key
    stats = storage.get_stats()
    assert "ollama" in stats["per_backend"]
    assert stats["per_backend"]["ollama"]["calls"] == 1


def test_openrouter_cost_aggregation():
    storage.log_call(_make_event(backend="openrouter", cost=0.001))
    storage.log_call(_make_event(backend="openrouter", cost=0.002))
    storage.log_call(_make_event(backend="openrouter", cost=None))
    stats = storage.get_stats()
    assert stats["per_backend"]["openrouter"]["total_cost"] == pytest.approx(0.003)
    assert stats["per_backend"]["openrouter"]["calls"] == 3


def test_empty_db_has_no_per_backend():
    stats = storage.get_stats()
    assert stats["per_backend"] == {}


# --- get_model_scores ---


def _seed_graded_calls(tool, model, scores, base_ts=None):
    """Insert call+grade pairs and return call_ids."""
    if base_ts is None:
        base_ts = time.time()
    for i, score in enumerate(scores):
        call_id = storage.log_call(_make_event(
            tool=tool, model=model, ts=base_ts + i,
        ))
        storage.log_grade({
            "ts": base_ts + i,
            "call_id": call_id,
            "tool": tool,
            "grade_type": "semantic",
            "checker": "overall",
            "score": score,
            "passed": score >= 0.5,
            "details": None,
            "grader_model": "judge",
            "grader_backend": "openrouter",
            "grader_ms": 100,
        })


def test_get_model_scores_basic():
    _seed_graded_calls("local_summarize", "gemma4-32k", [0.8, 0.9, 0.7])
    result = storage.get_model_scores("local_summarize", ["gemma4-32k"])
    assert len(result) == 1
    assert result[0]["model"] == "gemma4-32k"
    assert result[0]["avg_score"] == pytest.approx(0.8, abs=0.01)
    assert result[0]["sample_count"] == 3


def test_get_model_scores_multiple_models():
    _seed_graded_calls("local_summarize", "gemma4-32k", [0.8, 0.9])
    _seed_graded_calls("local_summarize", "llama3.1", [0.6, 0.7])
    result = storage.get_model_scores("local_summarize", ["gemma4-32k", "llama3.1"])
    assert len(result) == 2
    assert result[0]["model"] == "gemma4-32k"
    assert result[1]["model"] == "llama3.1"


def test_get_model_scores_filters_by_tool():
    _seed_graded_calls("local_summarize", "gemma4-32k", [0.9])
    _seed_graded_calls("local_review_diff", "gemma4-32k", [0.5])
    result = storage.get_model_scores("local_summarize", ["gemma4-32k"])
    assert len(result) == 1
    assert result[0]["avg_score"] == pytest.approx(0.9, abs=0.01)


def test_get_model_scores_recency_window():
    ts = time.time()
    _seed_graded_calls("local_summarize", "gemma4-32k",
                       [0.3, 0.3, 0.3, 0.9, 0.9], base_ts=ts)
    result_all = storage.get_model_scores(
        "local_summarize", ["gemma4-32k"], recency_window=100,
    )
    result_recent = storage.get_model_scores(
        "local_summarize", ["gemma4-32k"], recency_window=2,
    )
    assert result_all[0]["sample_count"] == 5
    assert result_recent[0]["sample_count"] == 2
    assert result_recent[0]["avg_score"] > result_all[0]["avg_score"]


def test_get_model_scores_empty_candidates():
    result = storage.get_model_scores("local_summarize", [])
    assert result == []


def test_get_model_scores_no_data():
    result = storage.get_model_scores("local_summarize", ["nonexistent"])
    assert result == []
