import json

import pytest

from ollama_mcp.backends import OllamaBackend, OpenRouterBackend
from ollama_mcp.config import MODEL
from ollama_mcp.router import (
    AdaptiveConfig,
    _adaptive_score,
    _parse_adaptive,
    _pick_adaptive,
    _resolve_candidate,
    get_backends,
    get_routes_info,
    resolve,
    resolve_model,
)


# --- Legacy resolve_model tests (backwards compat) ---


def test_resolve_model_no_config(monkeypatch):
    import ollama_mcp.router as router
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", router.ROUTES_CONFIG_PATH.parent / "nonexistent")
    assert resolve_model("local_summarize") == MODEL


def test_resolve_model_with_route(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default": "gemma4-32k",
        "routes": {"local_review_diff": "deepseek-coder"},
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    assert resolve_model("local_review_diff") == "deepseek-coder"


def test_resolve_model_falls_back_to_config_default(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default": "llama3.1",
        "routes": {"local_review_diff": "deepseek-coder"},
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    assert resolve_model("local_summarize") == "llama3.1"


def test_resolve_model_falls_back_to_env_model(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({"routes": {"local_review_diff": "deepseek-coder"}}))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    assert resolve_model("local_summarize") == MODEL


def test_resolve_model_invalid_json(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text("not json {{{")
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    assert resolve_model("local_summarize") == MODEL


# --- get_routes_info tests ---


def test_get_routes_info_no_config(monkeypatch):
    import ollama_mcp.router as router
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", router.ROUTES_CONFIG_PATH.parent / "nonexistent")
    info = get_routes_info()
    assert "No per-tool routes" in info
    assert MODEL in info


def test_get_routes_info_with_config(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default": "gemma4-32k",
        "routes": {
            "local_review_diff": "deepseek-coder",
            "local_generate_tests": "qwen2.5-coder",
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    info = get_routes_info()
    assert "deepseek-coder" in info
    assert "qwen2.5-coder" in info
    assert "gemma4-32k" in info


# --- resolve() tests (new API) ---


def test_resolve_returns_backend_and_model(monkeypatch):
    import ollama_mcp.router as router
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", router.ROUTES_CONFIG_PATH.parent / "nonexistent")
    backend, model = resolve("local_summarize")
    assert isinstance(backend, OllamaBackend)
    assert model == MODEL


def test_resolve_legacy_config_returns_ollama(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default": "gemma4-32k",
        "routes": {"local_review_diff": "deepseek-coder"},
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    backend, model = resolve("local_review_diff")
    assert isinstance(backend, OllamaBackend)
    assert model == "deepseek-coder"


def test_resolve_new_config_with_openrouter(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
            "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "default_model": "google/gemma-3-27b-it:free"},
        },
        "routes": {
            "local_review_diff": {"backend": "openrouter", "model": "google/gemma-3-27b-it"},
            "local_summarize": "gemma4-32k",
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    backend, model = resolve("local_review_diff")
    assert isinstance(backend, OpenRouterBackend)
    assert model == "google/gemma-3-27b-it"

    backend2, model2 = resolve("local_summarize")
    assert isinstance(backend2, OllamaBackend)
    assert model2 == "gemma4-32k"


def test_resolve_unrouted_tool_uses_default(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "openrouter",
        "backends": {
            "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "default_model": "google/gemma-3-27b-it:free"},
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    backend, model = resolve("local_summarize")
    assert isinstance(backend, OpenRouterBackend)
    assert model == "google/gemma-3-27b-it:free"


def test_resolve_bare_model_string_in_new_config(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
        },
        "routes": {
            "local_summarize": "llama3.1",
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    backend, model = resolve("local_summarize")
    assert isinstance(backend, OllamaBackend)
    assert model == "llama3.1"


# --- get_backends tests ---


def test_get_backends_returns_all(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
            "openrouter": {"api_key_env": "MY_KEY", "default_model": "test-model"},
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    backends = get_backends()
    assert "ollama" in backends
    assert "openrouter" in backends
    assert isinstance(backends["ollama"], OllamaBackend)
    assert isinstance(backends["openrouter"], OpenRouterBackend)


def test_get_backends_no_config_returns_ollama(monkeypatch):
    import ollama_mcp.router as router
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", router.ROUTES_CONFIG_PATH.parent / "nonexistent")

    backends = get_backends()
    assert "ollama" in backends
    assert len(backends) == 1


# --- get_routes_info with new config ---


def test_get_routes_info_new_config(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
            "openrouter": {"api_key_env": "KEY", "default_model": "test-model"},
        },
        "routes": {
            "local_review_diff": {"backend": "openrouter", "model": "google/gemma-3-27b-it"},
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)

    info = get_routes_info()
    assert "ollama (default)" in info
    assert "openrouter" in info
    assert "local_review_diff" in info
    assert "Unrouted tools use" in info


# --- Adaptive routing ---


def test_parse_adaptive_disabled():
    config = _parse_adaptive({})
    assert config.enabled is False
    assert config.candidates == {}


def test_parse_adaptive_enabled():
    raw = {
        "adaptive": {
            "enabled": True,
            "min_samples": 5,
            "quality_weight": 0.8,
            "explore_rate": 0.2,
            "recency_window": 50,
            "candidates": {
                "local_summarize": ["gemma4-32k", "llama3.1"],
            },
        }
    }
    config = _parse_adaptive(raw)
    assert config.enabled is True
    assert config.min_samples == 5
    assert config.quality_weight == 0.8
    assert config.explore_rate == 0.2
    assert config.recency_window == 50
    assert config.candidates == {"local_summarize": ["gemma4-32k", "llama3.1"]}


def test_parse_adaptive_defaults():
    raw = {"adaptive": {"enabled": True}}
    config = _parse_adaptive(raw)
    assert config.min_samples == 10
    assert config.quality_weight == 0.7
    assert config.explore_rate == 0.1
    assert config.recency_window == 100


def test_resolve_candidate_bare_name():
    backends = {"ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k")}
    bname, model = _resolve_candidate("llama3.1", backends, "ollama")
    assert bname == "ollama"
    assert model == "llama3.1"


def test_resolve_candidate_prefixed():
    backends = {
        "ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k"),
        "openrouter": OpenRouterBackend(
            api_key_env="KEY", default_model="test",
        ),
    }
    bname, model = _resolve_candidate(
        "openrouter/google/gemma-3-27b-it:free", backends, "ollama",
    )
    assert bname == "openrouter"
    assert model == "google/gemma-3-27b-it:free"


def test_adaptive_score_quality_dominant():
    stats = {"avg_score": 0.9, "avg_latency_ms": 5000}
    score = _adaptive_score(stats, quality_weight=1.0)
    assert score == pytest.approx(0.9)


def test_adaptive_score_balanced():
    fast = {"avg_score": 0.7, "avg_latency_ms": 100}
    slow = {"avg_score": 0.9, "avg_latency_ms": 10000}
    assert _adaptive_score(fast, 0.5) > _adaptive_score(slow, 0.5) or \
           _adaptive_score(slow, 0.5) > _adaptive_score(fast, 0.5)


def test_adaptive_score_empty():
    assert _adaptive_score({}, 0.7) == 0.0


def test_pick_adaptive_no_candidates():
    adaptive = AdaptiveConfig(enabled=True, candidates={})
    backends = {"ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k")}
    result = _pick_adaptive("local_summarize", adaptive, backends, "ollama")
    assert result is None


def test_pick_adaptive_single_candidate():
    adaptive = AdaptiveConfig(
        enabled=True,
        candidates={"local_summarize": ["gemma4-32k"]},
    )
    backends = {"ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k")}
    result = _pick_adaptive("local_summarize", adaptive, backends, "ollama")
    assert result is None


def test_pick_adaptive_not_enough_samples(monkeypatch):
    adaptive = AdaptiveConfig(
        enabled=True,
        min_samples=10,
        candidates={"local_summarize": ["gemma4-32k", "llama3.1"]},
    )
    backends = {"ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k")}
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [
            {"model": "gemma4-32k", "avg_score": 0.8, "avg_latency_ms": 100, "sample_count": 3},
        ],
    )
    result = _pick_adaptive("local_summarize", adaptive, backends, "ollama")
    assert result is None


def test_pick_adaptive_selects_best(monkeypatch):
    adaptive = AdaptiveConfig(
        enabled=True,
        min_samples=5,
        explore_rate=0.0,
        quality_weight=0.7,
        candidates={"local_summarize": ["gemma4-32k", "llama3.1"]},
    )
    backends = {"ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k")}
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [
            {"model": "gemma4-32k", "avg_score": 0.9, "avg_latency_ms": 200, "sample_count": 20},
            {"model": "llama3.1", "avg_score": 0.7, "avg_latency_ms": 100, "sample_count": 15},
        ],
    )
    bname, model = _pick_adaptive("local_summarize", adaptive, backends, "ollama")
    assert model == "gemma4-32k"


def test_pick_adaptive_explore_mode(monkeypatch):
    adaptive = AdaptiveConfig(
        enabled=True,
        min_samples=5,
        explore_rate=1.0,
        candidates={"local_summarize": ["gemma4-32k", "llama3.1"]},
    )
    backends = {"ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k")}
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [
            {"model": "gemma4-32k", "avg_score": 0.9, "avg_latency_ms": 200, "sample_count": 20},
            {"model": "llama3.1", "avg_score": 0.7, "avg_latency_ms": 100, "sample_count": 15},
        ],
    )
    monkeypatch.setattr("ollama_mcp.router.random.random", lambda: 0.0)
    monkeypatch.setattr("ollama_mcp.router.random.choice", lambda lst: lst[0])
    result = _pick_adaptive("local_summarize", adaptive, backends, "ollama")
    assert result is not None


def test_resolve_uses_adaptive(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
        },
        "routes": {
            "local_summarize": "gemma4-32k",
        },
        "adaptive": {
            "enabled": True,
            "min_samples": 5,
            "explore_rate": 0.0,
            "quality_weight": 0.7,
            "candidates": {
                "local_summarize": ["gemma4-32k", "llama3.1"],
            },
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [
            {"model": "llama3.1", "avg_score": 0.95, "avg_latency_ms": 150, "sample_count": 20},
            {"model": "gemma4-32k", "avg_score": 0.8, "avg_latency_ms": 200, "sample_count": 20},
        ],
    )
    backend, model = resolve("local_summarize")
    assert model == "llama3.1"
    assert isinstance(backend, OllamaBackend)


def test_resolve_falls_back_to_static_when_adaptive_cold(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
        },
        "routes": {
            "local_summarize": "gemma4-32k",
        },
        "adaptive": {
            "enabled": True,
            "min_samples": 10,
            "candidates": {
                "local_summarize": ["gemma4-32k", "llama3.1"],
            },
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [],
    )
    backend, model = resolve("local_summarize")
    assert model == "gemma4-32k"


def test_resolve_non_adaptive_tool_unaffected(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
        },
        "routes": {
            "local_commit_message": "deepseek-coder",
        },
        "adaptive": {
            "enabled": True,
            "candidates": {
                "local_summarize": ["gemma4-32k", "llama3.1"],
            },
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)
    backend, model = resolve("local_commit_message")
    assert model == "deepseek-coder"


def test_get_routes_info_shows_adaptive(tmp_path, monkeypatch):
    import ollama_mcp.router as router
    cfg = tmp_path / "routes.json"
    cfg.write_text(json.dumps({
        "default_backend": "ollama",
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "gemma4-32k"},
        },
        "adaptive": {
            "enabled": True,
            "min_samples": 10,
            "candidates": {
                "local_summarize": ["gemma4-32k", "llama3.1"],
            },
        },
    }))
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", cfg)
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [
            {"model": "gemma4-32k", "avg_score": 0.85, "avg_latency_ms": 200, "sample_count": 25},
            {"model": "llama3.1", "avg_score": 0.70, "avg_latency_ms": 100, "sample_count": 5},
        ],
    )
    info = get_routes_info()
    assert "Adaptive routing: enabled" in info
    assert "local_summarize" in info
    assert "gemma4-32k" in info
    assert "warming up" in info


def test_pick_adaptive_with_prefixed_candidate(monkeypatch):
    adaptive = AdaptiveConfig(
        enabled=True,
        min_samples=5,
        explore_rate=0.0,
        quality_weight=0.7,
        candidates={
            "local_summarize": [
                "gemma4-32k",
                "openrouter/google/gemma-3-27b-it:free",
            ],
        },
    )
    backends = {
        "ollama": OllamaBackend(url="http://localhost:11434", default_model="gemma4-32k"),
        "openrouter": OpenRouterBackend(
            api_key_env="KEY", default_model="test",
        ),
    }
    monkeypatch.setattr(
        "ollama_mcp.storage.get_model_scores",
        lambda *a, **kw: [
            {"model": "google/gemma-3-27b-it:free", "avg_score": 0.95,
             "avg_latency_ms": 300, "sample_count": 15},
            {"model": "gemma4-32k", "avg_score": 0.8,
             "avg_latency_ms": 200, "sample_count": 20},
        ],
    )
    bname, model = _pick_adaptive(
        "local_summarize", adaptive, backends, "ollama",
    )
    assert bname == "openrouter"
    assert model == "google/gemma-3-27b-it:free"
