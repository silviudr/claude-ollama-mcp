import json

from ollama_mcp.backends import OllamaBackend, OpenRouterBackend
from ollama_mcp.config import MODEL
from ollama_mcp.router import get_backends, get_routes_info, resolve, resolve_model


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
