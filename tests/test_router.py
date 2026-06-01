import json

from ollama_mcp.config import MODEL
from ollama_mcp.router import get_routes_info, resolve_model


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


def test_get_routes_info_no_config(monkeypatch):
    import ollama_mcp.router as router
    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", router.ROUTES_CONFIG_PATH.parent / "nonexistent")
    info = get_routes_info()
    assert "No routing config" in info
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
