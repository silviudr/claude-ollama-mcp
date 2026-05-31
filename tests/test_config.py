import importlib
import os


def test_defaults(monkeypatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_MCP_LOG", raising=False)

    import ollama_mcp.config as cfg
    cfg = importlib.reload(cfg)

    assert cfg.OLLAMA_URL == "http://localhost:11434"
    assert cfg.MODEL == "gemma4-32k"
    assert cfg.LOG_PATH.name == "ollama_mcp.jsonl"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://gpu-box:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "deepseek-coder")
    monkeypatch.setenv("OLLAMA_MCP_LOG", "/tmp/test.jsonl")

    import ollama_mcp.config as cfg
    cfg = importlib.reload(cfg)

    assert cfg.OLLAMA_URL == "http://gpu-box:11434"
    assert cfg.MODEL == "deepseek-coder"
    assert str(cfg.LOG_PATH) == "/tmp/test.jsonl"
