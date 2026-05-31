import json

import pytest

from ollama_mcp.privacy import PrivacyConfig, PrivacyError, privacy_guard, scan


# --- scan() tests ---


def test_scan_detects_env_file():
    matches = scan("reading from .env now", PrivacyConfig())
    assert any(m.kind == "file_pattern" and ".env" in m.pattern for m in matches)


def test_scan_detects_key_file():
    matches = scan("loaded server.key for TLS", PrivacyConfig())
    assert any(m.kind == "file_pattern" for m in matches)


def test_scan_detects_pem_file():
    matches = scan("cert at /etc/ssl/private/cert.pem", PrivacyConfig())
    assert any(".pem" in m.pattern for m in matches)


def test_scan_detects_private_key_content():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow..."
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "content_pattern" for m in matches)


def test_scan_detects_aws_key():
    text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "content_pattern" for m in matches)


def test_scan_detects_github_token():
    text = "GITHUB_TOKEN=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "content_pattern" for m in matches)


def test_scan_detects_api_key_assignment():
    text = "api_key = sk-proj-abc123def456ghi789jkl012mno"
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "content_pattern" for m in matches)


def test_scan_detects_password_assignment():
    text = 'password = "hunter2"'
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "content_pattern" for m in matches)


def test_scan_clean_input_no_matches():
    text = "def add(a, b):\n    return a + b"
    matches = scan(text, PrivacyConfig())
    assert matches == []


def test_scan_file_path_in_diff():
    text = "+++ b/.env.production\n-API_URL=http://old\n+API_URL=http://new"
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "file_pattern" for m in matches)


def test_scan_redacts_matched_content():
    text = "password = supersecretvalue123"
    matches = scan(text, PrivacyConfig())
    content_matches = [m for m in matches if m.kind == "content_pattern"]
    assert content_matches
    for m in content_matches:
        assert "supersecretvalue123" not in m.matched_text


def test_scan_nested_path():
    text = "loading customer_data/users.csv"
    matches = scan(text, PrivacyConfig())
    assert any(m.kind == "file_pattern" for m in matches)


# --- PrivacyConfig tests ---


def test_config_loads_defaults_when_no_file():
    config = PrivacyConfig.load()
    assert len(config.file_patterns) > 0
    assert len(config.content_patterns) > 0
    assert config.action == "warn"


def test_config_loads_from_file(tmp_path, monkeypatch):
    import ollama_mcp.privacy as priv

    cfg_path = tmp_path / "privacy.json"
    cfg_path.write_text(json.dumps({
        "file_patterns": ["*.secret"],
        "content_patterns": ["CUSTOM_TOKEN_[A-Z]+"],
        "action": "reject",
    }))
    monkeypatch.setattr(priv, "PRIVACY_CONFIG_PATH", cfg_path)

    config = PrivacyConfig.load()
    assert config.file_patterns == ["*.secret"]
    assert config.content_patterns == ["CUSTOM_TOKEN_[A-Z]+"]
    assert config.action == "reject"


def test_config_handles_invalid_json(tmp_path, monkeypatch):
    import ollama_mcp.privacy as priv

    cfg_path = tmp_path / "privacy.json"
    cfg_path.write_text("not valid json {{{")
    monkeypatch.setattr(priv, "PRIVACY_CONFIG_PATH", cfg_path)

    config = PrivacyConfig.load()
    assert config.action == "warn"  # falls back to defaults


def test_custom_patterns_from_config():
    config = PrivacyConfig(
        file_patterns=["*.custom"],
        content_patterns=[r"CUSTOM_\d+"],
    )
    matches = scan("loading data.custom with CUSTOM_12345", config)
    assert len(matches) == 2


# --- privacy_guard decorator tests ---


async def test_guard_allows_clean_input():
    @privacy_guard
    async def my_tool(text: str):
        return "result", {"model": "test"}

    result = await my_tool("def add(a, b): return a + b")
    assert result == ("result", {"model": "test"})


async def test_guard_warns_but_continues(monkeypatch):
    import ollama_mcp.privacy as priv

    monkeypatch.setattr(priv, "PRIVACY_CONFIG_PATH", priv.PRIVACY_CONFIG_PATH.parent / "nonexistent")

    called = False

    @privacy_guard
    async def my_tool(text: str):
        nonlocal called
        called = True
        return "result", {"model": "test"}

    await my_tool("reading .env file")
    assert called


async def test_guard_rejects_when_configured(tmp_path, monkeypatch):
    import ollama_mcp.privacy as priv

    cfg_path = tmp_path / "privacy.json"
    cfg_path.write_text(json.dumps({"action": "reject"}))
    monkeypatch.setattr(priv, "PRIVACY_CONFIG_PATH", cfg_path)

    @privacy_guard
    async def my_tool(text: str):
        return "result", {}

    with pytest.raises(PrivacyError, match="Rejected"):
        await my_tool("loading .env file")


async def test_guard_rejects_on_content_match(tmp_path, monkeypatch):
    import ollama_mcp.privacy as priv

    cfg_path = tmp_path / "privacy.json"
    cfg_path.write_text(json.dumps({"action": "reject"}))
    monkeypatch.setattr(priv, "PRIVACY_CONFIG_PATH", cfg_path)

    @privacy_guard
    async def my_tool(text: str):
        return "result", {}

    with pytest.raises(PrivacyError, match="Rejected"):
        await my_tool("password = hunter2")


async def test_guard_checks_all_arguments(tmp_path, monkeypatch):
    """Guard should scan all string args, not just the first."""
    import ollama_mcp.privacy as priv

    cfg_path = tmp_path / "privacy.json"
    cfg_path.write_text(json.dumps({"action": "reject"}))
    monkeypatch.setattr(priv, "PRIVACY_CONFIG_PATH", cfg_path)

    @privacy_guard
    async def my_tool(code: str, context: str = ""):
        return "result", {}

    with pytest.raises(PrivacyError):
        await my_tool("clean code", context="password = secret123")
