import json
import time

import pytest

from ollama_mcp import storage
from ollama_mcp.telemetry import observed, record


@pytest.fixture(autouse=True)
def _use_tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "test.db")
    if hasattr(storage._local, "conn"):
        del storage._local.conn


def test_record_appends_timestamp(tmp_path, monkeypatch):
    log_file = tmp_path / "test.jsonl"
    import logging

    logger = logging.getLogger("ollama_mcp")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    try:
        record({"tool": "test", "ok": True})
        line = json.loads(log_file.read_text().strip())
        assert line["tool"] == "test"
        assert line["ok"] is True
        assert "ts" in line
        assert abs(line["ts"] - time.time()) < 2
    finally:
        logger.removeHandler(handler)


async def test_observed_success():
    @observed("test_tool")
    async def fake_tool(prompt: str):
        return "result", {"model": "test", "wall_ms": 1}

    result = await fake_tool("hello")
    assert result == "result"


async def test_observed_failure():
    @observed("test_tool")
    async def failing_tool(prompt: str):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        await failing_tool("hello")
