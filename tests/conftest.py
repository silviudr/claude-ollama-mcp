import pytest


@pytest.fixture(autouse=True)
def _isolate_telemetry_db(tmp_path, monkeypatch):
    """Point storage at a throwaway DB for every test.

    Without this, any test that reaches real telemetry (swarm subtasks,
    benchmark runs) writes fixture rows into the user's production
    ~/.cache/ollama_mcp.db, where they skew adaptive routing scores and
    usage stats. Individual test files may still override this with their
    own tmp DB — that stays isolated either way."""
    from ollama_mcp import storage

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "telemetry.db")
    if hasattr(storage._local, "conn"):
        del storage._local.conn
    yield
    if hasattr(storage._local, "conn"):
        del storage._local.conn


@pytest.fixture(autouse=True)
def _reset_swarm_semaphores():
    """swarm._semaphores is process-global (by design, so the concurrency
    cap holds across concurrent tool calls) — reset it between tests so a
    cap configured in one test can't leak into another."""
    from ollama_mcp import swarm
    swarm._semaphores.clear()
    yield
    swarm._semaphores.clear()
