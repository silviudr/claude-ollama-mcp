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


@pytest.fixture(autouse=True)
def _isolate_routes_config(tmp_path, monkeypatch):
    """Point the router at a routes.json that does not exist.

    Without this the suite reads the developer's real
    ~/.config/ollama_mcp/routes.json, so results depend on whose machine it
    runs on: tests pass in CI (no config) and fail locally the moment someone
    follows the setup instructions and configures a non-default backend URL.
    respx mocks localhost:11434, the router resolves the user's actual host,
    the mock never matches, and six benchmark tests fail for reasons that have
    nothing to do with the code under test.

    router imports the path into its own namespace, so patching
    config.ROUTES_CONFIG_PATH would have no effect — patch it on router.
    Individual tests still override this with their own fixture config."""
    from ollama_mcp import router

    monkeypatch.setattr(router, "ROUTES_CONFIG_PATH", tmp_path / "no-routes.json")
    yield


@pytest.fixture(autouse=True)
def _disable_background_grading(monkeypatch):
    """Stop fire-and-forget grading from racing assertions.

    schedule_grading spawns a detached task (loop.create_task) that issues its
    own backend call. Any test asserting on exact request counts or models is
    therefore racy: when a fixture config omits a "grading" section the default
    sample rate applies, and roughly one run in six an extra grader request
    landed on the mocked route and failed the assertion — with no connection to
    the code under test.

    Tests that exercise grading override this: test_swarm and test_benchmark
    patch the same name with their own spy, and test_grading_engine imports
    schedule_grading directly so it holds a reference this never touches."""
    monkeypatch.setattr(
        "ollama_mcp.grading.schedule_grading",
        lambda tool, inp, out, call_id: None,
    )
    yield
