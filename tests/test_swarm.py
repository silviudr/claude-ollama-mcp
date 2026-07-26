import asyncio

from ollama_mcp.backends.base import Backend
from ollama_mcp.swarm import (
    SwarmTask,
    combine_meta,
    format_consensus,
    run_swarm,
)


class _FakeBackend(Backend):
    def __init__(self, name="ollama", delay=0.0, default_model="fake-model", fail_models=None):
        self.name = name
        self.default_model = default_model
        self.delay = delay
        self.fail_models = fail_models or set()
        self.active = 0
        self.max_active = 0

    async def generate(self, prompt, system=None, model=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if model in self.fail_models:
                raise RuntimeError(f"boom-{model}")
            return f"response-{model}", {
                "wall_ms": 1, "prompt_tokens": 1, "output_tokens": 1, "eval_ms": 1,
            }
        finally:
            self.active -= 1

    async def list_models(self):
        return []


async def test_concurrency_cap_respected():
    backend = _FakeBackend(delay=0.05)
    tasks = [
        SwarmTask(key=str(i), backend=backend, model=f"m{i}", prompt="p")
        for i in range(6)
    ]
    await run_swarm(tasks, concurrency={"ollama": 2})
    assert backend.max_active <= 2


async def test_backends_have_independent_caps():
    backend_a = _FakeBackend(name="ollama", delay=0.05)
    backend_b = _FakeBackend(name="openrouter", delay=0.05)
    tasks = [
        SwarmTask(key=f"a{i}", backend=backend_a, model=f"m{i}", prompt="p")
        for i in range(4)
    ] + [
        SwarmTask(key=f"b{i}", backend=backend_b, model=f"m{i}", prompt="p")
        for i in range(4)
    ]
    await run_swarm(tasks, concurrency={"ollama": 1, "openrouter": 4})
    assert backend_a.max_active <= 1
    assert backend_b.max_active <= 4
    assert backend_b.max_active > 1  # actually used its higher cap


async def test_one_failure_does_not_affect_others():
    backend = _FakeBackend(fail_models={"bad"})
    tasks = [
        SwarmTask(key="good1", backend=backend, model="good1", prompt="p"),
        SwarmTask(key="bad", backend=backend, model="bad", prompt="p"),
        SwarmTask(key="good2", backend=backend, model="good2", prompt="p"),
    ]
    results = await run_swarm(tasks)
    assert len(results) == 3
    assert results[0].success and results[2].success
    assert not results[1].success
    assert "boom-bad" in results[1].error


async def test_result_order_matches_task_order():
    backend = _FakeBackend()

    async def variable_delay_generate(prompt, system=None, model=None):
        delay = 0.05 if model == "slow" else 0.0
        await asyncio.sleep(delay)
        return f"response-{model}", {"wall_ms": 1}

    backend.generate = variable_delay_generate
    tasks = [
        SwarmTask(key="slow", backend=backend, model="slow", prompt="p"),
        SwarmTask(key="fast", backend=backend, model="fast", prompt="p"),
    ]
    results = await run_swarm(tasks)
    assert [r.key for r in results] == ["slow", "fast"]


async def test_per_task_telemetry_recorded(monkeypatch):
    recorded = []
    graded = []
    monkeypatch.setattr("ollama_mcp.telemetry.record", lambda event: recorded.append(event) or 1)
    monkeypatch.setattr(
        "ollama_mcp.grading.schedule_grading",
        lambda tool, inp, out, call_id: graded.append((tool, inp, out, call_id)),
    )

    backend = _FakeBackend(fail_models={"bad"})
    tasks = [
        SwarmTask(key="security", backend=backend, model="good", prompt="diff"),
        SwarmTask(key="bad", backend=backend, model="bad", prompt="diff"),
    ]
    await run_swarm(tasks, tool_name="local_review_diff")

    tools = {e["tool"] for e in recorded}
    assert tools == {"local_review_diff:security", "local_review_diff:bad"}
    assert any(e["tool"] == "local_review_diff:security" and e["ok"] for e in recorded)
    assert any(e["tool"] == "local_review_diff:bad" and not e["ok"] for e in recorded)
    assert {t for t, *_ in graded} == {"local_review_diff:security", "local_review_diff:bad"}


async def test_subtask_telemetry_records_total_ms(monkeypatch):
    """Every subtask row needs a total_ms — get_stats averages that column,
    so a NULL makes the whole tool read 'avg 0ms' in usage stats."""
    recorded = []
    monkeypatch.setattr("ollama_mcp.telemetry.record", lambda event: recorded.append(event) or 1)
    monkeypatch.setattr(
        "ollama_mcp.grading.schedule_grading", lambda tool, inp, out, call_id: None,
    )

    backend = _FakeBackend(delay=0.02, fail_models={"bad"})
    tasks = [
        SwarmTask(key="ok", backend=backend, model="good", prompt="p"),
        SwarmTask(key="bad", backend=backend, model="bad", prompt="p"),
    ]
    await run_swarm(tasks, tool_name="local_review_diff")

    assert len(recorded) == 2
    for event in recorded:
        assert event["total_ms"] is not None
        assert event["total_ms"] >= 0
    # The succeeding task slept 20ms; its own duration must be reflected.
    ok_event = next(e for e in recorded if e["tool"] == "local_review_diff:ok")
    assert ok_event["total_ms"] >= 15


async def test_subtask_total_ms_excludes_queue_wait(monkeypatch):
    """total_ms measures the task's own run, not time spent waiting for a
    concurrency slot — same semantics as the failure path's wall_ms."""
    recorded = []
    monkeypatch.setattr("ollama_mcp.telemetry.record", lambda event: recorded.append(event) or 1)
    monkeypatch.setattr(
        "ollama_mcp.grading.schedule_grading", lambda tool, inp, out, call_id: None,
    )

    backend = _FakeBackend()

    async def per_model_generate(prompt, system=None, model=None):
        await asyncio.sleep(0.15 if model.startswith("slow") else 0.0)
        return f"response-{model}", {"wall_ms": 1}

    backend.generate = per_model_generate

    tasks = [
        SwarmTask(key="slow1", backend=backend, model="slow1", prompt="p"),
        SwarmTask(key="slow2", backend=backend, model="slow2", prompt="p"),
        SwarmTask(key="quick", backend=backend, model="quick", prompt="p"),
    ]
    await run_swarm(tasks, tool_name="t", concurrency={"ollama": 1})

    quick = next(e for e in recorded if e["tool"] == "t:quick")
    assert quick["total_ms"] < 100


def test_combine_meta_sums_and_maxes():
    from ollama_mcp.swarm import SwarmResult

    results = [
        SwarmResult(
            key="a", backend="ollama", model="m1", success=True, text="x",
            meta={"wall_ms": 100, "eval_ms": 90, "prompt_tokens": 10, "output_tokens": 5},
        ),
        SwarmResult(
            key="b", backend="ollama", model="m2", success=True, text="y",
            meta={"wall_ms": 200, "eval_ms": 150, "prompt_tokens": 20, "output_tokens": 8},
        ),
    ]
    meta = combine_meta(results)
    assert meta["wall_ms"] == 200
    assert meta["eval_ms"] == 150
    assert meta["prompt_tokens"] == 30
    assert meta["output_tokens"] == 13
    assert meta["backend"] == "swarm"


def test_combine_meta_handles_all_failed():
    from ollama_mcp.swarm import SwarmResult

    results = [
        SwarmResult(key="a", backend="ollama", model="m1", success=False, error="boom", meta={"wall_ms": 5}),
    ]
    meta = combine_meta(results)
    assert meta["wall_ms"] == 5
    assert meta["prompt_tokens"] == 0


def test_combine_meta_all_free_cost_is_zero_not_none():
    from ollama_mcp.swarm import SwarmResult

    results = [
        SwarmResult(key="a", backend="ollama", model="m1", success=True, text="x", meta={"cost": 0.0}),
        SwarmResult(key="b", backend="ollama", model="m2", success=True, text="y", meta={"cost": 0.0}),
    ]
    meta = combine_meta(results)
    assert meta["cost"] == 0.0


def test_combine_meta_no_cost_data_is_none():
    from ollama_mcp.swarm import SwarmResult

    results = [
        SwarmResult(key="a", backend="ollama", model="m1", success=True, text="x", meta={}),
        SwarmResult(key="b", backend="ollama", model="m2", success=True, text="y", meta={}),
    ]
    meta = combine_meta(results)
    assert meta["cost"] is None


async def test_concurrency_cap_is_global_across_run_swarm_calls():
    backend = _FakeBackend(delay=0.05)

    async def run_batch():
        tasks = [
            SwarmTask(key=str(i), backend=backend, model=f"m{i}", prompt="p")
            for i in range(2)
        ]
        await run_swarm(tasks, concurrency={"ollama": 1})

    # Two separate run_swarm calls, same backend type, launched concurrently.
    # A per-instance cap would let each call's tasks run independently
    # (up to 4 concurrent); the cap must hold across both calls (max 1).
    await asyncio.gather(run_batch(), run_batch())
    assert backend.max_active <= 1


async def test_concurrency_cap_change_takes_effect_without_restart():
    """routes.json is re-read per call, so a changed swarm.concurrency must
    apply on the next call. Caching the semaphore on first sight of a backend
    name froze the cap for the life of the process."""
    backend = _FakeBackend(delay=0.05)

    def tasks_for(n):
        return [
            SwarmTask(key=str(i), backend=backend, model=f"m{i}", prompt="p")
            for i in range(n)
        ]

    await run_swarm(tasks_for(4), concurrency={"ollama": 1})
    assert backend.max_active == 1

    # Same process, same backend name, higher cap — must be honored.
    backend.max_active = 0
    await run_swarm(tasks_for(4), concurrency={"ollama": 3})
    assert backend.max_active > 1, "cap change ignored — semaphore was cached"
    assert backend.max_active <= 3

    # And back down again.
    backend.max_active = 0
    await run_swarm(tasks_for(4), concurrency={"ollama": 1})
    assert backend.max_active == 1


async def test_unchanged_cap_reuses_same_semaphore():
    """The cap only rebuilds when it actually changes — otherwise concurrent
    calls would each get a fresh semaphore and collectively exceed the cap."""
    from ollama_mcp import swarm

    backend = _FakeBackend()
    tasks = [SwarmTask(key="a", backend=backend, model="m", prompt="p")]

    await run_swarm(tasks, concurrency={"ollama": 2})
    first = swarm._semaphores["ollama"][1]
    await run_swarm(tasks, concurrency={"ollama": 2})
    assert swarm._semaphores["ollama"][1] is first


async def test_failed_task_wall_ms_excludes_semaphore_queue_wait():
    backend = _FakeBackend()

    async def per_model_generate(prompt, system=None, model=None):
        if model == "bad":
            raise RuntimeError("boom-bad")
        await asyncio.sleep(0.15)
        return f"response-{model}", {"wall_ms": 1}

    backend.generate = per_model_generate

    tasks = [
        SwarmTask(key="slow1", backend=backend, model="slow1", prompt="p"),
        SwarmTask(key="slow2", backend=backend, model="slow2", prompt="p"),
        SwarmTask(key="bad", backend=backend, model="bad", prompt="p"),
    ]
    # cap=1 forces "bad" to queue behind both slow tasks before it can run
    results = await run_swarm(tasks, concurrency={"ollama": 1})

    failed = next(r for r in results if r.key == "bad")
    assert not failed.success
    # Should reflect only the failed call's own (near-instant) duration,
    # not the ~300ms spent queued behind the two slow tasks.
    assert failed.meta["wall_ms"] < 100


# --- format_consensus ---


def _result(key, model, text, success=True, error=None, wall_ms=100):
    from ollama_mcp.swarm import SwarmResult
    return SwarmResult(
        key=key, backend="ollama", model=model, success=success, text=text,
        error=error, meta={"wall_ms": wall_ms, "prompt_tokens": 1, "output_tokens": 1},
    )


def test_format_consensus_all_agree():
    results = [
        _result("m1", "m1", "the answer is 42"),
        _result("m2", "m2", "the answer is 42"),
        _result("m3", "m3", "the answer is 42"),
    ]
    out = format_consensus(results)
    assert "3/3 models agree" in out
    assert "the answer is 42" in out


def test_format_consensus_majority_split():
    results = [
        _result("m1", "m1", "def foo(): return 1"),
        _result("m2", "m2", "def foo(): return 1"),
        _result("m3", "m3", "completely different unrelated text about cats"),
    ]
    out = format_consensus(results)
    assert "Agreement: 2/3" in out
    assert "Divergent" in out


def test_format_consensus_no_majority():
    results = [
        _result("m1", "m1", "alpha alpha alpha alpha alpha"),
        _result("m2", "m2", "beta beta beta beta beta"),
    ]
    out = format_consensus(results)
    assert "No majority" in out


def test_format_consensus_single_success_no_comparison():
    results = [_result("m1", "m1", "only answer")]
    out = format_consensus(results)
    assert "only answer" in out
    assert "Agreement" not in out
    assert "No majority" not in out


def test_format_consensus_failed_candidate_excluded():
    results = [
        _result("m1", "m1", "shared answer text here"),
        _result("m2", "m2", "shared answer text here"),
        _result("m3", "m3", "", success=False, error="timeout"),
    ]
    out = format_consensus(results)
    assert "2/2 models agree" in out
    assert "m3" in out and "timeout" in out


def test_format_consensus_verbose_and_terse_same_answer_agree():
    """Regression: these are verbatim outputs from a real qwen3:8b /
    llama3.2 consensus run. Both are correct and say the same thing, but
    the verbose one is ~1.7x longer — under the old SequenceMatcher metric
    that length gap alone scored them 0.53, below the 0.55 cutoff, and the
    report claimed 'all 2 models disagree'."""
    results = [
        _result("qwen3", "qwen3:8b",
                "In Python, `dict.get(key)` returns `None` if the key is missing "
                "and no default value is provided."),
        _result("llama", "llama3.2",
                "If the key is not found in a dictionary using `dict.get(key)`, "
                "it returns the default value specified as the second argument "
                "(or `None` if no default value is provided)."),
    ]
    out = format_consensus(results)
    assert "2/2 models agree" in out
    assert "No majority" not in out


def test_similarity_ignores_verbosity_but_not_content():
    from ollama_mcp.swarm import _similarity

    terse = "the function returns None when the key is absent"
    verbose = (
        "When the key is absent from the mapping, the function returns None, "
        "which is the documented behavior for this lookup helper."
    )
    unrelated = "cats are excellent companions and enjoy sleeping in sunbeams"

    assert _similarity(terse, verbose) >= 0.6
    assert _similarity(terse, unrelated) < 0.6


def test_similarity_short_texts_need_real_overlap():
    """Containment alone would score 'yes' as 1.0 against any answer that
    happens to contain the word — require a floor of shared words."""
    from ollama_mcp.swarm import _similarity

    assert _similarity("yes", "yes, but only when the cache is cold") < 0.6
    assert _similarity("", "") == 1.0
    assert _similarity("", "something") == 0.0


def test_format_consensus_all_failed():
    results = [
        _result("m1", "m1", "", success=False, error="down"),
        _result("m2", "m2", "", success=False, error="down"),
    ]
    out = format_consensus(results)
    assert "All candidates failed" in out
