"""Concurrent fan-out execution primitive for multi-model/multi-dimension calls.

Backend/domain-agnostic: knows nothing about review dimensions or consensus
semantics. Callers build a list of SwarmTask, get back a list of SwarmResult
in the same order, and interpret them however they need to.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field

from pydantic import BaseModel

from .backends import Backend

DEFAULT_CONCURRENCY = {"ollama": 2, "openrouter": 8}
_FALLBACK_CAP = 4

# Keyed by backend.name (a fixed per-type string, e.g. "ollama"), not by
# instance — Backend objects are rebuilt fresh on every router resolve, so a
# per-instance semaphore would let concurrent tool calls each get their own
# cap and collectively exceed it. Module-level so the cap holds process-wide.
_semaphores: dict[str, asyncio.Semaphore] = {}


@dataclass
class SwarmTask:
    key: str
    backend: Backend
    model: str
    prompt: str
    system: str | None = None
    schema: type[BaseModel] | None = None


@dataclass
class SwarmResult:
    key: str
    backend: str
    model: str
    success: bool
    text: str = ""
    parsed: BaseModel | None = None
    meta: dict = field(default_factory=dict)
    error: str | None = None


async def _run_one(task: SwarmTask, sem: asyncio.Semaphore, tool_name: str | None) -> SwarmResult:
    async with sem:
        t0 = time.perf_counter()
        try:
            if task.schema is not None:
                parsed, text, meta = await task.backend.generate_json(
                    task.prompt, task.schema, system=task.system, model=task.model,
                )
            else:
                parsed = None
                text, meta = await task.backend.generate(
                    task.prompt, system=task.system, model=task.model,
                )
            result = SwarmResult(
                key=task.key,
                backend=task.backend.name,
                model=task.model,
                success=True,
                text=text,
                parsed=parsed,
                meta=meta,
            )
        except Exception as e:
            result = SwarmResult(
                key=task.key,
                backend=task.backend.name,
                model=task.model,
                success=False,
                error=str(e),
                meta={"wall_ms": int((time.perf_counter() - t0) * 1000)},
            )
        # Measured the same way observed() measures a whole tool call, and
        # for the same reason: wall_ms is backend-reported and absent on some
        # paths, so without this every swarm subtask records total_ms=NULL and
        # reads as "avg 0ms" in usage stats. Taken inside the semaphore, so it
        # excludes queue wait — consistent with the failure path's wall_ms.
        total_ms = int((time.perf_counter() - t0) * 1000)

    if tool_name is not None:
        _record_subtask(tool_name, task, result, total_ms)

    return result


def _record_subtask(
    tool_name: str, task: SwarmTask, result: SwarmResult, total_ms: int,
) -> None:
    from .grading import schedule_grading
    from .telemetry import record

    synthetic_tool = f"{tool_name}:{result.key}"
    event = {
        "tool": synthetic_tool,
        "ok": result.success,
        "model": result.model,
        "backend": result.backend,
        "input_chars": len(task.prompt),
        "output_chars": len(result.text),
        "total_ms": total_ms,
        "error": result.error,
        **result.meta,
    }
    call_id = record(event)
    schedule_grading(synthetic_tool, task.prompt, result.text, call_id)


async def run_swarm(
    tasks: list[SwarmTask],
    tool_name: str | None = None,
    concurrency: dict[str, int] | None = None,
) -> list[SwarmResult]:
    """Run all tasks concurrently, capped per backend type, process-wide.

    Returns results in the same order as tasks, regardless of completion
    order. A task that raises never propagates — it becomes a failed
    SwarmResult so one bad candidate/dimension can't sink the others.
    """
    caps = concurrency or DEFAULT_CONCURRENCY
    for task in tasks:
        name = task.backend.name
        if name not in _semaphores:
            limit = caps.get(name, DEFAULT_CONCURRENCY.get(name, _FALLBACK_CAP))
            _semaphores[name] = asyncio.Semaphore(limit)

    coros = [_run_one(task, _semaphores[task.backend.name], tool_name) for task in tasks]
    return await asyncio.gather(*coros)


def combine_meta(results: list[SwarmResult]) -> dict:
    """Build a single top-level meta dict summarizing a swarm run."""
    label = ",".join(f"{r.key}={r.model}" for r in results)
    costs = [r.meta.get("cost") for r in results if r.meta.get("cost") is not None]
    return {
        "model": f"swarm({label})",
        "backend": "swarm",
        "wall_ms": max((r.meta.get("wall_ms", 0) for r in results), default=0),
        "eval_ms": max((r.meta.get("eval_ms", 0) for r in results), default=0),
        "prompt_tokens": sum(r.meta.get("prompt_tokens") or 0 for r in results),
        "output_tokens": sum(r.meta.get("output_tokens") or 0 for r in results),
        # None only when no result reported cost data at all — a genuine
        # $0 total (e.g. all-local swarm) must stay 0, not collapse to None.
        "cost": sum(costs) if costs else None,
    }


_WORD = re.compile(r"[a-z0-9_]+")

# Below this many shared words, containment is too easy to satisfy by
# accident ("yes" is trivially contained in any answer starting "yes, ...").
_MIN_SHARED_WORDS = 3


def _similarity(a: str, b: str) -> float:
    """Word-set containment: overlap over the *smaller* text.

    Deliberately not difflib.SequenceMatcher.ratio(), which is 2M/T and so
    penalizes length mismatch directly — two models giving the same correct
    answer at different verbosity scored 0.53 on it, below the old 0.55
    cutoff, while unrelated answers scored 0.34. Containment normalizes by
    the shorter text, scoring that same real pair 0.81 against 0.00 for
    unrelated ones.
    """
    x, y = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not x or not y:
        return 1.0 if x == y else 0.0
    shared = len(x & y)
    if shared < _MIN_SHARED_WORDS:
        return 0.0
    return shared / min(len(x), len(y))


def _cluster(successes: list[SwarmResult], threshold: float = 0.6) -> list[list[SwarmResult]]:
    """Greedy-cluster successful results by pairwise text similarity."""
    clusters: list[list[SwarmResult]] = []
    for r in successes:
        placed = False
        for cluster in clusters:
            if _similarity(cluster[0].text, r.text) >= threshold:
                cluster.append(r)
                placed = True
                break
        if not placed:
            clusters.append([r])
    return clusters


def format_consensus(results: list[SwarmResult]) -> str:
    """Render a multi-model consensus report: agreement/disagreement, not
    just a side-by-side table.

    Similarity is lexical (word-set containment), not semantic — models that
    agree on substance but share no vocabulary still read as disagreement.
    It no longer penalizes verbosity differences, which was the dominant
    false-disagreement cause; see _similarity.
    """
    failed = [r for r in results if not r.success]
    successes = [r for r in results if r.success]

    lines: list[str] = []

    if not successes:
        lines.append("All candidates failed.")
        for r in failed:
            lines.append(f"  [{r.key}] {r.model}: {r.error}")
        return "\n".join(lines)

    if len(successes) < 2:
        for r in successes:
            lines.append(f"── {r.key} ({r.model}) ──")
            lines.append(r.text)
        if failed:
            lines.append("")
            lines.append("Failed:")
            for r in failed:
                lines.append(f"  [{r.key}] {r.model}: {r.error}")
        return "\n".join(lines)

    clusters = sorted(_cluster(successes), key=len, reverse=True)
    majority = clusters[0]

    if len(clusters) == 1:
        lines.append(f"Agreement: {len(successes)}/{len(successes)} models agree.")
    elif len(majority) > len(successes) / 2:
        lines.append(f"Agreement: {len(majority)}/{len(successes)} models agree.")
    else:
        lines.append(f"No majority — all {len(successes)} models disagree.")

    lines.append("")
    lines.append(f"Consensus answer (from {', '.join(r.key for r in majority)}):")
    lines.append(majority[0].text)

    for cluster in clusters[1:]:
        lines.append("")
        preview = cluster[0].text.replace("\n", " ")[:200]
        lines.append(f"Divergent ({', '.join(r.key for r in cluster)}): {preview}")

    lines.append("")
    lines.append("── Latency / tokens ──")
    for r in results:
        if r.success:
            tok = f"{r.meta.get('prompt_tokens', 0)}+{r.meta.get('output_tokens', 0)}"
            lines.append(f"  {r.key} ({r.model}): {r.meta.get('wall_ms', 0)}ms, {tok} tok")
        else:
            lines.append(f"  {r.key} ({r.model}): FAILED — {r.error}")

    return "\n".join(lines)
