"""Run the same prompt against multiple models and compare results."""

from __future__ import annotations

from dataclasses import dataclass, field

from .backends import Backend
from .swarm import SwarmTask, run_swarm


@dataclass
class ModelResult:
    model: str
    backend: str
    success: bool
    response: str
    wall_ms: int
    prompt_tokens: int
    output_tokens: int
    eval_ms: int
    error: str | None = None


async def run_benchmark(
    prompt: str,
    system: str | None = None,
    models: list[str] | None = None,
    backend: Backend | None = None,
    targets: list[tuple[Backend, str]] | None = None,
) -> list[ModelResult]:
    """Benchmark one or more models.

    Args:
        targets: Explicit (backend, model) pairs. Takes priority.
        models: Model names to benchmark on a single backend.
        backend: Backend to use with ``models``. Defaults to the
                 default Ollama backend resolved from config.
    """
    if targets is None:
        if backend is None:
            from .router import resolve
            backend, _ = resolve("local_benchmark")
        if models is None:
            models = await backend.list_models()
        targets = [(backend, m) for m in models]

    from .router import get_swarm_concurrency

    tasks = [
        SwarmTask(key=model, backend=be, model=model, prompt=prompt, system=system)
        for be, model in targets
    ]
    # No tool_name: benchmarking is a side-effect-free comparison, not a
    # judged tool call — it must not trigger telemetry/grading per model
    # (that would silently multiply grading cost by the number of models).
    swarm_results = await run_swarm(tasks, concurrency=get_swarm_concurrency())

    return [
        ModelResult(
            model=r.model,
            backend=r.backend,
            success=r.success,
            response=r.text,
            wall_ms=r.meta.get("wall_ms", 0),
            prompt_tokens=r.meta.get("prompt_tokens") or 0,
            output_tokens=r.meta.get("output_tokens") or 0,
            eval_ms=r.meta.get("eval_ms", 0),
            error=r.error,
        )
        for r in swarm_results
    ]


def format_results(results: list[ModelResult]) -> str:
    if not results:
        return "No models available for benchmarking."

    multi_backend = len({r.backend for r in results}) > 1

    lines = [
        f"{'model':<25} {'latency':>8} {'tokens':>8} {'eval_ms':>8} {'status':>8}  notes",
        "-" * 85,
    ]

    for r in sorted(results, key=lambda r: r.wall_ms if r.success else 999_999):
        name = f"{r.backend}/{r.model}" if multi_backend else r.model
        if r.success:
            tok = f"{r.prompt_tokens}+{r.output_tokens}"
            preview = r.response.replace("\n", " ")[:40]
            lines.append(
                f"{name:<25} {r.wall_ms:>7}ms {tok:>8} {r.eval_ms:>7}ms {'ok':>8}  {preview}"
            )
        else:
            lines.append(
                f"{name:<25} {'—':>8} {'—':>8} {'—':>8} {'FAIL':>8}  {r.error or 'unknown'}"
            )

    succeeded = [r for r in results if r.success]
    if len(succeeded) >= 2:
        fastest = min(succeeded, key=lambda r: r.wall_ms)
        fewest_tok = min(succeeded, key=lambda r: r.output_tokens)
        lines.append("")
        lines.append(f"Fastest:       {fastest.model} ({fastest.wall_ms}ms)")
        lines.append(f"Most concise:  {fewest_tok.model} ({fewest_tok.output_tokens} output tokens)")

    return "\n".join(lines)
