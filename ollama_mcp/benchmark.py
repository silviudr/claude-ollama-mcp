"""Run the same prompt against multiple models and compare results."""

from __future__ import annotations

from dataclasses import dataclass

from .client import generate, list_models


@dataclass
class ModelResult:
    model: str
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
) -> list[ModelResult]:
    if models is None:
        models = await list_models()

    results: list[ModelResult] = []
    for model in models:
        try:
            text, meta = await generate(prompt, system=system, model=model)
            results.append(ModelResult(
                model=model,
                success=True,
                response=text,
                wall_ms=meta.get("wall_ms", 0),
                prompt_tokens=meta.get("prompt_tokens") or 0,
                output_tokens=meta.get("output_tokens") or 0,
                eval_ms=meta.get("eval_ms", 0),
            ))
        except Exception as e:
            results.append(ModelResult(
                model=model,
                success=False,
                response="",
                wall_ms=0,
                prompt_tokens=0,
                output_tokens=0,
                eval_ms=0,
                error=str(e),
            ))

    return results


def format_results(results: list[ModelResult]) -> str:
    if not results:
        return "No models available for benchmarking."

    # Header
    lines = [
        f"{'model':<25} {'latency':>8} {'tokens':>8} {'eval_ms':>8} {'status':>8}  notes",
        "-" * 85,
    ]

    for r in sorted(results, key=lambda r: r.wall_ms if r.success else 999_999):
        if r.success:
            tok = f"{r.prompt_tokens}+{r.output_tokens}"
            preview = r.response.replace("\n", " ")[:40]
            lines.append(
                f"{r.model:<25} {r.wall_ms:>7}ms {tok:>8} {r.eval_ms:>7}ms {'ok':>8}  {preview}"
            )
        else:
            lines.append(
                f"{r.model:<25} {'—':>8} {'—':>8} {'—':>8} {'FAIL':>8}  {r.error or 'unknown'}"
            )

    succeeded = [r for r in results if r.success]
    if len(succeeded) >= 2:
        fastest = min(succeeded, key=lambda r: r.wall_ms)
        fewest_tok = min(succeeded, key=lambda r: r.output_tokens)
        lines.append("")
        lines.append(f"Fastest:       {fastest.model} ({fastest.wall_ms}ms)")
        lines.append(f"Most concise:  {fewest_tok.model} ({fewest_tok.output_tokens} output tokens)")

    return "\n".join(lines)
