"""Entry point: python -m ollama_mcp"""

import argparse
import asyncio
import sys

from . import mcp


def main():
    parser = argparse.ArgumentParser(
        prog="ollama-mcp",
        description="Local Ollama MCP server and utilities",
    )
    sub = parser.add_subparsers(dest="command")

    # --- serve (default) ---
    sub.add_parser("serve", help="Start the MCP server (default)")

    # --- bench ---
    bench_parser = sub.add_parser("bench", help="Benchmark models against a prompt")
    bench_parser.add_argument(
        "prompt",
        help="Prompt string, or path to a file containing the prompt",
    )
    bench_parser.add_argument(
        "-m", "--models",
        help="Comma-separated model names (default: all available)",
        default="",
    )
    bench_parser.add_argument(
        "-s", "--system",
        help="System prompt to apply to all models",
        default="",
    )

    # --- stats ---
    sub.add_parser("stats", help="Show usage statistics and cost avoidance")

    args = parser.parse_args()

    if args.command == "bench":
        asyncio.run(_bench(args))
    elif args.command == "stats":
        _stats()
    else:
        mcp.run()


async def _bench(args):
    from .benchmark import format_results, run_benchmark

    prompt = args.prompt
    try:
        from pathlib import Path
        p = Path(prompt)
        if p.is_file():
            prompt = p.read_text()
    except OSError:
        pass

    model_list = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models.strip()
        else None
    )
    results = await run_benchmark(
        prompt,
        system=args.system or None,
        models=model_list,
    )
    print(format_results(results))


def _stats():
    from .storage import get_stats

    stats = get_stats()
    total = stats["total_calls"]
    if total == 0:
        print("No local tool calls recorded yet.")
        return

    ok = stats["successful"]
    rate = ok / total * 100
    in_tok = stats["total_prompt_tokens"]
    out_tok = stats["total_output_tokens"]
    costs = stats["estimated_cost_avoided"]

    print(f"Local calls:   {total}")
    print(f"Successful:    {ok} ({rate:.1f}%)")
    print(f"Failed:        {stats['failed']}")
    print(f"Avg latency:   {stats['avg_total_ms']}ms")
    print()
    print(f"Prompt tokens:  {in_tok:,}")
    print(f"Output tokens:  {out_tok:,}")
    print()
    print("Estimated cloud cost avoided:")
    print(f"  Opus:   ${costs['opus']:.4f}")
    print(f"  Sonnet: ${costs['sonnet']:.4f}")

    if stats["per_tool"]:
        print()
        print("Per tool:")
        for t in stats["per_tool"]:
            print(
                f"  {t['tool']}: {t['calls']} calls, "
                f"{t['prompt_tokens']:,}+{t['output_tokens']:,} tok, "
                f"avg {t['avg_ms']}ms"
            )


if __name__ == "__main__":
    main()
