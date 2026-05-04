"""Local Ollama MCP server.

Exposes a small set of MCP tools that delegate work to a local Ollama model
(default: gemma4-32k, override with OLLAMA_MODEL).

Logging goes to a JSON-lines file (default ~/.cache/ollama_mcp.jsonl) — never
to stdout, which is reserved for the MCP stdio protocol.
"""

import functools
import json
import logging
import os
import time
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4-32k")
LOG_PATH = Path(
    os.environ.get("OLLAMA_MCP_LOG", str(Path.home() / ".cache" / "ollama_mcp.jsonl"))
)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("ollama_mcp")
logger.setLevel(logging.INFO)
_handler = logging.FileHandler(LOG_PATH)
_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_handler)
logger.propagate = False  # ensure nothing escapes to stdout

mcp = FastMCP("ollama-local")


def _record(event: dict) -> None:
    event["ts"] = time.time()
    logger.info(json.dumps(event))


async def _gen(prompt: str, system: str | None = None) -> tuple[str, dict]:
    payload: dict = {"model": MODEL, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=180) as c:
        r = await c.post(f"{OLLAMA_URL}/api/generate", json=payload)
        r.raise_for_status()
        data = r.json()
    return data["response"], {
        "wall_ms": int((time.perf_counter() - t0) * 1000),
        "prompt_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
        "eval_ms": (data.get("eval_duration") or 0) // 1_000_000,
        "model": MODEL,
    }


def observed(tool_name: str):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                text, meta = await fn(*args, **kwargs)
                _record(
                    {
                        "tool": tool_name,
                        "ok": True,
                        "input_chars": sum(
                            len(str(v)) for v in list(args) + list(kwargs.values())
                        ),
                        "output_chars": len(text),
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                        **meta,
                    }
                )
                return text
            except Exception as e:
                _record(
                    {
                        "tool": tool_name,
                        "ok": False,
                        "error": repr(e),
                        "total_ms": int((time.perf_counter() - t0) * 1000),
                    }
                )
                raise

        return wrapper

    return deco


@mcp.tool()
@observed("local_summarize")
async def local_summarize(text: str) -> str:
    """Summarize a long file, log, or document using a local model.
    Prefer this over reading the whole thing yourself when only the gist is
    needed. Do NOT use for content where exact wording matters."""
    return await _gen(text, system="Summarize concisely. No preamble.")


@mcp.tool()
@observed("local_draft_boilerplate")
async def local_draft_boilerplate(spec: str) -> str:
    """Generate mechanical, repetitive code where the structure is fully
    dictated by the spec. Output is code only — no prose, no markdown fences.

    Use when the user says things like:
      - "write a Dockerfile for a Python 3.11 service that..."
      - "give me a pytest stub for these functions"
      - "draft a GitHub Actions workflow that runs ruff and pytest"
      - "scaffold a dataclass with these fields"
      - "write a .gitignore for a Node project"
      - "make me a basic Makefile with build/test/clean targets"

    Do NOT use when:
      - The output must reference symbols, types, or paths from existing
        project files
      - The "right" structure depends on unstated conventions in this repo"""
    return await _gen(
        spec,
        system="Output code only. No explanation, no markdown fences.",
    )


@mcp.tool()
@observed("local_implement_small")
async def local_implement_small(spec: str) -> str:
    """Implement a single self-contained function or short script (≤ ~50 lines)
    from a clear, specified interface. Output is code only.

    Use when:
      - User says "write a function that...", "a quick script to...",
        "give me a helper that...", "implement X"
      - The spec names inputs, outputs, and behavior unambiguously
      - The result does NOT need to call into existing project code

    Do NOT use when:
      - The implementation must import from or match existing repo modules
      - Behavior depends on details not in the prompt (config, schemas,
        project conventions)
      - Correctness requires multi-file reasoning or knowledge of the codebase

    Pass a self-contained spec — the local model has no view of the repo."""
    return await _gen(
        spec,
        system=(
            "You are implementing a single small, self-contained piece of code. "
            "Output only the code. No explanation, no markdown fences, no preamble."
        ),
    )


@mcp.tool()
@observed("local_commit_message")
async def local_commit_message(diff: str) -> str:
    """Draft a single conventional-commit subject line from a unified diff.
    Use after staging changes when a quick message is wanted."""
    return await _gen(
        diff,
        system="Write one concise conventional-commit subject. No body unless necessary.",
    )


if __name__ == "__main__":
    mcp.run()
