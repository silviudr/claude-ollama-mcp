"""MCP tool definitions."""

from .client import generate
from .server import mcp
from .telemetry import observed


@mcp.tool()
@observed("local_summarize")
async def local_summarize(text: str) -> str:
    """Summarize a long file, log, or document using a local model.
    Prefer this over reading the whole thing yourself when only the gist is
    needed. Do NOT use for content where exact wording matters."""
    return await generate(text, system="Summarize concisely. No preamble.")


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
    return await generate(
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
    return await generate(
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
    return await generate(
        diff,
        system="Write one concise conventional-commit subject. No body unless necessary.",
    )
