"""MCP tool definitions."""

from .client import generate
from .server import mcp
from .storage import get_stats
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


@mcp.tool()
@observed("local_review_diff")
async def local_review_diff(diff: str, focus: str = "") -> str:
    """Review a code diff or file content for issues. Returns a structured
    list of findings covering: bugs, security concerns, risky assumptions,
    missing tests, and complexity issues.

    Use when:
      - User asks to review a diff, patch, or file for quality
      - A quick local sanity check before committing or opening a PR
      - Screening code that does not require deep repo context

    Do NOT use when:
      - The review requires understanding cross-file dependencies
      - Correctness depends on project conventions, configs, or schemas
      - The diff is very large (>500 lines) — split it first

    Args:
        diff: The unified diff or file content to review.
        focus: Optional comma-separated focus areas to prioritize, e.g.
               "security,performance". When empty, all categories are
               covered equally."""
    focus_instruction = ""
    if focus.strip():
        focus_instruction = f" Focus especially on: {focus.strip()}."

    return await generate(
        diff,
        system=(
            "You are a code reviewer. Analyze the provided diff or code and "
            "return a structured review.\n\n"
            "For each finding, output a line in this format:\n"
            "[SEVERITY] CATEGORY: description (file:line if available)\n\n"
            "Severities: HIGH, MEDIUM, LOW\n"
            "Categories: BUG, SECURITY, PERFORMANCE, COMPLEXITY, STYLE, "
            "MISSING_TEST, RISKY_ASSUMPTION\n\n"
            "End with a one-line summary: 'Summary: N findings "
            "(X high, Y medium, Z low)'\n"
            "If the code looks clean, say 'No issues found.'"
            f"{focus_instruction}"
        ),
    )


@mcp.tool()
@observed("local_generate_tests")
async def local_generate_tests(source: str, context: str = "") -> str:
    """Generate pytest tests for a Python file. Identifies public functions
    and classes, then produces test cases covering normal inputs, edge cases,
    and error conditions. Output is executable Python test code.

    Use when:
      - User asks to generate tests for a standalone module or utility
      - The source file has clear public interfaces (functions, classes)
      - The code is self-contained enough to test without complex fixtures

    Do NOT use when:
      - The code under test has deep dependencies on other project modules
      - Tests need database fixtures, network mocks, or complex setup
      - The file is mostly imports/glue with no testable logic

    Args:
        source: The full Python source code to generate tests for.
        context: Optional description of what the module does or any
                 constraints the tests should respect."""
    context_instruction = ""
    if context.strip():
        context_instruction = f"\n\nAdditional context: {context.strip()}"

    return await generate(
        source,
        system=(
            "You are a test engineer. Given the Python source code, generate "
            "a complete pytest test file.\n\n"
            "Rules:\n"
            "- Import the functions/classes being tested at the top\n"
            "- Test each public function and class method\n"
            "- Include tests for: normal inputs, edge cases (empty, None, "
            "boundary values), and expected errors\n"
            "- Use descriptive test names: test_<function>_<scenario>\n"
            "- Use pytest.raises for expected exceptions\n"
            "- No markdown fences, no explanation — output only valid Python"
            f"{context_instruction}"
        ),
    )


@mcp.tool()
async def local_usage_stats() -> str:
    """Return usage statistics for all local Ollama tool calls: total calls,
    success rate, token counts, per-tool breakdown, and estimated cloud cost
    avoided.

    Use when:
      - User asks how much local delegation has saved
      - User wants to see Ollama usage or performance stats
      - Reporting on local vs. cloud trade-offs"""
    stats = get_stats()

    total = stats["total_calls"]
    if total == 0:
        return "No local tool calls recorded yet."

    ok = stats["successful"]
    rate = ok / total * 100
    in_tok = stats["total_prompt_tokens"]
    out_tok = stats["total_output_tokens"]
    costs = stats["estimated_cost_avoided"]

    lines = [
        f"Local calls:   {total}",
        f"Successful:    {ok} ({rate:.1f}%)",
        f"Failed:        {stats['failed']}",
        f"Avg latency:   {stats['avg_total_ms']}ms",
        "",
        f"Prompt tokens:  {in_tok:,}",
        f"Output tokens:  {out_tok:,}",
        "",
        "Estimated cloud cost avoided:",
        f"  Opus:   ${costs['opus']:.4f}",
        f"  Sonnet: ${costs['sonnet']:.4f}",
    ]

    if stats["per_tool"]:
        lines += ["", "Per tool:"]
        for t in stats["per_tool"]:
            lines.append(
                f"  {t['tool']}: {t['calls']} calls, "
                f"{t['prompt_tokens']:,}+{t['output_tokens']:,} tok, "
                f"avg {t['avg_ms']}ms"
            )

    return "\n".join(lines)
