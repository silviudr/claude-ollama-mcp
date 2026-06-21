"""MCP tool definitions."""

from .analyzer import read_csv_meta
from .benchmark import format_results, run_benchmark
from .privacy import privacy_guard
from .router import get_backends, get_routes_info, resolve
from .schemas import AnalysisResult, ReviewResult, TaskClassification
from .server import mcp
from .storage import get_grading_report, get_stats
from .telemetry import observed


@mcp.tool()
@observed("local_summarize")
@privacy_guard
async def local_summarize(text: str) -> str:
    """Summarize a long file, log, or document using a local model.
    Prefer this over reading the whole thing yourself when only the gist is
    needed. Do NOT use for content where exact wording matters."""
    backend, model = resolve("local_summarize")
    return await backend.generate(
        text, system="Summarize concisely. No preamble.", model=model,
    )


@mcp.tool()
@observed("local_draft_boilerplate")
@privacy_guard
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
    backend, model = resolve("local_draft_boilerplate")
    return await backend.generate(
        spec,
        system="Output code only. No explanation, no markdown fences.",
        model=model,
    )


@mcp.tool()
@observed("local_implement_small")
@privacy_guard
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
    backend, model = resolve("local_implement_small")
    return await backend.generate(
        spec,
        system=(
            "You are implementing a single small, self-contained piece of code. "
            "Output only the code. No explanation, no markdown fences, no preamble."
        ),
        model=model,
    )


@mcp.tool()
@observed("local_commit_message")
@privacy_guard
async def local_commit_message(diff: str) -> str:
    """Draft a single conventional-commit subject line from a unified diff.
    Use after staging changes when a quick message is wanted."""
    backend, model = resolve("local_commit_message")
    return await backend.generate(
        diff,
        system="Write one concise conventional-commit subject. No body unless necessary.",
        model=model,
    )


@mcp.tool()
@observed("local_review_diff")
@privacy_guard
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

    system = (
        "You are a code reviewer. Analyze the provided diff or code.\n"
        "Return your review as JSON with a 'findings' array and a 'summary' string.\n"
        "Each finding has: severity (HIGH/MEDIUM/LOW), category "
        "(BUG/SECURITY/PERFORMANCE/COMPLEXITY/STYLE/MISSING_TEST/RISKY_ASSUMPTION), "
        "message, and optionally file and line.\n"
        "If the code looks clean, return an empty findings array."
        f"{focus_instruction}"
    )

    backend, model = resolve("local_review_diff")
    parsed, raw, meta = await backend.generate_json(
        diff, ReviewResult, system=system, model=model,
    )
    if parsed is not None:
        return parsed.format(), meta
    return raw, meta


@mcp.tool()
@observed("local_generate_tests")
@privacy_guard
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

    backend, model = resolve("local_generate_tests")
    return await backend.generate(
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
        model=model,
    )


@mcp.tool()
async def local_usage_stats() -> str:
    """Return usage statistics for all tool calls: total calls,
    success rate, token counts, per-tool and per-backend breakdown,
    cost avoided (Ollama) and cost spent (OpenRouter).

    Use when:
      - User asks how much local delegation has saved
      - User wants to see usage or performance stats
      - Reporting on local vs. cloud trade-offs"""
    stats = get_stats()

    total = stats["total_calls"]
    if total == 0:
        return "No tool calls recorded yet."

    ok = stats["successful"]
    rate = ok / total * 100

    lines = [
        f"Total calls:   {total}",
        f"Successful:    {ok} ({rate:.1f}%)",
        f"Failed:        {stats['failed']}",
        f"Avg latency:   {stats['avg_total_ms']}ms",
        "",
        f"Prompt tokens:  {stats['total_prompt_tokens']:,}",
        f"Output tokens:  {stats['total_output_tokens']:,}",
    ]

    # Per-backend breakdown
    per_backend = stats.get("per_backend", {})
    for name in sorted(per_backend):
        b = per_backend[name]
        b_rate = b["successful"] / b["calls"] * 100 if b["calls"] else 0
        lines += [
            "",
            f"── {name} ──",
            f"  Calls:       {b['calls']} ({b_rate:.0f}% success)",
            f"  Avg latency: {b['avg_ms']}ms",
            f"  Tokens:      {b['prompt_tokens']:,} in / {b['output_tokens']:,} out",
        ]
        if name == "ollama":
            avoided = b.get("estimated_cost_avoided", {})
            if avoided:
                lines.append("  Cost avoided (vs cloud):")
                for tier, amount in sorted(avoided.items()):
                    lines.append(f"    {tier.capitalize()}: ${amount:.4f}")
        elif name == "openrouter":
            cost = b.get("total_cost", 0)
            if cost > 0:
                lines.append(f"  Cost spent:  ${cost:.6f}")
            else:
                lines.append("  Cost spent:  $0 (free models or cost not reported)")

    if stats["per_tool"]:
        lines += ["", "Per tool:"]
        for t in stats["per_tool"]:
            lines.append(
                f"  {t['tool']}: {t['calls']} calls, "
                f"{t['prompt_tokens']:,}+{t['output_tokens']:,} tok, "
                f"avg {t['avg_ms']}ms"
            )

    grading = stats.get("grading", {})
    if grading:
        lines += ["", "Grading:"]
        for tool in sorted(grading):
            for gtype, g in sorted(grading[tool].items()):
                pass_rate = g["passed"] / g["total"] * 100 if g["total"] else 0
                lines.append(
                    f"  {tool} ({gtype}): {g['total']} checks, "
                    f"{pass_rate:.0f}% pass, avg score {g['avg_score']:.2f}"
                )

    return "\n".join(lines)


@mcp.tool()
async def local_grading_report(tool: str = "") -> str:
    """Show a detailed grading quality report for local model outputs.
    Includes per-tool pass rates, top failing checkers, semantic score
    distributions, and recent failures with details.

    Use when:
      - User asks "how are the grades looking?" or "what's the quality?"
      - Investigating which tools or models produce low-quality output
      - Deciding whether to adjust routing or model selection
      - Reviewing grading trends before enabling adaptive routing

    Args:
        tool: Optional tool name to filter by (e.g. "local_implement_small").
              If empty, shows report across all tools."""
    report = get_grading_report(tool=tool)

    s = report["summary"]
    if s["total_checks"] == 0:
        msg = "No grading data yet."
        if tool:
            msg += f" (filtered by {tool})"
        msg += " Use the local_* tools to generate data — grading runs automatically."
        return msg

    pass_rate = s["passed"] / s["total_checks"] * 100 if s["total_checks"] else 0

    lines = ["═══ Grading Report ═══"]
    if tool:
        lines.append(f"Filter: {tool}")
    lines += [
        "",
        f"Total checks:    {s['total_checks']}",
        f"Passed:          {s['passed']} ({pass_rate:.0f}%)",
        f"Failed:          {s['failed']}",
        f"Avg score:       {s['avg_score']:.2f}",
        f"Calls graded:    {s['calls_graded']}",
    ]

    # Per-tool breakdown
    if report["breakdown"]:
        lines += ["", "── Per-tool breakdown ──"]
        current_tool = None
        for b in report["breakdown"]:
            if b["tool"] != current_tool:
                current_tool = b["tool"]
                lines.append(f"\n  {current_tool}:")
            b_rate = b["passed"] / b["total"] * 100 if b["total"] else 0
            extra = ""
            if b["grade_type"] == "semantic" and b["avg_grader_ms"]:
                extra = f", avg {b['avg_grader_ms']}ms"
            lines.append(
                f"    {b['grade_type']}: {b['total']} checks, "
                f"{b_rate:.0f}% pass, avg score {b['avg_score']:.2f}{extra}"
            )

    # Top failing checkers
    if report["top_failures"]:
        lines += ["", "── Top failing checkers ──"]
        for f in report["top_failures"]:
            lines.append(f"  {f['checker']} ({f['tool']}): {f['fail_count']} failures")

    # Semantic score distribution
    if report["semantic_scores"]:
        lines += ["", "── Semantic scores by tool ──"]
        for sc in report["semantic_scores"]:
            lines.append(
                f"  {sc['tool']}: "
                f"avg {sc['avg_score']:.2f}, "
                f"min {sc['min_score']:.2f}, "
                f"max {sc['max_score']:.2f} "
                f"({sc['count']} graded)"
            )

    # Model ranking (overall)
    if report["model_ranking"]:
        lines += ["", "── Model scoreboard (overall) ──"]
        for i, mr in enumerate(report["model_ranking"]):
            mr_rate = mr["passed"] / mr["total"] * 100 if mr["total"] else 0
            medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
            lines.append(
                f"  {medal} {mr['model']}: "
                f"avg {mr['avg_score']:.2f}, "
                f"{mr_rate:.0f}% pass, "
                f"{mr['total']} checks, "
                f"{mr['tools_used']} tools, "
                f"avg {mr['avg_latency_ms']}ms"
            )

    # Per-model per-tool breakdown
    if report["model_scores"]:
        lines += ["", "── Model scores by tool ──"]
        current_tool = None
        for ms in report["model_scores"]:
            if ms["tool"] != current_tool:
                current_tool = ms["tool"]
                lines.append(f"\n  {current_tool}:")
            ms_rate = ms["passed"] / ms["total"] * 100 if ms["total"] else 0
            lines.append(
                f"    {ms['model']} ({ms['grade_type']}): "
                f"avg {ms['avg_score']:.2f} "
                f"[{ms['min_score']:.2f}–{ms['max_score']:.2f}], "
                f"{ms_rate:.0f}% pass ({ms['total']})"
            )

    # Recent failures
    if report["recent_failures"]:
        lines += ["", "── Recent failures (newest first) ──"]
        for rf in report["recent_failures"][:10]:
            detail_str = ""
            if rf["details"]:
                reason = rf["details"].get("reason", "")
                if reason:
                    detail_str = f" — {reason}"
                elif "error" in rf["details"]:
                    detail_str = f" — {rf['details']['error']}"
            lines.append(
                f"  [{rf['grade_type']}] {rf['tool']} / {rf['checker']}"
                f"{detail_str}"
            )

    return "\n".join(lines)


@mcp.tool()
async def local_benchmark(prompt: str, system: str = "", models: str = "") -> str:
    """Run the same prompt against multiple local Ollama models and compare
    latency, token counts, and output. Returns a comparison table.

    Use when:
      - User wants to compare local model performance
      - Choosing which model to use for a specific task type
      - Evaluating a new model against existing ones

    Args:
        prompt: The prompt to send to each model.
        system: Optional system prompt applied to all models.
        models: Optional comma-separated list of model names to test.
                If empty, benchmarks all models available in Ollama."""
    model_list = (
        [m.strip() for m in models.split(",") if m.strip()]
        if models.strip()
        else None
    )
    results = await run_benchmark(
        prompt,
        system=system or None,
        models=model_list,
    )
    return format_results(results)


@mcp.tool()
async def local_list_models() -> str:
    """List all models available across configured backends.

    Use when:
      - You need to know which local models are installed
      - Before calling local_benchmark to pick model names
      - User asks what models are available locally"""
    backends = get_backends()
    sections = []
    for name in sorted(backends):
        backend = backends[name]
        try:
            models = await backend.list_models()
            if models:
                header = f"{name} models:"
                sections.append(
                    header + "\n" + "\n".join(f"  - {m}" for m in models)
                )
            else:
                sections.append(f"{name}: no models available")
        except Exception as e:
            sections.append(f"{name}: error listing models ({e})")

    if not sections:
        return "No backends configured."
    return "\n\n".join(sections)


@mcp.tool()
async def local_show_routes() -> str:
    """Show the current model routing configuration — which backend and
    model is assigned to which tool.

    Use when:
      - User asks which model handles which task
      - Debugging why a tool is using a particular model
      - Reviewing routing config before changing it"""
    return get_routes_info()


@mcp.tool()
@observed("local_classify_task")
async def local_classify_task(prompt: str) -> str:
    """Classify a task and recommend which local tool and model to use.
    Returns the task type, risk level, recommended tool, recommended model,
    and whether the task is suitable for local processing.

    Use when:
      - Deciding which local tool to call for a given user request
      - Evaluating whether a task should be handled locally or by cloud
      - User asks "which model should handle this?"

    Args:
        prompt: The task description or user request to classify."""
    available_tools = (
        "local_summarize, local_draft_boilerplate, local_implement_small, "
        "local_commit_message, local_review_diff, local_generate_tests"
    )

    backend, model = resolve("local_classify_task")
    parsed, raw, meta = await backend.generate_json(
        prompt,
        TaskClassification,
        system=(
            "You are a task classifier for a local AI routing system.\n"
            "Given a task description, classify it and recommend the best "
            "local tool and model.\n\n"
            f"Available tools: {available_tools}\n\n"
            "Guidelines:\n"
            "- task_type: summarize, code, review, test, explain, boilerplate, or other\n"
            "- risk: low (simple/mechanical), medium (needs care), high (complex/critical)\n"
            "- should_use_local: false if the task needs cross-file reasoning, "
            "repo context, or high accuracy on critical code\n"
            "- recommended_model: suggest 'default' unless the task clearly "
            "benefits from a specialized model (e.g. code models for code tasks)"
        ),
        model=model,
    )

    if parsed is not None:
        return parsed.format(), meta
    return raw, meta


@mcp.tool()
@observed("local_analyze_data")
async def local_analyze_data(
    file_path: str, question: str = "", force_handoff: bool = False
) -> str:
    """Analyze a CSV file locally. Reads the file, profiles columns, and
    scores complexity. Simple datasets are analyzed by the local model.
    Complex datasets return a structured handoff with metadata so Claude
    can take over.

    Use when:
      - User asks to analyze, explore, or summarize a CSV file
      - Quick data profiling or quality checks
      - Initial triage before deeper analysis

    Do NOT use when:
      - The file is not CSV
      - The analysis requires joining multiple files

    Args:
        file_path: Absolute path to the CSV file.
        question: Optional specific question about the data. If empty,
                  a general analysis is performed.
        force_handoff: If true, skip local analysis entirely and return
                       metadata for Claude to handle. Use when the user
                       wants Claude's reasoning regardless of complexity."""
    from .config import ANALYSIS_THRESHOLD

    try:
        meta = read_csv_meta(file_path)
    except (FileNotFoundError, ValueError) as e:
        return str(e), {"model": "none", "wall_ms": 0}

    summary = meta.format_summary()

    if force_handoff or meta.handoff:
        reason = "User requested Claude analysis" if force_handoff else (
            f"Complexity score {meta.complexity_score:.2f} "
            f"exceeds threshold {ANALYSIS_THRESHOLD}"
        )
        handoff_msg = (
            f"HANDOFF: {reason}.\n\n"
            f"{summary}\n\n"
            f"Sample (first 10 rows):\n{meta.sample_as_text()}\n\n"
            "Claude should take over for deeper analysis."
        )
        if question.strip():
            handoff_msg += f"\n\nUser question: {question.strip()}"
        return handoff_msg, {"model": "triage", "wall_ms": 0}

    prompt_parts = [
        f"Dataset summary:\n{summary}",
        f"\nSample data (first 10 rows):\n{meta.sample_as_text()}",
    ]
    if question.strip():
        prompt_parts.append(f"\nSpecific question: {question.strip()}")

    backend, model = resolve("local_analyze_data")
    parsed, raw, gen_meta = await backend.generate_json(
        "\n".join(prompt_parts),
        AnalysisResult,
        system=(
            "You are a data analyst. Analyze the CSV dataset described below.\n"
            "Return JSON with: summary (overview of the data), insights "
            "(list of findings about distributions, outliers, quality issues, "
            "patterns), row_count, and col_count.\n"
            "Be specific — reference column names and values."
        ),
        model=model,
    )

    if parsed is not None:
        return parsed.format(), gen_meta
    return f"{summary}\n\nAnalysis:\n{raw}", gen_meta
