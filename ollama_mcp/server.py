"""FastMCP server instance."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "ollama-local",
    instructions=(
        "This server exposes local models for self-contained work. Before "
        "writing any new function, script, scaffold, or commit message "
        "yourself, check whether a local_* tool fits and call it — that is "
        "the point of this server existing.\n\n"
        "When you skip a local_* tool, say why in one line, in the same turn "
        "you decide to skip — not retrospectively when asked. Two skip "
        "reasons are NOT equivalent, so name which one applies:\n"
        "  - invariant: the work requires holding a design/security "
        "invariant across multiple files at once (e.g. a data-handling "
        "boundary) — genuinely non-delegable.\n"
        "  - missing-context: the work only looks cross-file because the "
        "relevant schema/convention/interface hasn't been included in the "
        "prompt yet — inject that context and delegate anyway, don't skip.\n\n"
        "If unsure which applies, call local_classify_task first."
    ),
)
