# claude-ollama-mcp

A small MCP (Model Context Protocol) server that lets Claude Code delegate
specific, low-stakes coding tasks to a local Ollama model (default Gemma 4).
The goal is **routing**, not replacement: Claude stays the orchestrator and
calls these tools when the work is cheap enough to be worth offloading.

## Why this exists

Claude Code is built around Anthropic models. There is no first-class
"route this subagent to Ollama" knob. The clean way to use a local model
from inside Claude Code is to expose it as an MCP tool — Claude then
chooses to call it the same way it calls any other tool. The **tool
descriptions are the router**: edit them when Claude is over- or
under-delegating.

## Layout

```
claude-ollama-mcp/
├── ollama_mcp.py     # the MCP server (stdio transport, FastMCP)
├── requirements.txt  # mcp[cli], httpx
├── examples/         # toy project built entirely via local delegation
│   └── textkit/      # pure-function text utilities (see Example project)
├── .gitignore
└── README.md
```

No package, no tests yet — single-file server by design.

## Tools exposed

| Tool                       | When Claude should call it                                               |
| -------------------------- | ------------------------------------------------------------------------ |
| `local_summarize`          | Long file/log/doc, only the gist matters                                 |
| `local_draft_boilerplate`  | Mechanical scaffolds (Dockerfile, CI workflow, .gitignore, dataclass)    |
| `local_implement_small`    | Self-contained function or short script (~≤50 lines) from a clear spec   |
| `local_commit_message`     | One-line conventional-commit subject from a diff                         |

Each tool's docstring is what Claude sees. Iterate on the docstrings — that
is the tuning loop, not the code.

## Building a 32K context Gemma model

The default Gemma 4 context window is too small for many coding tasks. Create
a custom model with an extended context window:

```bash
# Create a Modelfile
cat > Modelfile <<'EOF'
FROM gemma4
PARAMETER num_ctx 32768
EOF

# Build the custom model
ollama create gemma4-32k -f Modelfile

# Verify it appears
ollama list
```

### Resource requirements

On an NVIDIA RTX 4090, the 32K context model uses approximately **13 GB of
VRAM** and generates at **~112 tok/s**. Adjust `num_ctx` down if your GPU has
less memory — 16384 is a reasonable fallback that roughly halves the VRAM
requirement.

## Setup

```bash
git clone <this-repo>
cd claude-ollama-mcp
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# confirm the exact tag of your local model
ollama list
```

## Register with Claude Code

User scope (available in every project on this machine):

```bash
claude mcp add -s user ollama-local \
  --env OLLAMA_MODEL=gemma4-32k \
  -- $(pwd)/.venv/bin/python \
     $(pwd)/ollama_mcp.py
```

Replace `gemma4-32k` with whatever `ollama list` shows for your model.
Use `-s project` instead if you only want one repo to see the server.

Verify:

```bash
claude mcp list
# inside a session:
/mcp
```

## Configuration (env vars)

| Var              | Default                              | Purpose                              |
| ---------------- | ------------------------------------ | ------------------------------------ |
| `OLLAMA_URL`     | `http://localhost:11434`             | Where to reach Ollama                |
| `OLLAMA_MODEL`   | `gemma4-32k`                         | Model tag to use                     |
| `OLLAMA_MCP_LOG` | `~/.cache/ollama_mcp.jsonl`          | Per-call structured log destination  |

Set these via `--env` at `claude mcp add`, by editing `~/.claude.json`, or
in `.mcp.json` if registered per-project.

## Observability

Every tool call appends one JSON line to `OLLAMA_MCP_LOG` with:

- `tool`, `ok`, `error?`
- `input_chars`, `output_chars`
- `total_ms` (whole tool call), `wall_ms` (Ollama HTTP), `eval_ms`
- `prompt_tokens`, `output_tokens` (from Ollama's response metadata)
- `model`, `ts`

Quick aggregation:

```bash
tail -f ~/.cache/ollama_mcp.jsonl | jq .

jq -s 'group_by(.tool) | map({
  tool: .[0].tool,
  n: length,
  p50_ms: (map(.total_ms) | sort | .[length/2|floor]),
  in_tok: (map(.prompt_tokens // 0) | add),
  out_tok: (map(.output_tokens // 0) | add)
})' ~/.cache/ollama_mcp.jsonl
```

Two metrics worth watching:

1. **Routing rate** — fraction of total tool calls in a session that go to
   `local_*`. If it stays below ~5%, the docstrings need work, not the infra.
2. **Quality regressions** — keep a notes file when Claude has to redo work
   the local model produced. That is the real cost of delegation.

## Example project

The `examples/textkit/` directory contains a toy Python library built entirely
through local delegation as a proof-of-concept. It includes a `TASKS.md` with
20 rows — each one a self-contained spec sized for a single Ollama call:

- **Phase 1** (5 tasks) — boilerplate via `local_draft_boilerplate`
  (`.gitignore`, `pyproject.toml`, CI workflow, etc.)
- **Phase 2** (10 tasks) — pure-function implementations via
  `local_implement_small` (`slugify`, `wrap_by_width`, `redact_secrets`, etc.)
- **Phase 3** (5 tasks) — pytest test files via `local_draft_boilerplate`

Results from the test run (Gemma 4 27B, 32K context, RTX 4090):

| Metric               | Value     |
| -------------------- | --------- |
| Total Ollama calls   | 20        |
| Success rate         | 100%      |
| Tests passing        | 24/24     |
| Orchestrator fixups  | 0         |
| Total Gemma tokens   | ~15,700   |
| Equivalent Opus cost | ~$1.03    |
| Equivalent Sonnet cost | ~$0.21  |
| Actual cost          | **$0.00** |

To try it yourself, start Claude Code inside the example directory and say
*"work TASKS.md top-to-bottom."*

## Critical gotchas

- **Never write to stdout from this process.** stdio is the MCP transport.
  All logging goes to a file via `FileHandler`; `logger.propagate = False`
  to be safe. If you `print()` anywhere, Claude Code will silently drop the
  server.
- **Model tag mismatch** is the #1 setup failure. `ollama list` is the
  authoritative source for `OLLAMA_MODEL`.
- **No streaming.** Claude waits for the full response. Keep prompts focused
  so Ollama doesn't drag.
- **Decorator order matters.** `@mcp.tool()` outermost, `@observed(...)`
  innermost. `functools.wraps` preserves the signature so FastMCP's schema
  introspection still works.

## Debugging

```bash
# Claude Code will surface stderr from the spawned process here:
claude --mcp-debug

# Or manually start the server to see import / startup errors directly:
.venv/bin/python ollama_mcp.py
# (it will block on stdin waiting for MCP traffic — Ctrl-C to exit;
#  you only need it to *not* crash to know imports are fine)
```

## Adding a tool

1. Add an `async def local_xxx(...)` function calling `_gen(...)`.
2. Decorate with `@mcp.tool()` then `@observed("local_xxx")`.
3. Write a docstring that says clearly **when to use** and **when NOT to
   use** the tool — that's the only thing Claude sees.
4. Restart Claude Code (or `/mcp reconnect`) so the tool list refreshes.

## Status / next ideas

- No automated tests yet. A small smoke script that calls `_gen` once
  against the configured model would catch the most common breakage
  (Ollama down, wrong tag).
- If routing rate is too low, try splitting `local_draft_boilerplate` into
  more specific tools (e.g. `local_draft_pytest`, `local_draft_dockerfile`)
  — narrower descriptions get picked more reliably than broad ones.
- If logs grow large, rotate via `logging.handlers.RotatingFileHandler` or
  ship the JSONL into DuckDB / Loki.
