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
├── ollama_mcp/           # Python package
│   ├── __init__.py       # re-exports mcp server instance
│   ├── __main__.py       # CLI: serve (default), bench, stats
│   ├── benchmark.py      # multi-model performance comparison
│   ├── client.py         # Ollama HTTP client with error handling
│   ├── config.py         # env-driven settings
│   ├── errors.py         # structured error types
│   ├── privacy.py        # sensitive content detection and intercept
│   ├── schemas.py        # Pydantic models for structured outputs
│   ├── server.py         # FastMCP instance
│   ├── storage.py        # SQLite telemetry and cost tracking
│   ├── telemetry.py      # JSON-lines logging, observed() decorator
│   └── tools.py          # MCP tool definitions
├── tests/                # pytest suite (89 tests)
├── pyproject.toml        # packaging (pip installable)
├── requirements.txt      # mcp[cli], httpx
├── examples/             # toy project built entirely via local delegation
│   └── textkit/          # pure-function text utilities (see Example project)
├── .github/workflows/    # CI runs tests on PRs
├── .gitignore
└── README.md
```

## Tools exposed

| Tool                       | When Claude should call it                                               |
| -------------------------- | ------------------------------------------------------------------------ |
| `local_summarize`          | Long file/log/doc, only the gist matters                                 |
| `local_draft_boilerplate`  | Mechanical scaffolds (Dockerfile, CI workflow, .gitignore, dataclass)    |
| `local_implement_small`    | Self-contained function or short script (~≤50 lines) from a clear spec   |
| `local_commit_message`     | One-line conventional-commit subject from a diff                         |
| `local_review_diff`        | Review a diff/file for bugs, security, complexity (structured output)    |
| `local_generate_tests`     | Generate pytest tests for a Python file                                  |
| `local_usage_stats`        | Show usage statistics and estimated cloud cost avoided                   |
| `local_benchmark`          | Compare prompt performance across multiple local models                  |
| `local_list_models`        | List all models available in the local Ollama instance                   |
| `local_show_routes`        | Show current model routing configuration                                 |
| `local_classify_task`      | Classify a prompt and recommend the best tool and model                  |

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

# Option A: pip install (editable)
.venv/bin/pip install -e .

# Option B: from requirements
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
     -m ollama_mcp
```

Replace `gemma4-32k` with whatever `ollama list` shows for your model.
Use `-s project` instead if you only want one repo to see the server.

If installed via `pip install` (non-editable), the console script also works:

```bash
claude mcp add -s user ollama-local \
  --env OLLAMA_MODEL=gemma4-32k \
  -- $(pwd)/.venv/bin/ollama-mcp
```

Verify:

```bash
claude mcp list
# inside a session:
/mcp
```

## Configuration (env vars)

| Var                  | Default                                     | Purpose                              |
| -------------------- | ------------------------------------------- | ------------------------------------ |
| `OLLAMA_URL`         | `http://localhost:11434`                    | Where to reach Ollama                |
| `OLLAMA_MODEL`       | `gemma4-32k`                                | Model tag to use                     |
| `OLLAMA_MCP_LOG`     | `~/.cache/ollama_mcp.jsonl`                 | Per-call structured log (JSON lines) |
| `OLLAMA_MCP_DB`      | `~/.cache/ollama_mcp.db`                    | SQLite database for telemetry        |
| `OLLAMA_MCP_PRIVACY` | `~/.config/ollama_mcp/privacy.json`         | Privacy intercept config             |
| `OLLAMA_MCP_ROUTES`  | `~/.config/ollama_mcp/routes.json`          | Model routing config                 |

Set these via `--env` at `claude mcp add`, by editing `~/.claude.json`, or
in `.mcp.json` if registered per-project.

## Model routing

By default, all tools use the model set in `OLLAMA_MODEL`. To route
different tools to different models, create a routing config:

```bash
mkdir -p ~/.config/ollama_mcp
cat > ~/.config/ollama_mcp/routes.json <<'EOF'
{
  "default": "gemma4-32k",
  "routes": {
    "local_review_diff": "deepseek-coder",
    "local_generate_tests": "qwen2.5-coder",
    "local_implement_small": "deepseek-coder",
    "local_summarize": "llama3.1",
    "local_classify_task": "llama3.1"
  }
}
EOF
```

Resolution order for each tool call:
1. Exact match in `routes` → use that model
2. `default` key in the config → use that
3. `OLLAMA_MODEL` env var → use that

Use `local_show_routes` (MCP tool) to see current routing, or
`local_classify_task` to ask a local model which tool and model best fit
a given prompt.

To see which models are installed: `local_list_models` or `ollama list`.

## CLI

The package provides a CLI with three subcommands:

```bash
# Start the MCP server (default, used by Claude Code)
ollama-mcp
ollama-mcp serve

# Benchmark models against a prompt
ollama-mcp bench "Explain closures in Python" -m gemma4-32k,llama3.1
ollama-mcp bench prompts/code_review.md          # reads prompt from file
ollama-mcp bench "Write fizzbuzz" -s "Output code only"  # with system prompt
ollama-mcp bench "Hello"                          # all available models

# Show usage statistics
ollama-mcp stats
```

### Benchmark output

```
model                      latency   tokens   eval_ms   status  notes
-------------------------------------------------------------------------------------
deepseek-coder               3800ms     8+760    3200ms       ok  clean implementation...
qwen2.5-coder                4200ms    10+820    3800ms       ok  well structured with...
llama3.1                     6100ms    12+910    5500ms       ok  verbose but thorough...

Fastest:       deepseek-coder (3800ms)
Most concise:  deepseek-coder (760 output tokens)
```

## Structured output

`local_review_diff` uses Pydantic validation to produce reliable structured
findings. The model is asked for JSON matching a schema; if the response
validates, findings are formatted cleanly:

```
[HIGH] BUG: off-by-one error in loop bound (parser.py:42)
[MEDIUM] SECURITY: user input passed to eval() without sanitization (handler.py:17)
[LOW] STYLE: inconsistent naming — snake_case mixed with camelCase (utils.py:5)

3 findings (1 high, 1 medium, 1 low)
```

If the model returns invalid JSON, the raw text is passed through as a
graceful fallback — the call is never wasted.

New tools can opt into structured output by defining a Pydantic model in
`schemas.py` and calling `generate_json()` from `client.py`.

## Privacy intercept

Every tool input is scanned for sensitive content before processing. The
privacy layer detects:

- **File patterns** (glob): `*.env`, `*.key`, `*.pem`, `credentials.json`,
  `customer_data/*`, `.ssh/*`, etc.
- **Content patterns** (regex): private keys, AWS access keys (`AKIA...`),
  GitHub tokens (`ghp_...`), OpenAI keys (`sk-...`), `password=`, `api_key=`

### Configuration

Create `~/.config/ollama_mcp/privacy.json` (optional — sensible defaults
are built in):

```json
{
  "file_patterns": ["*.env", "*.key", "*.pem", "customer_data/*"],
  "content_patterns": ["-----BEGIN.*PRIVATE KEY-----", "AKIA[0-9A-Z]{16}"],
  "action": "warn"
}
```

| Action         | Behavior                                                  |
| -------------- | --------------------------------------------------------- |
| `"warn"`       | Log the detection, continue processing (default)          |
| `"redact_log"` | Process normally but omit input from telemetry logs       |
| `"reject"`     | Block the call, return an error to Claude                 |

Matched content is redacted in log entries — only the first 6 characters are
recorded.

## Error handling

The server returns actionable error messages instead of raw tracebacks:

| Failure                 | Error raised              | Message includes                          |
| ----------------------- | ------------------------- | ----------------------------------------- |
| Ollama not running      | `OllamaConnectionError`  | URL tried + `ollama serve` hint           |
| Model tag doesn't exist | `OllamaModelNotFound`    | Model name + `ollama pull` hint           |
| Request too slow        | `OllamaTimeout`          | Timeout duration + suggested causes       |
| HTTP 4xx/5xx            | `OllamaServerError`      | Status code + response excerpt            |
| Garbled JSON            | `OllamaMalformedResponse`| What was expected vs. what arrived        |

All errors are subclasses of `OllamaError` and are logged to the telemetry
file before propagating to the MCP client.

## Observability

Every tool call is recorded in two places:

1. **JSON lines** (`OLLAMA_MCP_LOG`) — append-only, good for `tail -f | jq`
2. **SQLite** (`OLLAMA_MCP_DB`) — queryable, powers `local_usage_stats` and
   `ollama-mcp stats`

Each record includes:

- `tool`, `ok`, `error?`
- `input_chars`, `output_chars`
- `total_ms` (whole tool call), `wall_ms` (Ollama HTTP), `eval_ms`
- `prompt_tokens`, `output_tokens` (from Ollama's response metadata)
- `model`, `ts`

Quick aggregation from the JSON log:

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

Or use the built-in stats (via MCP tool or CLI):

```bash
ollama-mcp stats
```

```
Local calls:   42
Successful:    40 (95.2%)
Prompt tokens:  180,000
Output tokens:  35,000
Estimated cloud cost avoided:
  Opus:   $5.3250
  Sonnet: $1.0650
Per tool:
  local_summarize: 20 calls, avg 1200ms
  local_review_diff: 12 calls, avg 2100ms
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
  innermost, `@privacy_guard` between them. `functools.wraps` preserves
  the signature so FastMCP's schema introspection still works.

## Debugging

```bash
# Claude Code will surface stderr from the spawned process here:
claude --mcp-debug

# Or manually start the server to see import / startup errors directly:
.venv/bin/python -m ollama_mcp
# (it will block on stdin waiting for MCP traffic — Ctrl-C to exit;
#  you only need it to *not* crash to know imports are fine)
```

## Adding a tool

1. Add an `async def local_xxx(...)` function in `ollama_mcp/tools.py`
   calling `generate(...)` (or `generate_json(...)` for structured output).
2. Decorate with `@mcp.tool()`, then `@observed("local_xxx")`, then
   `@privacy_guard`.
3. Write a docstring that says clearly **when to use** and **when NOT to
   use** the tool — that's the only thing Claude sees.
4. Restart Claude Code (or `/mcp reconnect`) so the tool list refreshes.

## Running tests

```bash
pip install -e ".[test]"
pytest -v
```

Tests use `respx` to mock Ollama HTTP calls and run entirely offline.
CI runs automatically on pull requests via GitHub Actions.
