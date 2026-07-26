# claude-ollama-mcp

A small MCP (Model Context Protocol) server that lets Claude Code delegate
specific, low-stakes coding tasks to a local Ollama model or to
[OpenRouter](https://openrouter.ai) (a cloud LLM gateway with hundreds of
models). The goal is **routing**, not replacement: Claude stays the
orchestrator and calls these tools when the work is cheap enough to be worth
offloading.

**No GPU?** No problem — configure OpenRouter as your backend and route
every tool through the cloud for pennies per call.

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
│   ├── analyzer.py       # CSV triage and data profiling
│   ├── backends/         # pluggable LLM backends
│   │   ├── base.py       # Backend ABC (generate, generate_json, list_models)
│   │   ├── ollama.py     # local Ollama inference
│   │   └── openrouter.py # OpenRouter cloud inference (OpenAI-compatible)
│   ├── benchmark.py      # multi-model performance comparison
│   ├── client.py         # compatibility shim (delegates to OllamaBackend)
│   ├── config.py         # env-driven settings
│   ├── errors.py         # structured error types (Ollama + OpenRouter)
│   ├── grading/          # async output quality grading framework
│   │   ├── capacity.py   # Ollama VRAM/model capacity detection
│   │   ├── engine.py     # async orchestration, sampling, config resolution
│   │   ├── heuristics.py # zero-cost mechanical checks (AST, format, regex)
│   │   └── semantic.py   # LLM-based rubric grading via OpenRouter or local
│   ├── privacy.py        # sensitive content detection and intercept
│   ├── router.py         # per-tool backend + model routing
│   ├── schemas.py        # Pydantic models for structured outputs
│   ├── server.py         # FastMCP instance
│   ├── storage.py        # SQLite telemetry, cost tracking, and grading storage
│   ├── telemetry.py      # JSON-lines logging, observed() decorator
│   └── tools.py          # MCP tool definitions
├── tests/                # pytest suite (334 tests)
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
| `local_review_diff`        | Review a diff/file for bugs, security, complexity (structured output). Fans out to parallel dimension reviewers when `swarm.review_dimensions` is configured — see [Agent swarm](#agent-swarm) |
| `local_generate_tests`     | Generate pytest tests for a Python file                                  |
| `local_usage_stats`        | Show usage statistics and estimated cloud cost avoided                   |
| `local_benchmark`          | Compare prompt performance across multiple local models (runs concurrently) |
| `local_consensus`          | Send one prompt to multiple models concurrently, report agreement/disagreement — see [Agent swarm](#agent-swarm) |
| `local_list_models`        | List all models available in the local Ollama instance                   |
| `local_show_routes`        | Show current model routing configuration                                 |
| `local_classify_task`      | Classify a prompt and recommend the best tool and model                  |
| `local_analyze_data`       | Analyze a CSV file — triage locally or hand off to Claude                |
| `local_grading_report`     | Show quality grades, model scoreboard, and failing checkers              |

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

If using OpenRouter (with or without Ollama), add the API key:

```bash
claude mcp add -s user ollama-local \
  --env OLLAMA_MODEL=gemma4-32k \
  --env OPENROUTER_API_KEY=sk-or-v1-... \
  -- $(pwd)/.venv/bin/python \
     -m ollama_mcp
```

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

**Core:**

| Var                  | Default                                     | Purpose                              |
| -------------------- | ------------------------------------------- | ------------------------------------ |
| `OLLAMA_URL`         | `http://localhost:11434`                    | Where to reach Ollama                |
| `OLLAMA_MODEL`       | `gemma4-32k`                                | Default model tag                    |
| `OPENROUTER_API_KEY` | *(none)*                                    | OpenRouter API key (required if using OpenRouter via `api_key_env`) |
| `OLLAMA_MCP_ROUTES`  | `~/.config/ollama_mcp/routes.json`          | Backend and model routing config     |

**Telemetry & privacy:**

| Var                  | Default                                     | Purpose                              |
| -------------------- | ------------------------------------------- | ------------------------------------ |
| `OLLAMA_MCP_LOG`     | `~/.cache/ollama_mcp.jsonl`                 | Per-call structured log (JSON lines) |
| `OLLAMA_MCP_DB`      | `~/.cache/ollama_mcp.db`                    | SQLite database for telemetry        |
| `OLLAMA_MCP_PRIVACY` | `~/.config/ollama_mcp/privacy.json`         | Privacy intercept config             |

**Data analyzer:**

| Var                  | Default                                     | Purpose                              |
| -------------------- | ------------------------------------------- | ------------------------------------ |
| `OLLAMA_MCP_SAMPLE_ROWS` | `50`                                   | Rows to sample for data analysis     |
| `OLLAMA_MCP_MAX_COLS` | `20`                                       | Column count threshold for handoff   |
| `OLLAMA_MCP_COMPLEXITY_THRESHOLD` | `0.7`                          | Complexity score above which to hand off |

**Grading:**

| Var                  | Default                                     | Purpose                              |
| -------------------- | ------------------------------------------- | ------------------------------------ |
| `OLLAMA_MCP_GRADING` | `1`                                         | Set to `0` to disable grading        |
| `OLLAMA_MCP_GRADING_SAMPLE_RATE` | `0.2`                          | Fraction of calls that get semantic grading |

Set these via `--env` at `claude mcp add`, by editing `~/.claude.json`, or
in `.mcp.json` if registered per-project.

## Backends & routing

The server supports two backends:

| Backend        | Where it runs    | Cost       | Needs GPU? | Best for                       |
| -------------- | ---------------- | ---------- | ---------- | ------------------------------ |
| **Ollama**     | Your machine     | Free       | Yes        | Fast iteration, privacy        |
| **OpenRouter** | openrouter.ai    | Per-token  | No         | No GPU, access to many models  |

Concurrent fan-out calls (agent swarm, `local_benchmark`) cap concurrency
per backend type — see [Agent swarm](#agent-swarm) for the defaults.

### Quick start: Ollama only (default)

By default, all tools use the local Ollama model set in `OLLAMA_MODEL`.
To route different tools to different local models:

```bash
mkdir -p ~/.config/ollama_mcp
cat > ~/.config/ollama_mcp/routes.json <<'EOF'
{
  "default": "gemma4-32k",
  "routes": {
    "local_review_diff": "deepseek-coder",
    "local_generate_tests": "qwen2.5-coder",
    "local_implement_small": "deepseek-coder"
  }
}
EOF
```

This legacy format still works and is treated as pure Ollama.

### Quick start: OpenRouter (no GPU needed)

1. Get an API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Set the environment variable:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

3. Create the routing config:

```bash
mkdir -p ~/.config/ollama_mcp
cat > ~/.config/ollama_mcp/routes.json <<'EOF'
{
  "default_backend": "openrouter",
  "backends": {
    "openrouter": {
      "api_key_env": "OPENROUTER_API_KEY",
      "default_model": "google/gemma-3-27b-it:free"
    }
  }
}
EOF
```

That's it — all tools now route through OpenRouter. The `:free` suffix
on the model name selects OpenRouter's free tier (rate-limited but $0).

### Hybrid: Ollama + OpenRouter

Mix local and cloud backends per tool. Keep cheap tasks local, route
quality-sensitive ones to a stronger cloud model:

```json
{
  "default_backend": "ollama",
  "backends": {
    "ollama": {
      "url": "http://localhost:11434",
      "default_model": "gemma4-32k"
    },
    "openrouter": {
      "api_key_env": "OPENROUTER_API_KEY",
      "default_model": "google/gemma-3-27b-it:free"
    }
  },
  "routes": {
    "local_review_diff": { "backend": "openrouter", "model": "google/gemma-3-27b-it" },
    "local_generate_tests": { "backend": "openrouter", "model": "google/gemma-3-27b-it" },
    "local_summarize": "gemma4-32k",
    "local_commit_message": "gemma4-32k"
  },
  "grading": {
    "enabled": true,
    "backend": "openrouter",
    "model": "google/gemma-4-31b-it:free",
    "sample_rate": 0.2
  }
}
```

Routes can be either:
- A **bare model string** (`"gemma4-32k"`) — uses the default backend
- A **dict** with `backend` and `model` — explicit backend selection

### Backend configuration reference

**Ollama backend:**
```json
{
  "url": "http://localhost:11434",
  "default_model": "gemma4-32k"
}
```

**OpenRouter backend:**
```json
{
  "api_key_env": "OPENROUTER_API_KEY",
  "default_model": "google/gemma-3-27b-it:free",
  "url": "https://openrouter.ai/api/v1"
}
```

- `api_key_env` — name of the environment variable holding your API key
  (recommended — keeps keys out of config files)
- `api_key` — alternative: put the key directly in the config (less
  secure, but simpler for local-only setups)
- `default_model` — used when a route doesn't specify a model
- `url` — override if using an OpenRouter-compatible proxy (optional)

Browse available models at [openrouter.ai/models](https://openrouter.ai/models).
Free models have `:free` in their ID.

### Resolution order

For each tool call:
1. Exact match in `routes` → use that backend + model
2. `default_backend` → use that backend's `default_model`
3. Fallback → Ollama with `OLLAMA_MODEL` env var

### Useful tools

- `local_show_routes` — see current routing config
- `local_list_models` — list models from all configured backends
- `local_classify_task` — ask which tool and model best fit a prompt
- `local_usage_stats` — per-backend usage, cost avoided, cost spent

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

`local_consensus` and configured `local_review_diff` fan-out are MCP-tool-only
for now — there's no CLI subcommand for them yet (the `bench` subcommand is
the precedent to follow if CLI parity is wanted later).

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

## Data analyzer

`local_analyze_data` provides CSV analysis with an automatic triage
layer. Simple datasets are analyzed locally; complex ones are handed off
to Claude with full metadata.

### How triage works

The analyzer reads the CSV, profiles every column, and scores complexity
on five dimensions:

| Signal                  | What it detects                                  | Score weight |
| ----------------------- | ------------------------------------------------ | ------------ |
| Structural complexity   | Column count > `OLLAMA_MCP_MAX_COLS` (default 20)| proportional |
| Data cardinality        | High ratio of unique values → reasoning challenge| avg ratio    |
| Nested JSON             | JSON strings in column values                    | 0.4 per col  |
| Multiple datetimes      | 2+ datetime columns needing alignment            | 0.3          |
| Free-text columns       | Columns avg > 60 chars / 8 words (NLP needed)    | 0.5 per col  |
| Foreign keys            | 2+ columns ending in `_id`, `_uuid`, `_key`, etc.| 0.3          |

If the weighted score exceeds `OLLAMA_MCP_COMPLEXITY_THRESHOLD` (default
0.7), the tool returns a `HANDOFF` with the dataset profile and sample
rows attached so Claude can take over.

### Usage

```
# Simple dataset → analyzed locally
"Analyze examples/manual-testing/sample_simple.csv"

# Complex dataset → auto-handoff to Claude
"Analyze examples/manual-testing/sample_complex.csv"

# Force Claude to handle it regardless of complexity
"Analyze sample_simple.csv but use Claude for this one"
# (Claude passes force_handoff=true)

# With a specific question
"What's the average salary by department in sample_simple.csv?"
```

### Handoff output

When the triage triggers a handoff, Claude receives the full profile:

```
HANDOFF: Complexity score 0.82 exceeds threshold 0.7.

File: data.csv
Rows: 50000 (sampled 50)
Columns: 22

Reasons:
  - High column count: 22 (max 20)
  - Nested JSON in: metadata, ip_geo, feature_flags
  - Free-text columns: error_message
  - Multiple foreign keys (3): user_id, session_id, order_id

Sample (first 10 rows):
[CSV data]

Claude should take over for deeper analysis.
```

Claude then uses its own reasoning on the metadata and sample — cross-column
correlations, temporal patterns, causal questions — things a local model
can't reliably do.

## Output grading

Every tool call is automatically graded in the background — the grading
never blocks the response to Claude. Grades are stored in SQLite alongside
telemetry, building a quality dataset that tells you which models produce
the best output for which tasks.

### How it works

Grading runs in two layers:

1. **Heuristic checks (100% of calls)** — zero-cost mechanical validation
   that runs instantly after every tool call.
2. **Semantic LLM grading (sampled, default 20%)** — a rubric-based
   evaluation sent to OpenRouter (free tier) or a local model.

Both layers run as async fire-and-forget tasks — the tool response is
returned to Claude immediately, and grading happens in the background.

### Heuristic checks by tool

| Tool                    | Checks                                                                  |
| ----------------------- | ----------------------------------------------------------------------- |
| `local_implement_small` | Python AST syntax, hallucinated imports, truncation, repetition         |
| `local_generate_tests`  | AST syntax, `def test_*` presence, pytest import, **pytest collection** |
| `local_draft_boilerplate` | Truncation, repetition, **format validation** (Dockerfile/YAML/JSON/Makefile/TOML/INI/.gitignore) |
| `local_commit_message`  | Non-empty, conventional-commit format                                   |
| `local_review_diff`     | JSON parseable, truncation, diff file references (phantom file detection) |
| `local_summarize`       | Non-empty, truncation, repetition                                       |
| `local_classify_task`   | Non-empty, JSON parseable                                               |

**Test collection** (`local_generate_tests`): the grader writes the
generated test code and the source code to a temp directory, auto-detects
the module name from imports, and runs `pytest --collect-only`. This
catches broken imports, bad fixtures, and mangled signatures that AST
parsing alone misses.

**Format validation** (`local_draft_boilerplate`): the grader auto-detects
the output format from the spec keywords (e.g., "Dockerfile", "yaml",
"Makefile") or from the output content itself, then validates with the
appropriate parser (`yaml.safe_load`, `json.loads`, `tomllib.loads`,
Dockerfile instruction check, etc.).

### Semantic grading

When sampled, a grading prompt is sent to the configured grading backend
with a rubric tailored to the tool type. The rubric scores four dimensions
(0-5 each):

- **Correctness** — does the output accurately address the input?
- **Completeness** — are important aspects covered?
- **Format** — does it follow the expected format?
- **Conciseness** — appropriately brief without losing meaning?

The grader returns a normalised 0-1 overall score. Scores below 0.5 are
marked as failures.

### Configuration

Add a `grading` section to `~/.config/ollama_mcp/routes.json`:

```json
{
  "default_backend": "ollama",
  "backends": { ... },
  "routes": { ... },
  "grading": {
    "enabled": true,
    "backend": "openrouter",
    "model": "google/gemma-4-31b-it:free",
    "sample_rate": 0.2,
    "tool_sample_rates": {
      "local_review_diff": 1.0,
      "local_implement_small": 1.0,
      "local_generate_tests": 0.5
    }
  }
}
```

| Field                | Default                      | Purpose                                |
| -------------------- | ---------------------------- | -------------------------------------- |
| `enabled`            | `true`                       | Master switch for grading              |
| `backend`            | `"openrouter"`               | Which backend grades outputs           |
| `model`              | `"google/gemma-3-27b-it:free"` | Model used for semantic grading      |
| `sample_rate`        | `0.2`                        | Default fraction of calls that get semantic grading (0.0-1.0) |
| `tool_sample_rates`  | `{}`                         | Per-tool overrides for `sample_rate` — tools not listed use the global rate |

Per-tool sample rates let you run 100% grading on critical tools (like
code review) where quality signal matters most, while keeping low-stakes
tools (like commit messages) at the default rate. This is especially
useful for **adaptive routing** — higher sample rates on tools with
candidates mean faster convergence on model rankings.

Environment variables:

| Var                            | Default | Purpose                         |
| ------------------------------ | ------- | ------------------------------- |
| `OLLAMA_MCP_GRADING`           | `1`     | Set to `0` to disable grading   |
| `OLLAMA_MCP_GRADING_SAMPLE_RATE` | `0.2` | Semantic grading sample rate    |

**GPU capacity detection:** if grading is configured for local Ollama, the
engine queries `/api/ps` on first use to check if a second model can fit
in VRAM. If it can't (e.g., 2 models already loaded), it logs a warning
with a suggested config switch to OpenRouter.

### Model scoreboard

The grading data powers a per-model quality ranking. Since grades are
linked to the `calls` table via `call_id`, you can see which model
produces the best output per tool:

```
═══ Grading Report ═══

Total checks:    150
Passed:          135 (90%)
Avg score:       0.82

── Model scoreboard (overall) ──
  🥇 qwen2.5-coder: avg 0.88, 95% pass, 42 checks, 3 tools, avg 1200ms
  🥈 gemma4-32k:    avg 0.75, 80% pass, 68 checks, 5 tools, avg 2100ms
  🥉 llama3.2:      avg 0.61, 70% pass, 20 checks, 2 tools, avg 800ms

── Model scores by tool ──
  local_implement_small:
    qwen2.5-coder (heuristic): avg 0.95 [0.80–1.00], 100% pass (15)
    gemma4-32k (heuristic):    avg 0.72 [0.40–1.00], 80% pass (25)

── Top failing checkers ──
  conventional_commit (local_commit_message): 12 failures
  json_parseable (local_review_diff): 5 failures
```

Use `local_grading_report` (MCP tool) or query the database directly:

```bash
sqlite3 ~/.cache/ollama_mcp.db "\
  SELECT c.model, g.tool, AVG(g.score) as avg_score, COUNT(*) as checks
  FROM grades g JOIN calls c ON g.call_id = c.id
  WHERE g.score IS NOT NULL
  GROUP BY c.model, g.tool
  ORDER BY g.tool, avg_score DESC"
```

### Testing the grader

**Manual heuristic test** — exercises all checkers with known-good and
known-bad outputs, no Ollama or OpenRouter needed:

```bash
python3 examples/manual-testing/test_grading.py
```

Output shows each checker's pass/fail with reasons:

```
  ✓  local_implement_small — BAD — syntax error
        ✓ non_empty: score=1.0
        ✗ python_syntax: score=0.0  ('(' was never closed)
        ✓ import_check: score=1.0

  ✓  local_draft_boilerplate — BAD — Dockerfile missing FROM
        ✓ non_empty: score=1.0
        ✗ format_valid: score=0.0  (Dockerfile missing FROM instruction)
```

**Live end-to-end test** — use the tools normally and inspect grades:

```bash
# 1. Enable grading with 100% semantic sampling (for testing)
# Set "sample_rate": 1.0 in routes.json grading section

# 2. Use any tool via Claude Code
# "Write a function that validates email addresses"
# "Generate tests for this module"
# "Draft a Dockerfile for Python 3.11"

# 3. Check the grades
sqlite3 ~/.cache/ollama_mcp.db \
  "SELECT tool, grade_type, checker, passed, score, details
   FROM grades ORDER BY ts DESC LIMIT 20"

# 4. Or ask Claude: "show me the grading report"
# (calls local_grading_report)
```

**Unit tests** — 310 tests covering all grading modules:

```bash
pytest tests/test_grading_heuristics.py tests/test_grading_engine.py \
       tests/test_grading_capacity.py tests/test_grading_semantic.py -v
```

## Adaptive routing

Static routes pick one model per tool forever. Adaptive routing learns from
grading data and automatically promotes the best-performing model for each
tool — no manual tuning needed.

### How it works

Adaptive routing uses **epsilon-greedy selection with a recency window**:

1. **Exploit** (default 90%) — pick the candidate model with the highest
   combined score for the tool.
2. **Explore** (default 10%) — pick a random alternative to gather fresh
   data and detect improvements.

The combined score balances quality and speed:

```
combined = quality_weight × avg_quality_score + (1 - quality_weight) × (1 / avg_latency)
```

Only the most recent N graded calls count (the **recency window**), so if
you pull a new model version the old scores fade away naturally.

### Setup

Add an `adaptive` section to `~/.config/ollama_mcp/routes.json`:

```json
{
  "adaptive": {
    "enabled": true,
    "min_samples": 10,
    "quality_weight": 0.7,
    "explore_rate": 0.1,
    "recency_window": 100,
    "candidates": {
      "local_summarize": ["gemma4-32k", "openrouter/google/gemma-4-31b-it:free"],
      "local_implement_small": ["gemma4-32k", "qwen3:8b"],
      "local_review_diff": ["gemma4-32k", "deepseek-coder"]
    }
  }
}
```

**Candidate format:** bare model names (e.g. `"gemma4-32k"`) use the
default backend. Prefixed names (e.g. `"openrouter/google/gemma-4-31b-it:free"`)
route to the named backend.

Tools not listed in `candidates` use their static route as before —
adaptive routing only applies to tools you explicitly opt in.

### Configuration reference

| Key              | Default | Description                                        |
|------------------|---------|----------------------------------------------------|
| `enabled`        | `false` | Master switch for adaptive routing                 |
| `min_samples`    | `10`    | Minimum graded calls before a model is eligible    |
| `quality_weight` | `0.7`   | Weight for quality vs speed (0.0–1.0)              |
| `explore_rate`   | `0.1`   | Fraction of calls that explore non-best models     |
| `recency_window` | `100`   | Only consider the N most recent grades per model   |
| `candidates`     | `{}`    | Map of tool name → list of candidate model strings |

### Cold start and fallback

Until a tool's candidates accumulate `min_samples` graded calls, adaptive
routing falls back to the static route in `routes`. This means:

- **Fresh install:** everything works exactly as before — no adaptive
  behavior until grading data exists.
- **New model added:** add it to `candidates` and it will receive
  `explore_rate` traffic until it builds enough data to compete.

### Monitoring

Call `local_show_routes` to see adaptive state alongside static config:

```
Adaptive routing: enabled
  explore_rate: 0.1, quality_weight: 0.7, min_samples: 10, recency_window: 100
  local_implement_small:
    gemma4-32k: score 1.00, latency 6625ms, combined 0.700 (20 samples)
    qwen3:8b: score 0.95, latency 1200ms, combined 0.668 (15 samples)
  local_summarize:
    gemma4-32k: score 0.98, latency 7400ms, combined 0.686 (12 samples)
    openrouter/google/gemma-4-31b-it:free: warming up (4/10 samples)
```

Models still warming up show their progress toward `min_samples`. Models
with no grading data show "no data yet".

### Tips

- **For testing**, set `min_samples: 2` and `explore_rate: 0.5` so both
  candidates get traffic quickly. Reset to production values afterward.
- **Grading must be enabled** (`grading.enabled: true` in routes.json)
  for adaptive routing to collect scores. Without grading data, candidates
  never reach `min_samples` and the router stays on static routes.
- **Raise `quality_weight`** (toward 1.0) if you care more about output
  quality than speed. Lower it if latency matters more.
- **Pinned routes** are not overridden — if a tool has a static route and
  is not listed in `candidates`, adaptive routing ignores it.


## Agent swarm

Two tools fan out concurrent model calls instead of one call at a time:

- **`local_review_diff`** — parallel dimension review. Instead of one model
  covering every category, N focused reviewers (security, performance,
  correctness, tests, ...) run concurrently, each optionally a different
  model, and their findings are merged into one report.
- **`local_consensus`** — multi-model consensus. The same prompt goes to N
  models concurrently, and the result is an agreement/disagreement report,
  not just a latency table like `local_benchmark`.

Both sit on the same execution primitive (`ollama_mcp/swarm.py`), which caps
concurrency **per backend** — local Ollama is GPU-bound and can thrash VRAM
if too many models load at once, while a cloud backend like OpenRouter can
handle much higher concurrency:

| Backend      | Default concurrency cap |
|--------------|--------------------------|
| `ollama`     | 2                        |
| `openrouter` | 8                        |
| (unknown)    | 4                        |

**Without any `swarm` config, both tools behave exactly as before**:
`local_review_diff` makes its one existing call, and `local_consensus` is
present but returns an explanatory message until candidates are configured.

### Prerequisites

Configuration alone isn't enough — a swarm has to physically fit and the
backend has to actually serve calls in parallel.

**1. Every model you reference must already be pulled.** A dimension pointing
at a missing model doesn't fall back, it fails that dimension:

```bash
ollama list                  # what you have
ollama pull qwen3:8b         # add what you're missing
```

**2. Concurrently-loaded models must fit in VRAM.** This is the constraint
that most often turns a "concurrent" swarm into a slow serial one, because
Ollama evicts and reloads models that don't fit together. Budget by loaded
size, which runs larger than the on-disk size in `ollama list` — a 5.2 GB
model can occupy ~6 GB once loaded with its context:

```bash
nvidia-smi --query-gpu=memory.total,memory.used --format=csv
ollama ps                    # what's resident right now, and how big
```

On a 16 GB card, two mid-size models (say 6 GB + 3 GB) coexist comfortably,
while two 9.6 GB models do not. Prefer several smaller models over two large
ones, or split the swarm across `ollama` and a cloud backend so only some of
it competes for VRAM.

**3. Confirm Ollama serves requests in parallel.** The per-backend cap limits
how many calls this server *sends*; it can't make Ollama handle them at once.
Defaults vary by Ollama version and available memory, so measure rather than
assume:

```bash
# Serial baseline
time ( ollama run qwen3:8b "say hi" >/dev/null 2>&1
       ollama run llama3.2 "say hi" >/dev/null 2>&1 )

# Same two calls, concurrent
time ( ollama run qwen3:8b "say hi" >/dev/null 2>&1 &
       ollama run llama3.2 "say hi" >/dev/null 2>&1 &
       wait )
```

Run each twice and compare the second timings — the first run pays one-time
model loading that swamps the measurement. A healthy result is the concurrent
block landing near the slower single call (e.g. 9.3s serial vs 4.5s
concurrent). If the two timings are about equal, Ollama is serializing.
Raise the limits on the server (not in `routes.json` — these are Ollama's own
settings) and restart it:

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=3 ollama serve
```

Two separate limits are at play, and they fail differently:

| Setting | Governs | Symptom when too low |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | how many *distinct* models stay resident | models evict and reload between sub-tasks |
| `OLLAMA_NUM_PARALLEL` | concurrent requests to *one* model | two calls to the same model run back-to-back |

The second one catches people out: with `OLLAMA_NUM_PARALLEL` at 1,
**pointing several dimensions at the same model gains you nothing** — they
serialize even though the swarm dispatched them together. Two calls to one
model measured 7.21s serial vs 7.22s "concurrent" on a default install. Either
give each dimension a different model, or raise `OLLAMA_NUM_PARALLEL`.

**4. Assign each sub-task a model that can do the job.** Two constraints
combine here and they are easy to miss:

- **Structured-output tools need a model that can follow a JSON schema.**
  `local_review_diff` requires it. Smaller models often return the *schema*
  instead of an instance — that dimension is then reported as unparsed rather
  than silently dropped, but you lose its findings. Test a candidate before
  wiring it to a dimension.
- **Reasoning models are a poor fit for wide fan-out.** A model that emits
  long internal-monologue output can take several times longer on real input
  than a quick benchmark suggests, and a swarm waits for its slowest member.

A worked example on a 16 GB card: of the locally available models, three
could hold the review schema at 6 GB, 6 GB, and 10 GB loaded — but the two
that fit together included a reasoning model that pushed a 4-dimension review
past 100s, while the fastest one couldn't share VRAM with any other. The
workable configuration put two dimensions on a **cloud** backend (no VRAM
cost, genuinely parallel with local work) and two on a local model.

**Pairing local with remote is often the only way to get real parallelism on
a single GPU.** Consider it the default rather than a fallback.

**5. Pick models with similar latency.** A swarm's wall-clock time is its
*slowest* member, so pairing a 4-second model with a 0.3-second one costs
almost as much as the slow one alone. Grouping comparable models is what
turns fan-out into real speedup.

### Configuration

Add a `swarm` section to `~/.config/ollama_mcp/routes.json`:

```json
{
  "swarm": {
    "enabled": true,
    "concurrency": {"ollama": 2, "openrouter": 8},
    "review_dimensions": {
      "local_review_diff": {
        "security": "gemma4-32k",
        "performance": "qwen3:8b",
        "correctness": "openrouter/google/gemma-4-31b-it:free",
        "tests": "gemma4-32k"
      }
    },
    "consensus_candidates": {
      "local_consensus": ["gemma4-32k", "qwen3:8b", "openrouter/google/gemma-4-31b-it:free"]
    }
  }
}
```

Candidate strings use the same format as adaptive routing: a bare name
(`"gemma4-32k"`) uses the default backend, a prefixed name
(`"openrouter/google/gemma-4-31b-it:free"`) targets the named backend.

### Turning swarms on and off

`swarm.enabled` is the single switch. Set it to `false` to turn the whole
feature off **without deleting your configuration**, so you can flip it back
without retyping dimensions and candidates:

```json
{"swarm": {"enabled": false, "review_dimensions": {"...": {}}}}
```

| `swarm.enabled` | Effect |
|-----------------|--------|
| absent, with a `swarm` section present | **on** — presence of config means intent to use it |
| `true` | on |
| `false` | off: `local_review_diff` makes its single call, `local_consensus` explains it was switched off |
| no `swarm` section at all | off |

Note this defaults differently from `adaptive.enabled`, which is opt-in
(absent means off). Swarm defaults to on when a `swarm` section exists so
that configs written before this flag was added keep working on upgrade.

`local_show_routes` reports the current state (`Agent swarm: enabled` /
`Agent swarm: disabled`), so you never have to infer it from behavior.

Two things the switch deliberately does **not** affect:

- `local_benchmark` still fans out and still respects `swarm.concurrency` —
  it is multi-model benchmarking, not an agent swarm, and disabling swarms
  must not leave it uncapped.
- `local_consensus` still honors an explicit `models="a,b"` argument, so a
  one-off comparison works even with swarms off.

If a tool is configured under both `adaptive.candidates` and `swarm.*`, the
swarm config takes precedence.

### Verifying the setup

`routes.json` is re-read on every call, so **config changes take effect
immediately — no restart**. (Upgrading the package itself is code, not
config, and does need the MCP server restarted. With a stdio server that
just means starting a new client session.)

**1. Confirm the config parsed.** Call `local_show_routes`. Nothing is
inferred from behavior — it names the state outright:

```
Agent swarm: enabled
  concurrency: ollama=2, openrouter=4
  local_review_diff review dimensions:
    security -> openrouter/google/gemma-4-31b-it:free
    performance -> qwen3:8b
  local_consensus consensus candidates: qwen3:8b, llama3.2
```

A typo'd key (`review_dimension`, `consensus_candidate`) parses as *absent*
rather than erroring, so if a section you wrote is missing from this output,
check the spelling first.

**2. Watch a swarm run.** Each sub-task appends a row as it completes, so
fan-out looks like rows landing together rather than spaced out:

```bash
tail -f ~/.cache/ollama_mcp.jsonl | jq -c '{tool, model, backend, ok, wall_ms}'
watch -n 0.5 ollama ps       # models resident at the same time
```

**3. Check it actually paralleled.** Compare the sub-task rows for one call:
if the elapsed time for the whole call is close to the *slowest* sub-task,
the swarm ran concurrently; if it approaches the *sum*, something serialized
— revisit prerequisites 2 and 3.

Common first-run surprises:

| Symptom | Cause |
|---|---|
| `local_consensus` says candidates aren't configured | `swarm.enabled` is `false`, or the key is under the wrong tool name |
| One dimension always fails | model not pulled, or a rate-limited/unreachable cloud backend |
| No speedup vs a single call | VRAM thrashing, Ollama serializing, or mismatched model speeds |
| Review ignores your `focus` value | `focus` must match configured dimension names *exactly*; no match runs all of them |

### `local_review_diff` — the `focus` parameter changes meaning

- **No `swarm.review_dimensions` configured:** `focus` is a soft hint
  appended to the single review prompt, exactly as before.
- **Configured:** `focus` becomes a dimension filter. Pass a comma-separated
  list (e.g. `focus="security,performance"`) to run only the matching
  dimensions. If nothing matches (typo, or a freeform hint), all configured
  dimensions run rather than none.

Merged findings from more than one dimension are prefixed with
`[dimension]`, near-duplicate findings on the same file/line are
deduplicated (the higher-severity one survives, noted as "also flagged
by ..."), and a dimension that errors is never silently dropped — it's
named in the summary instead of fabricating a finding.

### `local_consensus`

```
local_consensus(prompt="...", models="gemma4-32k,qwen3:8b")
```

`models` is an optional ad-hoc comma-separated override — usable without
touching `routes.json`, mirroring `local_benchmark`. Without it, candidates
come from `swarm.consensus_candidates`.

Agreement is computed by **word-set containment** — shared words over the
word count of the *shorter* answer — not semantic understanding. Two answers
cluster when they overlap by ≥ 60% and share at least 3 words.

Normalizing by the shorter answer is deliberate: it makes the comparison
insensitive to verbosity, so a terse answer and a long-winded one that covers
the same ground still count as agreeing. The metric is still lexical, so
models that agree on substance while sharing no vocabulary will read as
disagreeing. Example output:

```
Agreement: 2/3 models agree.

Consensus answer (from gemma4-32k, qwen3:8b):
def is_palindrome(s): return s == s[::-1]

Divergent (deepseek-coder): def is_palindrome(s):
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]

── Latency / tokens ──
  gemma4-32k (gemma4-32k): 620ms, 40+12 tok
  qwen3:8b (qwen3:8b): 540ms, 40+15 tok
  deepseek-coder (deepseek-coder): 900ms, 40+28 tok
```

### Telemetry

Each sub-task (a review dimension, or one consensus candidate) is logged
under a synthetic tool name like `local_review_diff:security` or
`local_consensus:qwen3:8b`. These show up as their own rows in
`local_usage_stats` and `local_grading_report`.

`grading.tool_sample_rates` resolves against those names most-specific-first:
an exact entry (`"local_review_diff:security"`) wins, otherwise the base
tool's rate (`"local_review_diff"`) applies to all of its sub-tasks, and
failing both, the global `sample_rate`. So a per-tool rate covers a swarm
without you having to enumerate every dimension, and you can still single one
out when you want to.

**Known limitation — swarm-configured tools don't feed adaptive routing.**
Grades are stored under the synthetic name, while adaptive routing looks up
scores by exact tool name. A tool with `review_dimensions` configured
therefore accumulates no adaptive data, and any `adaptive.candidates` entry
for it will sit at "no data yet" indefinitely. The two features work, but not
on the same tool at the same time — which is consistent with swarm config
taking precedence over adaptive.


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

The server returns actionable error messages instead of raw tracebacks.
All errors inherit from `BackendError`, so you can catch any backend
failure generically or handle Ollama / OpenRouter errors separately.

**Ollama errors:**

| Failure                 | Error raised              | Message includes                          |
| ----------------------- | ------------------------- | ----------------------------------------- |
| Ollama not running      | `OllamaConnectionError`  | URL tried + `ollama serve` hint           |
| Model tag doesn't exist | `OllamaModelNotFound`    | Model name + `ollama pull` hint           |
| Request too slow        | `OllamaTimeout`          | Timeout duration + suggested causes       |
| HTTP 4xx/5xx            | `OllamaServerError`      | Status code + response excerpt            |
| Garbled JSON            | `OllamaMalformedResponse`| What was expected vs. what arrived        |

**OpenRouter errors:**

| Failure                 | Error raised                  | Message includes                      |
| ----------------------- | ----------------------------- | ------------------------------------- |
| Can't reach API         | `OpenRouterConnectionError`   | Check internet connection hint        |
| Bad or missing API key  | `OpenRouterAuthError`         | Key setup link                        |
| Rate limited            | `OpenRouterRateLimitError`    | Upgrade plan link                     |
| Model not available     | `OpenRouterModelNotFound`     | Model name + browse models link       |
| Request too slow        | `OpenRouterTimeout`           | Timeout duration                      |
| HTTP 4xx/5xx            | `OpenRouterServerError`       | Status code + response excerpt        |

All errors are logged to the telemetry file before propagating to the
MCP client.

## Observability

Every tool call is recorded in two places:

1. **JSON lines** (`OLLAMA_MCP_LOG`) — append-only, good for `tail -f | jq`
2. **SQLite** (`OLLAMA_MCP_DB`) — queryable, powers `local_usage_stats` and
   `ollama-mcp stats`

Each record includes:

- `tool`, `ok`, `error?`, `backend`
- `input_chars`, `output_chars`
- `total_ms` (whole tool call), `wall_ms` (HTTP), `eval_ms`
- `prompt_tokens`, `output_tokens`
- `model`, `cost` (OpenRouter only, when reported), `ts`

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
Total calls:   42
Successful:    40 (95.2%)
Prompt tokens:  180,000
Output tokens:  35,000

── ollama ──
  Calls:       30 (97% success)
  Avg latency: 1500ms
  Tokens:      120,000 in / 25,000 out
  Cost avoided (vs cloud):
    Opus: $4.6750
    Sonnet: $0.7350

── openrouter ──
  Calls:       12 (92% success)
  Avg latency: 2100ms
  Tokens:      60,000 in / 10,000 out
  Cost spent:  $0.001230

Per tool:
  local_summarize: 20 calls, avg 1200ms
  local_review_diff: 12 calls, avg 2100ms
```

Two metrics worth watching:

1. **Routing rate** — fraction of total tool calls in a session that go to
   `local_*`. If it stays below ~5%, the docstrings need work, not the infra.
2. **Quality regressions** — keep a notes file when Claude has to redo work
   the local model produced. That is the real cost of delegation.

## Examples

### textkit — pure local delegation

The `examples/textkit/` directory contains a toy Python library built entirely
through local delegation as a proof-of-concept. It includes a `TASKS.md` with
20 rows — each one a self-contained spec sized for a single Ollama call:

- **Phase 1** (5 tasks) — boilerplate via `local_draft_boilerplate`
- **Phase 2** (10 tasks) — pure-function implementations via
  `local_implement_small`
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

### hybrid-workflow — Claude + local models together

The `examples/hybrid-workflow/` directory demonstrates the real use case:
Claude Code orchestrates a feature build, delegating mechanical work to
Ollama while keeping cross-file reasoning for itself.

An 8-step guided walkthrough builds out a task tracker app:

| Step | Task                    | Handled by     | Why                                  |
|------|------------------------|----------------|--------------------------------------|
| 1    | Read & explain code    | Claude (cloud) | Needs repo context                   |
| 2    | Implement utility fn   | Ollama (local) | Self-contained spec                  |
| 3    | Draft pyproject.toml   | Ollama (local) | Mechanical boilerplate               |
| 4    | Integrate features     | Claude (cloud) | Must match existing code patterns    |
| 5    | Generate tests         | Ollama (local) | Given source, produce tests          |
| 6    | Review the diff        | Ollama (local) | Structured review                    |
| 7    | Write commit message   | Ollama (local) | One-liner from diff                  |
| 8    | Check routing & stats  | Ollama (local) | Verify the split                     |

3 cloud steps, 5 local steps. To try it: `cd examples/hybrid-workflow` and
follow `TASKS.md`.

### manual-testing — sample files for every feature

The `examples/manual-testing/` directory contains sample inputs for testing
each feature individually:

- `prompts/` — ready-made prompts for benchmarking (code review, implementation, summarization)
- `sample_module.py` — 5 pure functions for test generation
- `sample_diff.patch` — auth handler with real bugs for code review
- `sample_sensitive.patch` — diff full of secrets for testing privacy intercept
- `sample_log.txt` — realistic app log for summarization
- `sample_routes.json` — routing config ready to copy to `~/.config/ollama_mcp/`
- `test_grading.py` — 23 test cases exercising all heuristic graders (no Ollama needed)

Quick start: `ollama-mcp bench examples/manual-testing/prompts/code_review.md`

Test the grading checkers: `python3 examples/manual-testing/test_grading.py`

### Real-world result: hybrid code review

When asked to *"review this diff for bugs"* with `sample_diff.patch` (an
auth handler with intentional vulnerabilities), the local Gemma 4 model
and Claude split the work:

**Gemma 4 (local, free) found:**
- Arbitrary DB command execution — `db.execute(payload["action"])` runs
  unsanitized JWT claims
- No authorization check — any authenticated user can call `admin_action`
- NoneType crash — `verify_token` returns `None` on bad tokens, then
  `payload["action"]` raises `TypeError`
- Plaintext password comparison — `user.password == password` suggests
  no hashing
- Role escalation — `create_token(user.id)` now accepts a `role`
  parameter that callers can abuse

**Claude (cloud) supplemented with:**
- Missing input validation — `request.form["username"]` raises `KeyError`
  if the field is absent
- "changeme" default secret — `SECRET_KEY` falls back to a hardcoded
  string, making token forgery trivial

The local model caught all critical and high-severity bugs. Claude
reviewed the local output, confirmed the findings, and added
medium-severity issues it missed — a natural division of labor where the
expensive model only pays for the delta.

### Real-world result: usage stats after a manual test session

After running a handful of manual tests (summarize, implement, review,
generate tests), asking *"show me my Ollama usage stats"* returns:

| Metric       | Value                            |
| ------------ | -------------------------------- |
| Total calls  | 5                                |
| Successful   | 4 (80%)                          |
| Failed       | 1                                |
| Avg latency  | 23.7s                            |
| Tokens       | 2,725 in / 4,827 out             |
| Cost avoided | ~$0.40 vs Opus, ~$0.08 vs Sonnet |

Per-tool breakdown:

| Tool                    | Calls | Tokens (in+out) | Avg latency |
| ----------------------- | ----- | --------------- | ----------- |
| `local_review_diff`     | 2     | 991 + 1,610     | 15.5s       |
| `local_generate_tests`  | 1     | 492 + 2,469     | 23.3s       |
| `local_summarize`       | 1     | 1,040 + 113     | 49.9s       |
| `local_implement_small` | 1     | 202 + 635       | 6.0s        |

Notable: `local_implement_small` is the fastest at 6s.
`local_summarize` is the outlier at ~50s due to the long input — a
candidate for routing to a lighter model via `routes.json`. Claude
itself flagged this, suggesting *"could be worth routing that to a
lighter model like llama3.2."*

### Real-world result: multi-model benchmark

Running `local_benchmark` with the prompt *"Write a Python function that
validates email addresses"* across all installed models (RTX 4090):

| Model              | Latency | Output tokens | Eval time | Status  |
| ------------------ | ------- | ------------- | --------- | ------- |
| `llama3.2`         | 2.4s    | 179           | 0.9s      | OK      |
| `gemma4`           | 10.4s   | 750           | 6.6s      | OK      |
| `phi3:medium-128k` | 11.3s   | 191           | 3.1s      | OK      |
| `gemma4-32k`       | 12.1s   | 921           | 8.0s      | OK      |
| `gemma4-16k`       | 18.6s   | 1,595         | 14.1s     | OK      |
| `qwen3:8b`         | 83.5s   | 6,656         | 79.8s     | OK      |
| `deepseek-r1:8b`   | —       | —             | —         | TIMEOUT |

Takeaways:
- **llama3.2** is the speed winner — 2.4s, most concise (179 tokens),
  ideal for routing `local_implement_small` and `local_commit_message`
- **phi3:medium-128k** balances speed and context window — similar
  conciseness to llama3.2 but with 128K context
- **gemma4 variants** produce longer outputs (750–1,595 tokens), useful
  when thoroughness matters (reviews, test generation)
- **qwen3:8b** over-generates at 6,656 tokens — the thinking overhead
  isn't worth it for simple tasks
- **deepseek-r1:8b** timed out at 180s — reasoning models are too heavy
  for this task size

These results directly inform routing config — fast models for
mechanical tasks, thorough models for review and test generation.

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

If the tool needs to fan out to multiple models/candidates concurrently
instead of one `generate()`/`generate_json()` call, use `run_swarm()` from
`ollama_mcp/swarm.py` — see [Agent swarm](#agent-swarm) for the pattern.

## Running tests

```bash
pip install -e ".[test]"
pytest -v
```

Tests use `respx` to mock Ollama HTTP calls and run entirely offline.
CI runs automatically on pull requests via GitHub Actions.
