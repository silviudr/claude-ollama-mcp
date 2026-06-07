# Manual Testing Guide

Sample files for testing every feature of the ollama-mcp server.
Run these from the project root after installing with `pip install -e .`

## Prerequisites

```bash
# Confirm Ollama is running and has at least one model
ollama list
ollama serve  # if not running
```

## 1. CLI — Benchmarking

```bash
# Benchmark with a prompt file
ollama-mcp bench examples/manual-testing/prompts/code_review.md

# Benchmark with a system prompt
ollama-mcp bench examples/manual-testing/prompts/implement.md -s "Output code only"

# Benchmark specific models (replace with your installed models)
ollama-mcp bench "Explain closures in Python" -m gemma4-32k
```

## 2. CLI — Stats

```bash
ollama-mcp stats
```

## 3. Through Claude Code

Start a Claude Code session in any project and try each of these:

### Summarize
> Summarize this file: [paste contents of sample_log.txt]

### Implement
> Write me a function that validates email addresses

### Code Review
> Review this diff for bugs: [paste contents of sample_diff.patch]

### Generate Tests
> Generate pytest tests for this: [paste contents of sample_module.py]

### Classify Task
> Which tool should handle this: write a Dockerfile for a Flask app

### List Models
> What local models do I have?

### Benchmark
> Compare my local models on this: [paste contents of prompts/implement.md]

### Routes
> Show me the routing config

### Usage Stats
> Show me my Ollama usage stats

## 4. Data Analyzer

The `local_analyze_data` tool profiles a CSV and decides whether to
analyze it locally or hand off to Claude. The decision is based on a
complexity score computed from six dimensions:

| Signal                  | What it detects                                  | Score weight    |
| ----------------------- | ------------------------------------------------ | --------------- |
| Structural complexity   | Column count > 20 (`OLLAMA_MCP_MAX_COLS`)        | proportional    |
| Data cardinality        | High ratio of unique values across columns       | avg ratio       |
| Nested JSON             | JSON strings embedded in column values           | 0.4 per column  |
| Multiple datetimes      | 2+ datetime columns needing alignment            | 0.3             |
| Free-text columns       | Columns averaging > 60 chars / 8 words           | 0.5 per column  |
| Foreign keys            | 2+ columns ending in `_id`, `_uuid`, `_key`, etc.| 0.3             |
| Mixed-type columns      | Columns with both numeric and non-numeric values | 0.3             |

If the weighted score exceeds `OLLAMA_MCP_COMPLEXITY_THRESHOLD`
(default 0.7), the tool returns a **HANDOFF** with the dataset profile
and sample rows so Claude can take over.

### Simple dataset (analyzed locally)

> Analyze examples/manual-testing/sample_simple.csv

### Complex dataset (triggers handoff)

> Analyze examples/manual-testing/sample_complex.csv

### With a specific question

> What's the average salary by department in sample_simple.csv?

### Force Claude to handle it

> Analyze sample_simple.csv but use Claude for this one

Claude passes `force_handoff=true`, bypassing the triage.

## 5. Privacy Intercept

> Review this diff: [paste contents of sample_sensitive.patch]

Then check the telemetry log:
```bash
tail -1 ~/.cache/ollama_mcp.jsonl | python3 -m json.tool
```

Look for `"event": "privacy_intercept"` in the log.

## 6. Routing Config

```bash
# Create a routing config
mkdir -p ~/.config/ollama_mcp
cp examples/manual-testing/sample_routes.json ~/.config/ollama_mcp/routes.json

# Then in Claude Code:
# > Show me the routing config
# > Review this diff: [paste sample_diff.patch]  — should use the routed model
```

## 7. SQLite Inspection

After running a few tools, inspect the database:

```bash
sqlite3 ~/.cache/ollama_mcp.db "SELECT tool, count(*), sum(prompt_tokens), sum(output_tokens) FROM calls GROUP BY tool"
sqlite3 ~/.cache/ollama_mcp.db "SELECT * FROM calls ORDER BY ts DESC LIMIT 5"
```
