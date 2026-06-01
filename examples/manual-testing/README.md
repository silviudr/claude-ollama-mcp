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

## 4. Privacy Intercept

> Review this diff: [paste contents of sample_sensitive.patch]

Then check the telemetry log:
```bash
tail -1 ~/.cache/ollama_mcp.jsonl | python3 -m json.tool
```

Look for `"event": "privacy_intercept"` in the log.

## 5. Routing Config

```bash
# Create a routing config
mkdir -p ~/.config/ollama_mcp
cp examples/manual-testing/sample_routes.json ~/.config/ollama_mcp/routes.json

# Then in Claude Code:
# > Show me the routing config
# > Review this diff: [paste sample_diff.patch]  — should use the routed model
```

## 6. SQLite Inspection

After running a few tools, inspect the database:

```bash
sqlite3 ~/.cache/ollama_mcp.db "SELECT tool, count(*), sum(prompt_tokens), sum(output_tokens) FROM calls GROUP BY tool"
sqlite3 ~/.cache/ollama_mcp.db "SELECT * FROM calls ORDER BY ts DESC LIMIT 5"
```
