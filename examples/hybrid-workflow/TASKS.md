# Hybrid Workflow — Claude + Local Models

This example walks through a realistic feature-building workflow where
Claude Code orchestrates and delegates to local Ollama models when the
work is mechanical, keeping the expensive cloud model for tasks that
need repo context and cross-file reasoning.

## The scenario

You have a small task tracker (`src/app.py`) with several TODOs.
Your job: implement the missing features end-to-end.

## Instructions

Open this directory in Claude Code and paste each step below. Watch
which tools Claude calls — local_* tools go to Ollama, everything else
stays with Claude.

---

### Step 1 — Understand the codebase (Claude)

> Read src/app.py and explain the current architecture. What's there
> and what's missing based on the TODOs?

**Why Claude:** needs to read the file, understand the structure, reason
about what's missing. Cross-file reasoning.

---

### Step 2 — Implement a utility function (Local)

> Write me a self-contained function called `match_any_tag` that takes
> a list of item tags and a list of filter tags, and returns True if
> any filter tag appears in the item tags. Handle empty filter list by
> returning True (no filter = match all).

**Why local:** self-contained function with a clear spec, no repo context
needed. Should route to `local_implement_small`.

---

### Step 3 — Generate boilerplate (Local)

> Draft a pyproject.toml for this project. Name: task-tracker,
> version 0.1.0, Python >=3.11, no dependencies, with pytest as a
> test dependency.

**Why local:** mechanical scaffold from a spec. Should route to
`local_draft_boilerplate`.

---

### Step 4 — Integrate into the codebase (Claude)

> Now add these methods to the TaskStore class in src/app.py:
> - filter_by_tag(tag) — returns tasks that have the given tag
> - filter_by_status(complete: bool) — returns complete or incomplete tasks
> - search(query) — case-insensitive substring match on title
> - stats() — returns a dict with total, completed, pending, and
>   completion_rate (as a float 0-1)
>
> Use the existing patterns in the class. Make sure the return types
> are consistent with list_all().

**Why Claude:** needs to read the existing code, match patterns, understand
types, and modify the file in place. Cross-file reasoning.

---

### Step 5 — Generate tests (Local)

> Generate pytest tests for src/app.py

**Why local:** given the source, generate tests for all public methods.
Should route to `local_generate_tests`.

---

### Step 6 — Review the changes (Local)

> Review the diff of everything we changed

**Why local:** review a diff for bugs and issues. Should route to
`local_review_diff`.

---

### Step 7 — Commit message (Local)

> Write a commit message for these changes

**Why local:** generate a conventional-commit subject from the diff.
Should route to `local_commit_message`.

---

### Step 8 — Check routing and stats

> Show me the routing config

> Show me my Ollama usage stats

**Why:** verify which models handled which tools, see the cost avoidance.

---

## Expected routing breakdown

| Step | Task                    | Handled by     | Tool                      |
|------|------------------------|----------------|---------------------------|
| 1    | Read & explain code    | Claude (cloud) | —                         |
| 2    | Implement utility fn   | Ollama (local) | `local_implement_small`   |
| 3    | Draft pyproject.toml   | Ollama (local) | `local_draft_boilerplate` |
| 4    | Integrate features     | Claude (cloud) | —                         |
| 5    | Generate tests         | Ollama (local) | `local_generate_tests`    |
| 6    | Review diff            | Ollama (local) | `local_review_diff`       |
| 7    | Commit message         | Ollama (local) | `local_commit_message`    |
| 8    | Check routing/stats    | Ollama (local) | `local_show_routes`, `local_usage_stats` |

**Result:** 3 steps with Claude (understanding + integration), 5 steps
with local models (mechanical work). The local calls are free.
