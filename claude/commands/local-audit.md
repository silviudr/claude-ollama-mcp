---
description: Full-coverage security audit of this codebase using ONLY local Ollama models
argument-hint: "[path to audit, defaults to whole repo]"
allowed-tools: Bash, Read, Glob, Grep, mcp__ollama-local__local_show_routes
---

# Local-only security audit — every file

Scope: **$1** (empty = whole repository).

The audit is performed by `~/.claude/scripts/local_audit.py`, which sweeps every
source file through local models directly. It runs **outside your context
window**, so coverage does not degrade on large repos and cannot stop early.

Do not audit the code yourself and do not add findings of your own to the
report. Your job is to run the sweep, then explain the result.

## Step 1 — Confirm scope and cost

```bash
python3 ~/.claude/scripts/local_audit.py $1 --dry-run
```

This prints the local-only verification, file count, chunk count, and a time
estimate. Show the user the estimate. If it exceeds ~45 minutes, tell them the
number and suggest narrowing to a subdirectory — but if they already asked for
the whole project, just proceed.

## Step 2 — Run the sweep

```bash
python3 ~/.claude/scripts/local_audit.py $1 --resume
```

Run it with `run_in_background: true` and a generous timeout, then poll its
output rather than blocking. It prints a progress line every 10 calls with a
live ETA.

Notes:

- `--resume` makes an interrupted run continue from its checkpoint. It is safe
  to pass every time; the checkpoint is removed after a clean run.
- The script refuses to start if any non-Ollama backend is configured. If it
  exits with `FATAL`, report that verbatim and stop — do not work around it.
- Add `--include-tests` only if the user asks; test files are skipped by default.
- If many chunks land in NOT AUDITED, say so plainly. That is a coverage gap,
  not a clean result.

## Step 3 — Report back

The script writes `SECURITY-AUDIT.md` (report) and `.local-audit-findings.json`
(raw, per-dimension). Read the report and give the user:

1. Counts by severity, and how many were **confirmed by both** dimensions.
2. The most serious findings, with file and line. Prefer the confirmed ones —
   agreement between two different models is the strongest signal available.
3. Coverage: files audited, and anything in NOT AUDITED with the reason.
4. Explicit confirmation that no cloud model was used.

Be honest about reliability. These are unverified ~30B model outputs. Before
presenting any HIGH finding as real, open the cited file and line and check it
actually says what the model claims. If it does not, say so — a hallucinated
line number is worth reporting as a false positive rather than passing along.

Do not offer to fix anything unless the user asks.
