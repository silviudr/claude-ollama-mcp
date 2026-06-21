#!/usr/bin/env python3
"""Manual grading test — run heuristic checkers on known-good and known-bad
outputs to see the grading framework in action.

No Ollama or OpenRouter needed — this exercises the heuristic checkers only.

Usage:
    python3 examples/manual-testing/test_grading.py
"""

import json
import sys
from pathlib import Path

# Add project root to path so we can import without installing
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ollama_mcp.grading.heuristics import run_heuristics

# ── Test cases ──────────────────────────────────────────────────────

CASES = [
    # ── local_implement_small ──
    {
        "tool": "local_implement_small",
        "label": "GOOD — valid Python function",
        "input": "write a function that adds two numbers",
        "output": "def add(a, b):\n    return a + b\n",
        "expect_all_pass": True,
    },
    {
        "tool": "local_implement_small",
        "label": "BAD — syntax error",
        "input": "write a function that adds two numbers",
        "output": "def add(a, b\n    return a + b\n",
        "expect_all_pass": False,
    },
    {
        "tool": "local_implement_small",
        "label": "BAD — hallucinated heavy import",
        "input": "write a function that sorts a list",
        "output": "import torch\nimport numpy as np\n\ndef sort_list(lst):\n    return sorted(lst)\n",
        "expect_all_pass": False,
    },
    {
        "tool": "local_implement_small",
        "label": "BAD — empty output",
        "input": "write a function",
        "output": "",
        "expect_all_pass": False,
    },
    {
        "tool": "local_implement_small",
        "label": "BAD — truncated output",
        "input": "write a parser",
        "output": "def parse(data):\n    result = {\n        'name': data['name'],\n        'age': data['age'],",
        "expect_all_pass": False,
    },
    {
        "tool": "local_implement_small",
        "label": "BAD — degenerate repetition",
        "input": "write a function",
        "output": "pass\n" * 30,
        "expect_all_pass": False,
    },

    # ── local_commit_message ──
    {
        "tool": "local_commit_message",
        "label": "GOOD — conventional commit",
        "input": "--- a/foo.py\n+++ b/foo.py\n-old\n+new",
        "output": "fix: resolve null pointer in parser",
        "expect_all_pass": True,
    },
    {
        "tool": "local_commit_message",
        "label": "BAD — not conventional format",
        "input": "--- a/foo.py\n+++ b/foo.py",
        "output": "Updated the file to fix the thing that was broken",
        "expect_all_pass": False,
    },

    # ── local_generate_tests ──
    {
        "tool": "local_generate_tests",
        "label": "GOOD — valid pytest file",
        "input": "def add(a, b): return a + b",
        "output": (
            "import pytest\n\n"
            "from module import add\n\n"
            "def test_add_positive():\n"
            "    assert add(1, 2) == 3\n\n"
            "def test_add_negative():\n"
            "    assert add(-1, -2) == -3\n\n"
            "def test_add_zero():\n"
            "    assert add(0, 0) == 0\n"
        ),
        "expect_all_pass": True,
    },
    {
        "tool": "local_generate_tests",
        "label": "BAD — no test functions",
        "input": "def add(a, b): return a + b",
        "output": "# Here are some tests you could write:\n# test adding positive numbers\n# test adding negative numbers\n",
        "expect_all_pass": False,
    },
    {
        "tool": "local_generate_tests",
        "label": "BAD — no pytest import",
        "input": "def add(a, b): return a + b",
        "output": "def test_add():\n    assert add(1, 2) == 3\n",
        "expect_all_pass": False,
    },

    # ── local_review_diff ──
    {
        "tool": "local_review_diff",
        "label": "GOOD — valid structured review",
        "input": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
        "output": json.dumps({
            "findings": [
                {"severity": "MEDIUM", "category": "STYLE",
                 "message": "consider a more descriptive variable name",
                 "file": "foo.py", "line": 1}
            ],
            "summary": "1 minor style issue found.",
        }),
        "expect_all_pass": True,
    },
    {
        "tool": "local_review_diff",
        "label": "BAD — not JSON",
        "input": "--- a/foo.py\n+++ b/foo.py",
        "output": "The code looks fine to me. I don't see any issues.",
        "expect_all_pass": False,
    },
    {
        "tool": "local_review_diff",
        "label": "BAD — references file not in diff",
        "input": "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new",
        "output": json.dumps({
            "findings": [
                {"severity": "HIGH", "category": "BUG",
                 "message": "null pointer", "file": "bar.py", "line": 10}
            ],
            "summary": "1 bug found.",
        }),
        "expect_all_pass": False,
    },

    # ── local_generate_tests (collection check) ──
    {
        "tool": "local_generate_tests",
        "label": "GOOD — collectable tests with source",
        "input": "def add(a, b):\n    return a + b\n",
        "output": (
            "import pytest\n\n"
            "from module_under_test import add\n\n"
            "def test_add_positive():\n"
            "    assert add(1, 2) == 3\n\n"
            "def test_add_zero():\n"
            "    assert add(0, 0) == 0\n"
        ),
        "expect_all_pass": True,
    },
    {
        "tool": "local_generate_tests",
        "label": "BAD — tests import nonexistent package",
        "input": "x = 1",
        "output": (
            "from nonexistent_deep_package.sub import magic\n\n"
            "def test_magic():\n"
            "    assert magic() == 42\n"
        ),
        "expect_all_pass": False,
    },

    # ── local_draft_boilerplate (format validation) ──
    {
        "tool": "local_draft_boilerplate",
        "label": "GOOD — valid Dockerfile",
        "input": "Dockerfile for Python 3.11 service",
        "output": "FROM python:3.11\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]\n",
        "expect_all_pass": True,
    },
    {
        "tool": "local_draft_boilerplate",
        "label": "BAD — Dockerfile missing FROM",
        "input": "Dockerfile for Node",
        "output": "RUN apt-get update\nCOPY . /app\n",
        "expect_all_pass": False,
    },
    {
        "tool": "local_draft_boilerplate",
        "label": "GOOD — valid Makefile",
        "input": "Makefile with build/test/clean targets",
        "output": "build:\n\tgo build -o app\n\ntest:\n\tgo test ./...\n\nclean:\n\trm -f app\n",
        "expect_all_pass": True,
    },
    {
        "tool": "local_draft_boilerplate",
        "label": "GOOD — valid .gitignore",
        "input": "gitignore for Python",
        "output": "*.pyc\n__pycache__/\n.env\nvenv/\n*.egg-info/\n",
        "expect_all_pass": True,
    },
    {
        "tool": "local_draft_boilerplate",
        "label": "GOOD — valid JSON config",
        "input": "json config for prettier",
        "output": '{"semi": false, "singleQuote": true, "tabWidth": 2}\n',
        "expect_all_pass": True,
    },

    # ── local_summarize ──
    {
        "tool": "local_summarize",
        "label": "GOOD — concise summary",
        "input": "A very long document about software engineering best practices...",
        "output": "The document covers key software engineering practices including testing, code review, CI/CD, and documentation standards.",
        "expect_all_pass": True,
    },
    {
        "tool": "local_summarize",
        "label": "BAD — truncated with ellipsis",
        "input": "A long document",
        "output": "The document discusses several important topics including...",
        "expect_all_pass": False,
    },
]


# ── Runner ──────────────────────────────────────────────────────────

def main():
    total = 0
    passed = 0
    failed_cases = []

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            Manual Grading Heuristics Test                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    for case in CASES:
        total += 1
        tool = case["tool"]
        label = case["label"]
        results = run_heuristics(tool, case["input"], case["output"])

        all_passed = all(r["passed"] for r in results)
        expectation_met = all_passed == case["expect_all_pass"]

        status = "✓" if expectation_met else "✗"
        if expectation_met:
            passed += 1
        else:
            failed_cases.append(case["label"])

        print(f"  {status}  {tool} — {label}")

        for r in results:
            icon = "  ✓" if r["passed"] else "  ✗"
            detail = ""
            if not r["passed"] and r["details"].get("reason"):
                detail = f"  ({r['details']['reason']})"
            elif r["passed"] and r["score"] < 1.0:
                detail = f"  (score: {r['score']:.1f})"
            print(f"      {icon} {r['checker']}: score={r['score']:.1f}{detail}")

        print()

    # Summary
    print("─" * 62)
    rate = passed / total * 100 if total else 0
    print(f"  {passed}/{total} cases matched expectations ({rate:.0f}%)")
    if failed_cases:
        print(f"\n  Unexpected results:")
        for f in failed_cases:
            print(f"    - {f}")
    print()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
