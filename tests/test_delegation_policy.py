"""Regression tests for the server-level delegation policy.

These guard against silently reintroducing the "cross-file reasoning" /
"repo conventions" blanket-exclusion rationalization that let an agent
skip local delegation for basically everything. Any tool docstring that
mentions a cross-file/repo-context exclusion must also carry the
missing-context caveat (paste the context in and delegate anyway).
"""

import inspect

from ollama_mcp.server import mcp
from ollama_mcp.tools import (
    local_classify_task,
    local_draft_boilerplate,
    local_generate_tests,
    local_implement_small,
    local_review_diff,
)

BROAD_EXCLUSION_PHRASES = [
    "cross-file",
    "existing repo modules",
    "existing project files",
    "deep dependencies on other project modules",
    "project conventions",
]

GENERATION_TOOLS = [
    local_draft_boilerplate,
    local_implement_small,
    local_review_diff,
    local_generate_tests,
]


def test_server_instructions_encode_delegation_policy():
    instructions = (mcp.instructions or "").lower()
    assert "invariant" in instructions
    assert "missing-context" in instructions
    assert "local_classify_task" in instructions


def test_classify_task_guideline_avoids_blanket_cross_file_exclusion():
    source = inspect.getsource(local_classify_task).lower()
    assert "invariant" in source
    assert "pasted into the" in source


def test_tool_docstrings_pair_cross_file_exclusions_with_missing_context_caveat():
    for tool in GENERATION_TOOLS:
        doc = (tool.__doc__ or "")
        doc_lower = doc.lower()
        mentions_broad_exclusion = any(
            phrase in doc_lower for phrase in BROAD_EXCLUSION_PHRASES
        )
        if mentions_broad_exclusion:
            assert "delegate" in doc_lower, (
                f"{tool.__name__} mentions a cross-file/repo-context exclusion "
                "without a missing-context caveat (expected 'delegate' nearby, "
                "e.g. 'UNLESS you paste ... — then delegate anyway')"
            )
