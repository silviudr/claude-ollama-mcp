"""LLM-based semantic grading — runs against OpenRouter or a local model."""

from __future__ import annotations

import json
import re
import time

from pydantic import BaseModel, Field

from ..backends.base import Backend

_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)

# Rubric prompts keyed by tool category
_RUBRICS: dict[str, str] = {
    "code": (
        "Score the code on these dimensions (0-5 each):\n"
        "1. correctness — does it implement the spec accurately?\n"
        "2. completeness — are edge cases handled?\n"
        "3. format — clean code, no markdown fences, no prose?\n"
        "4. conciseness — no unnecessary bloat?\n"
    ),
    "test": (
        "Score the test code on these dimensions (0-5 each):\n"
        "1. correctness — do tests actually test the right behaviour?\n"
        "2. completeness — are normal, edge, and error cases covered?\n"
        "3. format — valid pytest code, proper naming, no prose?\n"
        "4. conciseness — no redundant tests?\n"
    ),
    "review": (
        "Score the code review on these dimensions (0-5 each):\n"
        "1. correctness — are the findings real issues in the diff?\n"
        "2. completeness — are important issues covered?\n"
        "3. format — proper structured output with severity/category?\n"
        "4. conciseness — no false positives or noise?\n"
    ),
    "summary": (
        "Score the summary on these dimensions (0-5 each):\n"
        "1. correctness — does it accurately reflect the source?\n"
        "2. completeness — are the key points included?\n"
        "3. format — clean prose, no hallucinated details?\n"
        "4. conciseness — appropriately brief?\n"
    ),
    "commit": (
        "Score the commit message on these dimensions (0-5 each):\n"
        "1. correctness — does it describe what the diff actually changes?\n"
        "2. completeness — does it capture the intent?\n"
        "3. format — conventional-commit format (type: description)?\n"
        "4. conciseness — single subject line, no fluff?\n"
    ),
}

_TOOL_TO_RUBRIC: dict[str, str] = {
    "local_implement_small": "code",
    "local_draft_boilerplate": "code",
    "local_generate_tests": "test",
    "local_review_diff": "review",
    "local_summarize": "summary",
    "local_commit_message": "commit",
    "local_classify_task": "summary",
    "local_analyze_data": "summary",
}


class SemanticGrade(BaseModel):
    correctness: int = Field(ge=0, le=5)
    completeness: int = Field(ge=0, le=5)
    format: int = Field(ge=0, le=5)
    conciseness: int = Field(ge=0, le=5)
    overall: float = Field(ge=0.0, le=1.0)
    issues: list[str] = []


async def grade_output(
    backend: Backend,
    tool_name: str,
    input_text: str,
    output_text: str,
    model: str | None = None,
) -> dict:
    """Send a grading prompt to the backend and return a grade result dict.

    Returns the standard grade dict:
      checker, passed, score, details, grader_model, grader_backend, grader_ms
    """
    rubric_key = _TOOL_TO_RUBRIC.get(tool_name, "summary")
    rubric = _RUBRICS[rubric_key]

    truncated_input = input_text[:2000] + ("…" if len(input_text) > 2000 else "")
    truncated_output = output_text[:3000] + ("…" if len(output_text) > 3000 else "")

    prompt = (
        f"You are grading the output of a '{tool_name}' tool.\n\n"
        f"## Input given to the tool\n{truncated_input}\n\n"
        f"## Output produced\n{truncated_output}\n\n"
        f"## Grading rubric\n{rubric}\n"
        "Return JSON matching this exact structure:\n"
        '{"correctness": N, "completeness": N, "format": N, '
        '"conciseness": N, "overall": 0.0-1.0, "issues": ["..."]}\n\n'
        "overall is a normalised 0-1 score. Be honest — a mediocre output "
        "should score 0.4-0.6, not 0.9."
    )

    t0 = time.perf_counter()
    try:
        raw, meta = await backend.generate(
            prompt,
            system="You are a strict output quality grader. Return only JSON.",
            model=model,
        )
        grader_ms = int((time.perf_counter() - t0) * 1000)

        parsed = _parse_grade(raw)
        if parsed:
            return {
                "checker": "llm_judge",
                "passed": parsed.overall >= 0.5,
                "score": parsed.overall,
                "details": parsed.model_dump(),
                "grader_model": meta.get("model", model),
                "grader_backend": meta.get("backend", backend.name),
                "grader_ms": grader_ms,
            }

        return {
            "checker": "llm_judge",
            "passed": None,
            "score": None,
            "details": {"raw": raw[:500], "parse_error": "could not parse grader output"},
            "grader_model": meta.get("model", model),
            "grader_backend": meta.get("backend", backend.name),
            "grader_ms": grader_ms,
        }
    except Exception as e:
        grader_ms = int((time.perf_counter() - t0) * 1000)
        return {
            "checker": "llm_judge",
            "passed": None,
            "score": None,
            "details": {"error": repr(e)},
            "grader_model": model,
            "grader_backend": backend.name,
            "grader_ms": grader_ms,
        }


def _parse_grade(raw: str) -> SemanticGrade | None:
    cleaned = raw.strip()
    m = _FENCE_RE.search(cleaned)
    if m:
        cleaned = m.group(1).strip()
    try:
        return SemanticGrade.model_validate_json(cleaned)
    except Exception:
        return None
