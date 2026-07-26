"""Pydantic models for structured tool outputs."""

from __future__ import annotations

import difflib
from typing import Literal

from pydantic import BaseModel, Field

_SEVERITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


class ReviewFinding(BaseModel):
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    category: str
    message: str
    file: str | None = None
    line: int | None = None


class ReviewResult(BaseModel):
    findings: list[ReviewFinding] = []
    summary: str

    def format(self) -> str:
        if not self.findings:
            return f"No issues found.\n{self.summary}"

        lines = []
        for f in self.findings:
            loc = ""
            if f.file:
                loc = f" ({f.file}:{f.line})" if f.line else f" ({f.file})"
            lines.append(f"[{f.severity}] {f.category}: {f.message}{loc}")
        lines.append("")
        lines.append(self.summary)
        return "\n".join(lines)


def _truncate(text: str, limit: int = 200) -> str:
    text = (text or "unknown").strip().replace("\n", " ")
    return text[:limit] + "..." if len(text) > limit else text


def merge_review_results(
    labeled: list[tuple[str, ReviewResult | None, bool, str | None]],
) -> ReviewResult:
    """Merge per-dimension review results into one ReviewResult.

    labeled: [(dimension_name, parsed_result_or_None, ok, raw_or_error), ...]
    `ok` is whether the dimension's call succeeded — a dimension can succeed
    but still have parsed=None (the model responded, just not with valid
    JSON); that's reported as "returned unparsed output", not as a failure,
    so a real (if malformed) answer is never mislabeled as an infra error.
    A dimension with parsed=None contributes no findings either way (never
    fabricates one). Always returns a valid ReviewResult.
    """
    successes = [(name, r) for name, r, _ok, _detail in labeled if r is not None]
    unparsed = [(name, detail) for name, r, ok, detail in labeled if r is None and ok]
    failures = [(name, detail) for name, r, ok, detail in labeled if r is None and not ok]
    multi = len(successes) > 1

    groups: dict[tuple, list[list]] = {}
    for name, result in successes:
        for f in result.findings:
            groups.setdefault((f.file, f.line), []).append([name, f])

    merged: list[ReviewFinding] = []
    for group in groups.values():
        kept: list[list] = []  # [[primary_name, other_names, finding], ...]
        for name, f in group:
            match = next(
                (
                    entry for entry in kept
                    if difflib.SequenceMatcher(None, entry[2].message, f.message).ratio() >= 0.8
                ),
                None,
            )
            if match is None:
                kept.append([name, [], f])
            else:
                if _SEVERITY_RANK[f.severity] < _SEVERITY_RANK[match[2].severity]:
                    match[1].append(match[0])
                    match[0] = name
                    match[2] = f
                else:
                    match[1].append(name)

        for primary, others, f in kept:
            message = f.message
            if multi:
                message = f"[{primary}] {message}"
                if others:
                    message += f" (also flagged by {', '.join(others)})"
            merged.append(ReviewFinding(
                severity=f.severity, category=f.category,
                message=message, file=f.file, line=f.line,
            ))

    merged.sort(key=lambda f: _SEVERITY_RANK.get(f.severity, 3))

    summary_parts = []
    if successes:
        dims = ", ".join(name for name, _ in successes)
        if merged:
            counts = {}
            for f in merged:
                counts[f.severity] = counts.get(f.severity, 0) + 1
            count_str = ", ".join(
                f"{counts[s]} {s.lower()}" for s in ("HIGH", "MEDIUM", "LOW") if s in counts
            )
            summary_parts.append(f"{len(merged)} findings across {dims} ({count_str}).")
        else:
            summary_parts.append(f"No issues found across {dims}.")
    for name, detail in unparsed:
        summary_parts.append(f"{name} dimension returned unparsed output: {_truncate(detail)}")
    for name, detail in failures:
        summary_parts.append(f"{name} dimension failed: {_truncate(detail)}")
    if not summary_parts:
        summary_parts.append("No dimensions produced results.")

    return ReviewResult(findings=merged, summary=" ".join(summary_parts))


class TaskClassification(BaseModel):
    task_type: Literal[
        "summarize", "code", "review", "test", "explain", "boilerplate", "other"
    ]
    risk: Literal["low", "medium", "high"]
    recommended_tool: str = Field(
        description="The local_* tool best suited for this task"
    )
    recommended_model: str = Field(
        description="Model name to use, or 'default' to use the configured default"
    )
    should_use_local: bool = Field(
        description="Whether this task is suitable for local model processing"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification"
    )

    def format(self) -> str:
        lines = [
            f"Task type:         {self.task_type}",
            f"Risk:              {self.risk}",
            f"Recommended tool:  {self.recommended_tool}",
            f"Recommended model: {self.recommended_model}",
            f"Use local:         {'yes' if self.should_use_local else 'no'}",
            f"Reasoning:         {self.reasoning}",
        ]
        return "\n".join(lines)


class DataInsight(BaseModel):
    category: str = Field(description="E.g. distribution, outlier, correlation, quality")
    column: str | None = None
    description: str
    severity: Literal["HIGH", "MEDIUM", "LOW"] = "MEDIUM"


class AnalysisResult(BaseModel):
    summary: str
    insights: list[DataInsight] = []
    row_count: int = 0
    col_count: int = 0

    def format(self) -> str:
        lines = [
            f"Rows: {self.row_count}, Columns: {self.col_count}",
            "",
            self.summary,
        ]
        if self.insights:
            lines.append("")
            lines.append("Insights:")
            for i in self.insights:
                col = f" ({i.column})" if i.column else ""
                lines.append(f"  [{i.severity}] {i.category}{col}: {i.description}")
        return "\n".join(lines)
