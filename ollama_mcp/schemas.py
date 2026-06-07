"""Pydantic models for structured tool outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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
