"""Pydantic models for structured tool outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


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
