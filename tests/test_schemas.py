from ollama_mcp.schemas import ReviewFinding, ReviewResult


def test_review_result_format_with_findings():
    result = ReviewResult(
        findings=[
            ReviewFinding(
                severity="HIGH", category="BUG",
                message="off-by-one error", file="foo.py", line=10,
            ),
            ReviewFinding(
                severity="LOW", category="STYLE",
                message="trailing whitespace", file="bar.py", line=None,
            ),
        ],
        summary="2 findings (1 high, 0 medium, 1 low)",
    )
    text = result.format()
    assert "[HIGH] BUG: off-by-one error (foo.py:10)" in text
    assert "[LOW] STYLE: trailing whitespace (bar.py)" in text
    assert "2 findings" in text


def test_review_result_format_no_findings():
    result = ReviewResult(findings=[], summary="No issues found.")
    text = result.format()
    assert "No issues found." in text


def test_review_finding_optional_location():
    f = ReviewFinding(severity="MEDIUM", category="SECURITY", message="test")
    assert f.file is None
    assert f.line is None
