from ollama_mcp.schemas import ReviewFinding, ReviewResult, TaskClassification


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


def test_task_classification_format():
    tc = TaskClassification(
        task_type="review",
        risk="medium",
        recommended_tool="local_review_diff",
        recommended_model="deepseek-coder",
        should_use_local=True,
        reasoning="Code review is well-suited for local models",
    )
    text = tc.format()
    assert "review" in text
    assert "medium" in text
    assert "local_review_diff" in text
    assert "deepseek-coder" in text
    assert "yes" in text


def test_task_classification_not_local():
    tc = TaskClassification(
        task_type="other",
        risk="high",
        recommended_tool="none",
        recommended_model="default",
        should_use_local=False,
        reasoning="Needs cross-file reasoning",
    )
    text = tc.format()
    assert "no" in text
    assert "high" in text
