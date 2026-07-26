from ollama_mcp.schemas import (
    ReviewFinding,
    ReviewResult,
    TaskClassification,
    merge_review_results,
)


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


# --- merge_review_results ---


def test_merge_concatenates_findings_across_dimensions():
    security = ReviewResult(
        findings=[ReviewFinding(severity="HIGH", category="SECURITY", message="SQL injection", file="a.py", line=1)],
        summary="1 finding",
    )
    perf = ReviewResult(
        findings=[ReviewFinding(severity="LOW", category="PERFORMANCE", message="N+1 query", file="b.py", line=2)],
        summary="1 finding",
    )
    merged = merge_review_results([
        ("security", security, True, None),
        ("performance", perf, True, None),
    ])
    assert len(merged.findings) == 2
    assert any("[security] SQL injection" in f.message for f in merged.findings)
    assert any("[performance] N+1 query" in f.message for f in merged.findings)


def test_merge_single_dimension_no_prefix():
    security = ReviewResult(
        findings=[ReviewFinding(severity="HIGH", category="SECURITY", message="SQL injection", file="a.py", line=1)],
        summary="1 finding",
    )
    merged = merge_review_results([("security", security, True, None)])
    assert merged.findings[0].message == "SQL injection"


def test_merge_dedupes_near_duplicate_findings():
    a = ReviewResult(
        findings=[ReviewFinding(
            severity="MEDIUM", category="BUG",
            message="off-by-one error in loop bound", file="a.py", line=10,
        )],
        summary="1 finding",
    )
    b = ReviewResult(
        findings=[ReviewFinding(
            severity="HIGH", category="BUG",
            message="off-by-one error in the loop bound", file="a.py", line=10,
        )],
        summary="1 finding",
    )
    merged = merge_review_results([("dim1", a, True, None), ("dim2", b, True, None)])
    assert len(merged.findings) == 1
    assert merged.findings[0].severity == "HIGH"
    assert "[dim2]" in merged.findings[0].message
    assert "also flagged by dim1" in merged.findings[0].message


def test_merge_sorts_by_severity():
    a = ReviewResult(
        findings=[
            ReviewFinding(severity="LOW", category="STYLE", message="low finding"),
            ReviewFinding(severity="HIGH", category="BUG", message="high finding"),
            ReviewFinding(severity="MEDIUM", category="BUG", message="medium finding"),
        ],
        summary="3 findings",
    )
    b = ReviewResult(findings=[], summary="clean")
    merged = merge_review_results([("dim1", a, True, None), ("dim2", b, True, None)])
    severities = [f.severity for f in merged.findings]
    assert severities == ["HIGH", "MEDIUM", "LOW"]


def test_merge_failed_dimension_contributes_no_findings_but_noted():
    clean = ReviewResult(findings=[], summary="clean")
    merged = merge_review_results([
        ("security", clean, True, None),
        ("performance", None, False, "connection refused"),
    ])
    assert merged.findings == []
    assert "performance dimension failed: connection refused" in merged.summary


def test_merge_all_failed_returns_valid_result():
    merged = merge_review_results([
        ("security", None, False, "timeout"),
        ("performance", None, False, "500 error"),
    ])
    assert merged.findings == []
    assert "security dimension failed: timeout" in merged.summary
    assert "performance dimension failed: 500 error" in merged.summary


def test_merge_unparsed_dimension_is_not_a_failure():
    clean = ReviewResult(findings=[], summary="clean")
    merged = merge_review_results([
        ("security", clean, True, None),
        ("performance", None, True, "some prose the model returned instead of JSON"),
    ])
    assert merged.findings == []
    assert "performance dimension returned unparsed output" in merged.summary
    assert "some prose the model returned instead of JSON" in merged.summary
    assert "performance dimension failed" not in merged.summary


def test_merge_all_clean_summary():
    clean_a = ReviewResult(findings=[], summary="clean")
    clean_b = ReviewResult(findings=[], summary="clean")
    merged = merge_review_results([("security", clean_a, True, None), ("performance", clean_b, True, None)])
    assert merged.findings == []
    assert "No issues found across security, performance" in merged.summary
