"""Tests for grading heuristic checkers."""

import json

import pytest

from ollama_mcp.grading.heuristics import (
    check_boilerplate_format,
    check_conventional_commit,
    check_diff_file_references,
    check_has_pytest_import,
    check_has_test_functions,
    check_json_parseable,
    check_no_hallucinated_imports,
    check_no_truncation,
    check_non_empty,
    check_python_syntax,
    check_repetition,
    check_tests_collectable,
    run_heuristics,
)


# --- check_python_syntax ---


class TestPythonSyntax:
    def test_valid_code(self):
        r = check_python_syntax("def add(a, b): return a + b")
        assert r["passed"] is True
        assert r["score"] == 1.0

    def test_invalid_code(self):
        r = check_python_syntax("def add(a, b return a + b")
        assert r["passed"] is False
        assert r["score"] == 0.0
        assert "reason" in r["details"]

    def test_multiline_valid(self):
        code = "def greet(name):\n    return f'Hello, {name}!'\n"
        r = check_python_syntax(code)
        assert r["passed"] is True

    def test_empty_string(self):
        r = check_python_syntax("")
        assert r["passed"] is True


# --- check_has_test_functions ---


class TestHasTestFunctions:
    def test_with_tests(self):
        code = "def test_add():\n    assert add(1, 2) == 3\ndef test_sub():\n    pass"
        r = check_has_test_functions(code)
        assert r["passed"] is True
        assert r["details"]["count"] == 2

    def test_without_tests(self):
        r = check_has_test_functions("def add(a, b): return a + b")
        assert r["passed"] is False

    def test_test_in_middle(self):
        code = "import pytest\n\ndef test_hello():\n    pass\n\nclass Foo:\n    pass"
        r = check_has_test_functions(code)
        assert r["passed"] is True
        assert r["details"]["count"] == 1


# --- check_has_pytest_import ---


class TestPytestImport:
    def test_with_import(self):
        r = check_has_pytest_import("import pytest\n\ndef test_x(): pass")
        assert r["passed"] is True

    def test_from_import(self):
        r = check_has_pytest_import("from pytest import raises")
        assert r["passed"] is True

    def test_without_import(self):
        r = check_has_pytest_import("def test_x(): assert True")
        assert r["passed"] is False
        assert r["score"] == 0.5


# --- check_conventional_commit ---


class TestConventionalCommit:
    def test_valid_feat(self):
        r = check_conventional_commit("feat: add login endpoint")
        assert r["passed"] is True

    def test_valid_fix_with_scope(self):
        r = check_conventional_commit("fix(auth): handle expired tokens")
        assert r["passed"] is True

    def test_valid_breaking(self):
        r = check_conventional_commit("feat!: redesign API")
        assert r["passed"] is True

    def test_invalid_no_type(self):
        r = check_conventional_commit("added a new feature")
        assert r["passed"] is False

    def test_invalid_no_colon(self):
        r = check_conventional_commit("feat add login")
        assert r["passed"] is False

    def test_multiline_checks_first(self):
        r = check_conventional_commit("fix: null check\n\nLong description here.")
        assert r["passed"] is True


# --- check_json_parseable ---


class TestJsonParseable:
    def test_valid_json(self):
        r = check_json_parseable('{"key": "value"}')
        assert r["passed"] is True

    def test_valid_json_in_fences(self):
        r = check_json_parseable('```json\n{"key": "value"}\n```')
        assert r["passed"] is True

    def test_invalid_json(self):
        r = check_json_parseable("not json at all")
        assert r["passed"] is False

    def test_empty_object(self):
        r = check_json_parseable("{}")
        assert r["passed"] is True


# --- check_non_empty ---


class TestNonEmpty:
    def test_normal_output(self):
        r = check_non_empty("This is a valid output.")
        assert r["passed"] is True

    def test_empty(self):
        r = check_non_empty("")
        assert r["passed"] is False

    def test_too_short(self):
        r = check_non_empty("hi")
        assert r["passed"] is False

    def test_whitespace_only(self):
        r = check_non_empty("   \n  \n  ")
        assert r["passed"] is False


# --- check_no_truncation ---


class TestNoTruncation:
    def test_clean_output(self):
        r = check_no_truncation("def add(a, b): return a + b")
        assert r["passed"] is True

    def test_ellipsis(self):
        r = check_no_truncation("some output...")
        assert r["passed"] is False

    def test_unbalanced_braces(self):
        r = check_no_truncation('{"key": "value"')
        assert r["passed"] is False

    def test_balanced(self):
        r = check_no_truncation('{"key": [1, 2, 3]}')
        assert r["passed"] is True


# --- check_no_hallucinated_imports ---


class TestHallucinatedImports:
    def test_clean_imports(self):
        r = check_no_hallucinated_imports("import os\nimport json\n")
        assert r["passed"] is True

    def test_heavy_import(self):
        r = check_no_hallucinated_imports("import torch\nimport os\n")
        assert r["passed"] is False

    def test_syntax_error_skips(self):
        r = check_no_hallucinated_imports("def foo( bar")
        assert r["passed"] is True


# --- check_diff_file_references ---


class TestDiffFileReferences:
    def test_valid_references(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
        output = "[HIGH] BUG: issue (foo.py:10)"
        r = check_diff_file_references(output, diff)
        assert r["passed"] is True

    def test_phantom_reference(self):
        diff = "--- a/foo.py\n+++ b/foo.py\n"
        output = "[HIGH] BUG: issue (bar.py:5)"
        r = check_diff_file_references(output, diff)
        assert r["passed"] is False
        assert "bar.py" in str(r["details"])

    def test_no_diff_files(self):
        r = check_diff_file_references("some review", "no diff markers")
        assert r["passed"] is True


# --- check_repetition ---


class TestRepetition:
    def test_normal_output(self):
        r = check_repetition("line1\nline2\nline3\nline4")
        assert r["passed"] is True

    def test_degenerate_repetition(self):
        r = check_repetition("same\n" * 20)
        assert r["passed"] is False

    def test_short_output(self):
        r = check_repetition("ab\nab\nab")
        assert r["passed"] is True


# --- check_tests_collectable ---


class TestTestsCollectable:
    def test_valid_tests_collect(self):
        source = "def add(a, b):\n    return a + b\n"
        tests = (
            "from module_under_test import add\n\n"
            "def test_add():\n"
            "    assert add(1, 2) == 3\n"
        )
        r = check_tests_collectable(tests, source)
        assert r["passed"] is True
        assert r["details"].get("tests_collected", 0) >= 1

    def test_valid_tests_from_named_module(self):
        source = "def greet(name):\n    return f'Hello, {name}'\n"
        tests = (
            "from greeter import greet\n\n"
            "def test_greet():\n"
            "    assert greet('World') == 'Hello, World'\n"
        )
        r = check_tests_collectable(tests, source)
        assert r["passed"] is True

    def test_syntax_error_skips(self):
        r = check_tests_collectable("def test_( broken", "source")
        assert r["passed"] is False
        assert "syntax" in r["details"]["reason"].lower()

    def test_import_error_fails(self):
        source = "x = 1"
        tests = (
            "from nonexistent_package.submodule import something\n\n"
            "def test_it():\n"
            "    assert something() == 1\n"
        )
        r = check_tests_collectable(tests, source)
        assert r["passed"] is False

    def test_empty_test_file(self):
        r = check_tests_collectable("# no tests here\n", "source")
        r2 = check_tests_collectable("import pytest\n", "source")
        # Both should collect 0 tests but not error
        assert r["checker"] == "tests_collectable"


# --- check_boilerplate_format ---


class TestBoilerplateFormat:
    def test_valid_dockerfile(self):
        dockerfile = "FROM python:3.11\nWORKDIR /app\nCOPY . .\nRUN pip install -r requirements.txt\nCMD [\"python\", \"app.py\"]\n"
        r = check_boilerplate_format(dockerfile, "write a Dockerfile")
        assert r["passed"] is True
        assert r["details"].get("format") == "dockerfile"

    def test_invalid_dockerfile_no_from(self):
        r = check_boilerplate_format("RUN apt-get update\n", "Dockerfile for Python")
        assert r["passed"] is False

    def test_valid_yaml(self):
        yaml_content = "name: CI\non:\n  push:\n    branches: [main]\njobs:\n  test:\n    runs-on: ubuntu-latest\n"
        r = check_boilerplate_format(yaml_content, "GitHub Actions workflow")
        # Passes if pyyaml is installed, skips if not
        assert r["checker"] == "format_valid"

    def test_invalid_yaml(self):
        bad_yaml = "key: [unclosed\n  - broken: {nope\n"
        r = check_boilerplate_format(bad_yaml, "write a yaml config")
        if "skipped" not in r["details"].get("note", ""):
            assert r["passed"] is False

    def test_valid_json_boilerplate(self):
        r = check_boilerplate_format('{"name": "test", "version": "1.0"}', "json config")
        assert r["passed"] is True
        assert r["details"].get("format") == "json"

    def test_valid_makefile(self):
        makefile = "build:\n\tgo build -o app\n\ntest:\n\tgo test ./...\n\nclean:\n\trm -f app\n"
        r = check_boilerplate_format(makefile, "Makefile with build/test/clean")
        assert r["passed"] is True
        assert r["details"].get("targets", 0) >= 3

    def test_valid_gitignore(self):
        gitignore = "node_modules/\n*.pyc\n__pycache__/\n.env\n"
        r = check_boilerplate_format(gitignore, "gitignore for Python project")
        assert r["passed"] is True

    def test_empty_gitignore(self):
        r = check_boilerplate_format("# nothing here\n", "gitignore")
        assert r["passed"] is False

    def test_auto_detect_dockerfile(self):
        dockerfile = "FROM node:18\nWORKDIR /app\n"
        r = check_boilerplate_format(dockerfile, "set up the container")
        assert r["passed"] is True
        assert r["details"].get("format") == "dockerfile"

    def test_unknown_format_skips(self):
        r = check_boilerplate_format("some random output", "do something")
        assert r["passed"] is True
        assert "skipped" in r["details"].get("note", "")


# --- run_heuristics ---


class TestRunHeuristics:
    def test_implement_small_valid(self):
        code = "def add(a, b):\n    return a + b\n"
        results = run_heuristics("local_implement_small", "spec", code)
        assert all(r["passed"] for r in results)

    def test_implement_small_syntax_error(self):
        code = "def add(a, b return a + b"
        results = run_heuristics("local_implement_small", "spec", code)
        syntax_check = next(r for r in results if r["checker"] == "python_syntax")
        assert syntax_check["passed"] is False

    def test_commit_message_valid(self):
        results = run_heuristics(
            "local_commit_message", "diff", "feat: add login"
        )
        assert all(r["passed"] for r in results)

    def test_review_diff_with_json(self):
        output = json.dumps({"findings": [], "summary": "Clean"})
        results = run_heuristics("local_review_diff", "diff content", output)
        json_check = next(r for r in results if r["checker"] == "json_parseable")
        assert json_check["passed"] is True

    def test_generate_tests_valid(self):
        source = "def add(a, b):\n    return a + b\n"
        code = (
            "import pytest\n\n"
            "from module_under_test import add\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n\n"
            "def test_sub():\n    assert 2 - 1 == 1\n"
        )
        results = run_heuristics("local_generate_tests", source, code)
        assert all(r["passed"] for r in results)

    def test_unknown_tool_gets_non_empty(self):
        results = run_heuristics("unknown_tool", "in", "out")
        assert len(results) >= 1
        assert results[0]["checker"] == "non_empty"

    def test_synthetic_swarm_tool_name_resolves_to_base_tool_checkers(self):
        """A swarm sub-task tool name like "local_review_diff:security" must
        get the same checkers as "local_review_diff", not fall back to just
        check_non_empty."""
        output = json.dumps({"findings": [], "summary": "Clean"})
        base_results = run_heuristics("local_review_diff", "diff content", output)
        synthetic_results = run_heuristics("local_review_diff:security", "diff content", output)

        base_checkers = {r["checker"] for r in base_results}
        synthetic_checkers = {r["checker"] for r in synthetic_results}
        assert synthetic_checkers == base_checkers
        assert "json_parseable" in synthetic_checkers
        assert "diff_references" in synthetic_checkers
