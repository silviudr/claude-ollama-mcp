"""Zero-cost mechanical checks per tool type.

Every checker returns a GradeResult dict with:
  checker, passed, score (0.0-1.0), details (dict).
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import tempfile
from pathlib import Path

# --- individual checkers ---


def check_python_syntax(code: str) -> dict:
    """AST-parse the output — catches hallucinated syntax."""
    cleaned = _strip_fences(code)
    try:
        ast.parse(cleaned)
        return _ok("python_syntax")
    except SyntaxError as e:
        return _fail("python_syntax", reason=str(e), line=e.lineno)


def check_has_test_functions(code: str) -> dict:
    """Generated test code must contain at least one `def test_*`."""
    matches = re.findall(r"^def (test_\w+)", code, re.MULTILINE)
    if matches:
        return _ok("has_test_functions", count=len(matches))
    return _fail("has_test_functions", reason="no test_* functions found")


def check_has_pytest_import(code: str) -> dict:
    """Test files should import pytest or use pytest fixtures."""
    if "import pytest" in code or "from pytest" in code:
        return _ok("pytest_import")
    return _fail("pytest_import", reason="no pytest import found", score=0.5)


def check_conventional_commit(message: str) -> dict:
    """Validate conventional-commit format: type(scope): description."""
    line = message.strip().split("\n")[0]
    pattern = r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+?\))?!?:\s.+"
    if re.match(pattern, line):
        return _ok("conventional_commit")
    return _fail("conventional_commit", reason=f"'{line}' does not match format")


def check_json_parseable(text: str) -> dict:
    """Output that should be JSON actually parses."""
    cleaned = _strip_fences(text)
    try:
        json.loads(cleaned)
        return _ok("json_parseable")
    except (json.JSONDecodeError, ValueError) as e:
        return _fail("json_parseable", reason=str(e))


def check_non_empty(text: str) -> dict:
    """Output is non-empty and non-trivial."""
    stripped = text.strip()
    if not stripped:
        return _fail("non_empty", reason="output is empty")
    if len(stripped) < 5:
        return _fail("non_empty", reason=f"output too short ({len(stripped)} chars)")
    return _ok("non_empty", length=len(stripped))


def check_no_truncation(text: str) -> dict:
    """Detect common signs of truncated output."""
    stripped = text.rstrip()
    truncation_markers = [
        stripped.endswith("..."),
        stripped.endswith("…"),
        stripped.count("{") != stripped.count("}"),
        stripped.count("[") != stripped.count("]"),
    ]
    issues = []
    if stripped.endswith("...") or stripped.endswith("…"):
        issues.append("ends with ellipsis")
    if stripped.count("{") != stripped.count("}"):
        issues.append("unbalanced braces")
    if stripped.count("[") != stripped.count("]"):
        issues.append("unbalanced brackets")
    if issues:
        return _fail("no_truncation", reason="; ".join(issues), score=0.3)
    return _ok("no_truncation")


def check_no_hallucinated_imports(code: str) -> dict:
    """Flag imports that are clearly wrong (common hallucinations)."""
    KNOWN_HALLUCINATIONS = {
        "from transformers import",
        "import tensorflow",
        "import torch",
        "import sklearn",
        "import pandas",
        "import numpy",
    }
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _ok("import_check", note="skipped — syntax error")

    suspicious = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            line = ast.get_source_segment(code, node) or ""
            for marker in KNOWN_HALLUCINATIONS:
                if marker in line:
                    suspicious.append(line.strip())

    if suspicious:
        return _fail(
            "import_check",
            reason="possibly hallucinated heavy imports",
            imports=suspicious,
            score=0.4,
        )
    return _ok("import_check")


def check_diff_file_references(output: str, diff_input: str) -> dict:
    """Verify that files mentioned in review output actually appear in the diff."""
    diff_files = set(re.findall(r"^[+-]{3}\s+[ab]/(.+)$", diff_input, re.MULTILINE))
    if not diff_files:
        return _ok("diff_references", note="no diff files parsed")

    # Match both (file.py:10) format and JSON "file": "file.py" format
    paren_refs = set(re.findall(r"\((\S+?\.\w+?)(?::\d+)?\)", output))
    json_refs = set(re.findall(r'"file"\s*:\s*"([^"]+?\.\w+)"', output))
    referenced = paren_refs | json_refs

    phantom = referenced - diff_files
    if phantom:
        return _fail(
            "diff_references",
            reason=f"references files not in diff: {phantom}",
            phantom_files=list(phantom),
            score=0.5,
        )
    return _ok("diff_references")


def check_repetition(text: str) -> dict:
    """Detect degenerate repetition in output."""
    lines = text.strip().splitlines()
    if len(lines) < 4:
        return _ok("no_repetition")
    unique = set(lines)
    ratio = len(unique) / len(lines)
    if ratio < 0.3:
        return _fail(
            "no_repetition",
            reason=f"only {len(unique)}/{len(lines)} unique lines ({ratio:.0%})",
            score=ratio,
        )
    return _ok("no_repetition", unique_ratio=round(ratio, 2))


# --- test collection checker ---


def check_tests_collectable(test_code: str, source_code: str) -> dict:
    """Write generated tests + source to a tmpdir and run pytest --collect-only.

    Validates that tests are importable and discoverable, not just syntactically
    valid. Catches broken imports, bad fixtures, and mangled signatures.
    """
    test_code = _strip_fences(test_code)
    source_code = _strip_fences(source_code)
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return _fail("tests_collectable", reason="syntax error — skipping collection")

    # Find the module name the tests import from
    module_name = _extract_import_module(tree)
    if not module_name:
        module_name = "module_under_test"

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Write source as the importable module
        (tmp / f"{module_name}.py").write_text(source_code)
        test_file = tmp / "test_generated.py"
        test_file.write_text(test_code)

        try:
            result = subprocess.run(
                ["python3", "-m", "pytest", "--collect-only", "-q", str(test_file)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmpdir,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except FileNotFoundError:
            return _ok("tests_collectable", note="pytest not found — skipped")
        except subprocess.TimeoutExpired:
            return _fail("tests_collectable", reason="pytest collection timed out")

        if result.returncode == 0:
            collected = re.search(r"(\d+) tests? collected", result.stdout)
            count = int(collected.group(1)) if collected else 0
            return _ok("tests_collectable", tests_collected=count)

        # Parse the error for a useful message
        stderr = result.stderr.strip() or result.stdout.strip()
        error_lines = [
            l for l in stderr.splitlines()
            if l.strip() and not l.startswith("=")
        ]
        short_error = error_lines[-1] if error_lines else "collection failed"
        return _fail(
            "tests_collectable",
            reason=short_error[:200],
            exit_code=result.returncode,
        )


def _extract_import_module(tree: ast.AST) -> str | None:
    """Find the first `from X import ...` that isn't stdlib/pytest."""
    SKIP = {"pytest", "unittest", "os", "sys", "json", "re", "math",
            "typing", "collections", "functools", "itertools", "pathlib",
            "dataclasses", "abc", "enum", "datetime", "io", "copy",
            "contextlib", "tempfile", "textwrap", "hashlib", "hmac",
            "base64", "uuid", "random", "string", "struct", "time"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top not in SKIP:
                return top
    return None


# --- boilerplate format validation ---


_FORMAT_HINTS = [
    (r"(?i)\bdockerfile\b", "dockerfile"),
    (r"(?i)\byaml\b|\byml\b|github.actions|ci.cd|workflow", "yaml"),
    (r"(?i)\btoml\b|pyproject", "toml"),
    (r"(?i)\bjson\b", "json"),
    (r"(?i)\bmakefile\b|\bmake\b", "makefile"),
    (r"(?i)\bgitignore\b", "gitignore"),
    (r"(?i)\bini\b|\bcfg\b|\.ini|\.cfg|setup\.cfg", "ini"),
]


def check_boilerplate_format(output: str, spec: str) -> dict:
    """Auto-detect the expected format from the spec and validate the output."""
    fmt = _detect_format(output, spec)
    if fmt is None:
        return _ok("format_valid", note="format not detected — skipped")

    validators = {
        "yaml": _validate_yaml,
        "json": _validate_json,
        "toml": _validate_toml,
        "dockerfile": _validate_dockerfile,
        "makefile": _validate_makefile,
        "ini": _validate_ini,
        "gitignore": _validate_gitignore,
    }

    validator = validators.get(fmt)
    if validator is None:
        return _ok("format_valid", note=f"no validator for {fmt}")

    return validator(output)


def _detect_format(output: str, spec: str) -> str | None:
    """Detect the boilerplate format from the spec or output content."""
    # Check spec first
    for pattern, fmt in _FORMAT_HINTS:
        if re.search(pattern, spec):
            return fmt

    # Fallback: detect from output content
    stripped = output.strip()
    if stripped.startswith("FROM ") or stripped.startswith("ARG "):
        return "dockerfile"
    if stripped.startswith("---") or re.match(r"^\w+:", stripped):
        return "yaml"
    if stripped.startswith("{"):
        return "json"
    if stripped.startswith("[") and "=" in stripped:
        return "toml"
    if re.match(r"^[\w.-]+\s*:", stripped, re.MULTILINE) and "\t" in stripped:
        return "makefile"
    return None


def _validate_yaml(output: str) -> dict:
    try:
        import yaml
    except ImportError:
        return _ok("format_valid", note="pyyaml not installed — skipped")
    try:
        result = yaml.safe_load(output)
        if result is None and output.strip():
            return _fail("format_valid", reason="YAML parsed as None/empty")
        return _ok("format_valid", format="yaml")
    except yaml.YAMLError as e:
        return _fail("format_valid", reason=f"invalid YAML: {e}", format="yaml")


def _validate_json(output: str) -> dict:
    cleaned = _strip_fences(output)
    try:
        json.loads(cleaned)
        return _ok("format_valid", format="json")
    except (json.JSONDecodeError, ValueError) as e:
        return _fail("format_valid", reason=f"invalid JSON: {e}", format="json")


def _validate_toml(output: str) -> dict:
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            return _ok("format_valid", note="tomllib not available — skipped")
    try:
        tomllib.loads(output)
        return _ok("format_valid", format="toml")
    except Exception as e:
        return _fail("format_valid", reason=f"invalid TOML: {e}", format="toml")


def _validate_dockerfile(output: str) -> dict:
    lines = [l.strip() for l in output.strip().splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return _fail("format_valid", reason="Dockerfile is empty", format="dockerfile")

    valid_instructions = {
        "FROM", "RUN", "CMD", "LABEL", "MAINTAINER", "EXPOSE", "ENV",
        "ADD", "COPY", "ENTRYPOINT", "VOLUME", "USER", "WORKDIR", "ARG",
        "ONBUILD", "STOPSIGNAL", "HEALTHCHECK", "SHELL",
    }
    has_from = False
    bad_lines = []
    for line in lines:
        # Handle multi-line continuations
        if line.endswith("\\"):
            continue
        instruction = line.split()[0].upper() if line.split() else ""
        if instruction == "FROM":
            has_from = True
        if instruction and instruction not in valid_instructions:
            # Could be a continuation or argument line
            if not any(line.startswith(c) for c in ("&&", "|", "-", "'")):
                bad_lines.append(line[:60])

    if not has_from:
        return _fail(
            "format_valid", reason="Dockerfile missing FROM instruction",
            format="dockerfile",
        )
    if len(bad_lines) > len(lines) * 0.5:
        return _fail(
            "format_valid",
            reason=f"many unrecognized instructions: {bad_lines[:3]}",
            format="dockerfile",
            score=0.4,
        )
    return _ok("format_valid", format="dockerfile")


def _validate_makefile(output: str) -> dict:
    lines = output.strip().splitlines()
    if not lines:
        return _fail("format_valid", reason="Makefile is empty", format="makefile")

    targets = [l for l in lines if re.match(r"^[\w.-]+\s*:", l) and not l.startswith("\t")]
    if not targets:
        return _fail(
            "format_valid", reason="no Makefile targets found",
            format="makefile",
        )
    return _ok("format_valid", format="makefile", targets=len(targets))


def _validate_ini(output: str) -> dict:
    import configparser
    parser = configparser.ConfigParser()
    try:
        parser.read_string(output)
        sections = parser.sections()
        if not sections and output.strip():
            return _fail("format_valid", reason="INI has no sections", format="ini")
        return _ok("format_valid", format="ini", sections=len(sections))
    except configparser.Error as e:
        return _fail("format_valid", reason=f"invalid INI: {e}", format="ini")


def _validate_gitignore(output: str) -> dict:
    lines = [l for l in output.strip().splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        return _fail("format_valid", reason=".gitignore has no patterns", format="gitignore")
    return _ok("format_valid", format="gitignore", patterns=len(lines))


# --- tool → checkers mapping ---

TOOL_CHECKERS: dict[str, list] = {
    "local_summarize": [check_non_empty, check_no_truncation, check_repetition],
    "local_draft_boilerplate": [
        check_non_empty,
        check_no_truncation,
        check_repetition,
    ],
    "local_implement_small": [
        check_non_empty,
        check_python_syntax,
        check_no_hallucinated_imports,
        check_no_truncation,
        check_repetition,
    ],
    "local_commit_message": [check_non_empty, check_conventional_commit],
    "local_review_diff": [check_non_empty, check_json_parseable, check_no_truncation],
    "local_generate_tests": [
        check_non_empty,
        check_python_syntax,
        check_has_test_functions,
        check_has_pytest_import,
        check_no_truncation,
        check_repetition,
    ],
    "local_classify_task": [check_non_empty, check_json_parseable],
    "local_analyze_data": [check_non_empty],
}

# Checkers that need (output, input) — dispatched specially
_DUAL_INPUT_CHECKERS = {check_diff_file_references, check_tests_collectable,
                        check_boilerplate_format}


def run_heuristics(
    tool_name: str, input_text: str, output_text: str
) -> list[dict]:
    """Run all applicable heuristic checks for a tool.

    Swarm sub-tasks are logged under synthetic names like
    "local_review_diff:security" so they get their own telemetry/grading
    rows — but checker selection must resolve to the base tool name, or
    every sub-task would silently fall back to only check_non_empty.
    """
    base_tool = tool_name.split(":", 1)[0]
    checkers = TOOL_CHECKERS.get(base_tool, [check_non_empty])
    results = []
    for checker in checkers:
        if checker in _DUAL_INPUT_CHECKERS:
            results.append(checker(output_text, input_text))
        else:
            results.append(checker(output_text))

    # Tool-specific extra checks that need both input and output
    if base_tool == "local_review_diff":
        results.append(check_diff_file_references(output_text, input_text))
    if base_tool == "local_generate_tests":
        results.append(check_tests_collectable(output_text, input_text))
    if base_tool == "local_draft_boilerplate":
        results.append(check_boilerplate_format(output_text, input_text))

    return results


# --- helpers ---

_FENCE_RE = re.compile(r"```\w*\s*\n?(.*?)\n?\s*```", re.DOTALL)


def _strip_fences(text: str) -> str:
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _ok(checker: str, score: float = 1.0, **extra) -> dict:
    return {"checker": checker, "passed": True, "score": score, "details": extra}


def _fail(checker: str, reason: str, score: float = 0.0, **extra) -> dict:
    return {
        "checker": checker,
        "passed": False,
        "score": score,
        "details": {"reason": reason, **extra},
    }
