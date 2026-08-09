"""Tests for the standalone audit sweeper (claude/scripts/local_audit.py).

The sweeper lives outside the package on purpose — it installs into
~/.claude/scripts and must run on a bare system python3 with no dependencies,
so it cannot import from ollama_mcp. Load it by path.
"""

import importlib.util
import json
import pathlib

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[1] / "claude" / "scripts" / "local_audit.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("local_audit_under_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


la = _load()


# ---------------------------------------------------------------- extract_json


def test_extract_json_plain():
    assert la.extract_json('{"findings": []}') == {"findings": []}


def test_extract_json_strips_markdown_fence():
    raw = 'Here you go:\n```json\n{"findings": [{"title": "x"}]}\n```\n'
    assert la.extract_json(raw)["findings"][0]["title"] == "x"


def test_extract_json_strips_reasoning_block():
    """Reasoning models emit <think> before the answer; a naive brace scan
    would latch onto a JSON-looking fragment inside it."""
    raw = '<think>maybe {"findings": [{"title": "wrong"}]} hmm</think>{"findings": []}'
    assert la.extract_json(raw) == {"findings": []}


def test_extract_json_repairs_trailing_comma():
    assert la.extract_json('{"findings": [{"title": "a"},]}')["findings"]


def test_extract_json_salvages_truncated_array():
    """A model that spends its budget thinking gets cut off mid-array. The
    finished objects are still well-formed and must not be thrown away."""
    raw = (
        '{"findings": [{"severity":"high","title":"SQLi","line":5,'
        '"description":"d"}, {"severity":"medium","title":"XSS","line":9,'
        '"description":"d"}, {"severity":"low","title":"trunc'
    )
    got = la.extract_json(raw)
    assert [f["title"] for f in got["findings"]] == ["SQLi", "XSS"]


@pytest.mark.parametrize("bad", ["", "   ", "no json here", "<think>only</think>"])
def test_extract_json_returns_none_not_empty_findings(bad):
    """Critical: an unparseable response must be distinguishable from a clean
    file. Returning {"findings": []} here would silently report broken scans
    as passing."""
    assert la.extract_json(bad) is None


def test_salvage_ignores_objects_without_title():
    assert la._salvage_objects('{"a": 1} {"title": "real"}') == [{"title": "real"}]


# ---------------------------------------------------------------- chunking


def test_chunk_numbers_lines_from_one(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("alpha\nbeta\ngamma\n")
    (lo, hi, body), = la.chunk(f, 100)
    assert (lo, hi) == (1, 3)
    assert "    1| alpha" in body
    assert "    3| gamma" in body


def test_chunk_splits_large_file_without_gaps(tmp_path):
    f = tmp_path / "big.py"
    f.write_text("\n".join(f"line{i}" for i in range(500)))
    chunks = la.chunk(f, 100)
    assert len(chunks) > 1
    assert chunks[0][0] == 1
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt[0] == prev[1] + 1, "chunk boundaries must not skip lines"
    assert chunks[-1][1] == 500


def test_chunk_prefers_definition_boundary(tmp_path):
    f = tmp_path / "m.py"
    lines = ["import os"] + [f"    x = {i}" for i in range(28)] + ["def later():", "    pass"]
    f.write_text("\n".join(lines))
    chunks = la.chunk(f, 30)
    assert chunks[0][1] == 29, "should break just before 'def later()', not mid-body"


def test_chunk_handles_empty_file(tmp_path):
    f = tmp_path / "empty.py"
    f.write_text("")
    assert la.chunk(f, 100) == []


# ---------------------------------------------------------------- discovery


def test_discover_skips_vendored_and_generated(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("x = 1")
    for junk in ("node_modules", ".venv", "__pycache__"):
        d = tmp_path / junk
        d.mkdir()
        (d / "dep.py").write_text("x = 1")
    (tmp_path / "bundle.min.js").write_text("var a=1")
    (tmp_path / "package-lock.json").write_text("{}")

    found = {p.name for p in la.discover(tmp_path, include_tests=False)}
    assert found == {"app.py"}


def test_discover_excludes_then_includes_tests(tmp_path):
    (tmp_path / "app.py").write_text("x = 1")
    (tmp_path / "test_app.py").write_text("x = 1")
    assert {p.name for p in la.discover(tmp_path, False)} == {"app.py"}
    assert {p.name for p in la.discover(tmp_path, True)} == {"app.py", "test_app.py"}


def test_discover_orders_risky_files_first(tmp_path):
    for name in ("zebra.py", "auth.py"):
        (tmp_path / name).write_text("x = 1")
    assert la.discover(tmp_path, False)[0].name == "auth.py"


# ---------------------------------------------------------------- config gate


def _write_routes(tmp_path, monkeypatch, cfg):
    p = tmp_path / "routes.json"
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr(la, "ROUTES", p)
    return p


def test_load_config_refuses_cloud_backend(tmp_path, monkeypatch):
    """The local-only guarantee is the whole point; it must fail loudly."""
    _write_routes(tmp_path, monkeypatch, {
        "backends": {
            "ollama": {"url": "http://localhost:11434", "default_model": "m"},
            "openrouter": {"api_key": "sk-x", "default_model": "a/b"},
        }
    })
    with pytest.raises(SystemExit) as e:
        la.load_config()
    assert "openrouter" in str(e.value)


def test_load_config_refuses_prefixed_model_in_dimension(tmp_path, monkeypatch):
    _write_routes(tmp_path, monkeypatch, {
        "backends": {"ollama": {"url": "http://h:11434", "default_model": "m"}},
        "swarm": {"review_dimensions": {
            "local_review_diff": {"security": "openrouter/vendor/model"}
        }},
    })
    with pytest.raises(SystemExit) as e:
        la.load_config()
    assert "non-local" in str(e.value)


def test_load_config_reads_dimensions(tmp_path, monkeypatch):
    _write_routes(tmp_path, monkeypatch, {
        "backends": {"ollama": {"url": "http://h:11434/", "default_model": "gen"}},
        "swarm": {"review_dimensions": {
            "local_review_diff": {"security": "sec-model", "correctness": "cor-model"}
        }},
    })
    url, dims = la.load_config()
    assert url == "http://h:11434"
    assert dims == {"security": "sec-model", "correctness": "cor-model"}


def test_load_config_falls_back_to_default_model(tmp_path, monkeypatch):
    _write_routes(tmp_path, monkeypatch, {
        "backends": {"ollama": {"url": "http://h:11434", "default_model": "gen"}}
    })
    _, dims = la.load_config()
    assert dims == {"security": "gen"}


def test_load_config_ignores_unknown_dimension_names(tmp_path, monkeypatch):
    """Only dimensions with a defined focus prompt can be run."""
    _write_routes(tmp_path, monkeypatch, {
        "backends": {"ollama": {"url": "http://h:11434", "default_model": "gen"}},
        "swarm": {"review_dimensions": {
            "local_review_diff": {"security": "a", "performance": "b", "tests": "c"}
        }},
    })
    _, dims = la.load_config()
    assert dims == {"security": "a"}


# ---------------------------------------------------------------- merging


def _f(dim, line, sev, title, model="m"):
    return {
        "dimension": dim, "line": line, "severity": sev, "title": title,
        "description": "d", "fix": "", "model": model, "file": "a.py",
    }


def test_cluster_merges_differently_worded_titles():
    """Two models never phrase the same defect identically."""
    got = la._cluster_findings([
        _f("security", 16, "high", "SQL Injection Vulnerability"),
        _f("correctness", 16, "high", "SQL injection in user lookup query"),
    ])
    assert len(got) == 1


def test_cluster_merges_across_nearby_lines():
    got = la._cluster_findings([
        _f("security", 30, "high", "Path Traversal Vulnerability"),
        _f("correctness", 31, "high", "Path traversal when serving avatar files"),
    ])
    assert len(got) == 1


def test_cluster_keeps_distinct_defects_apart():
    """Both end in 'injection' but they are different bugs."""
    got = la._cluster_findings([
        _f("security", 16, "high", "SQL injection"),
        _f("security", 17, "high", "Command injection"),
    ])
    assert len(got) == 2


def test_cluster_keeps_same_title_far_apart():
    got = la._cluster_findings([
        _f("security", 10, "high", "Hardcoded secret"),
        _f("security", 400, "high", "Hardcoded secret"),
    ])
    assert len(got) == 2


# ---------------------------------------------------------------- report


def _stats(failed=None):
    return {
        "url": "http://h:11434", "files": 1, "chunks": 1, "calls": 2,
        "failed": failed or [], "elapsed": "0m 1s",
    }


def test_report_marks_cross_dimension_agreement(tmp_path):
    out = la.build_report(
        tmp_path,
        [
            _f("security", 16, "high", "SQL Injection Vulnerability", "model-a"),
            _f("correctness", 16, "high", "SQL injection in lookup", "model-b"),
        ],
        _stats(), {"security": "model-a", "correctness": "model-b"},
    )
    assert "CONFIRMED BY BOTH" in out
    assert "correctness+security" in out


def test_report_takes_most_severe_when_models_disagree(tmp_path):
    out = la.build_report(
        tmp_path,
        [
            _f("security", 52, "high", "Debug mode enabled"),
            _f("correctness", 52, "low", "Debug mode enabled in production"),
        ],
        _stats(), {"security": "a", "correctness": "b"},
    )
    assert "[HIGH]" in out and "[LOW]" not in out


def test_report_surfaces_coverage_gaps(tmp_path):
    """A failed chunk must never be silently absent from the report."""
    failed = [{"file": "x.py", "range": "1-350", "dim": "security",
               "error": "unparseable model output (0 chars)"}]
    out = la.build_report(tmp_path, [], _stats(failed), {"security": "a"})
    assert "NOT AUDITED" in out
    assert "x.py" in out


def test_report_groups_each_file_once(tmp_path):
    """Sorting purely by severity used to split one file across sections."""
    out = la.build_report(
        tmp_path,
        [
            {**_f("security", 5, "high", "Alpha"), "file": "a.py"},
            {**_f("security", 9, "medium", "Beta"), "file": "b.py"},
            {**_f("security", 20, "medium", "Gamma"), "file": "a.py"},
        ],
        _stats(), {"security": "m"},
    )
    assert out.count("### `a.py`") == 1


def test_report_states_output_is_unverified(tmp_path):
    out = la.build_report(tmp_path, [], _stats(), {"security": "a"})
    assert "Unverified" in out
