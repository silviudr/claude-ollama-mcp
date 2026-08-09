#!/usr/bin/env python3
"""Full-coverage, local-only security audit.

Sweeps EVERY source file in a project through local Ollama models. Runs
outside any LLM context window, so coverage does not degrade on large repos
and cannot silently stop early.

Refuses to run if any review dimension routes to a non-Ollama backend. A cloud
backend merely being declared is reported but allowed, so a hybrid config can
still audit locally; pass --strict to refuse on declaration alone.

Usage:
    local_audit.py [PATH] [--out SECURITY-AUDIT.md] [--resume] [--limit N]
                          [--chunk-lines 350] [--concurrency 2] [--strict]
                          [--include-tests] [--seed N] [--keep-alive 30m]

Progress goes to stdout, flushed on every line: piped stdout (background runs,
tee, CI) is fully buffered, so an unflushed line can sit invisible for minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROUTES = Path(
    os.environ.get(
        "OLLAMA_MCP_ROUTES", Path.home() / ".config" / "ollama_mcp" / "routes.json"
    )
)

SOURCE_EXT = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".mjs", ".cjs", ".go", ".rb", ".php",
    ".java", ".kt", ".cs", ".rs", ".c", ".cc", ".cpp", ".h", ".hpp", ".sh",
    ".bash", ".sql", ".tf", ".yaml", ".yml",
}
CONFIG_NAMES = {"dockerfile", "docker-compose.yml", "makefile", ".env.example"}

EXCLUDE_DIRS = {
    ".git", "node_modules", ".venv", "venv", "env", "__pycache__", "dist",
    "build", "site-packages", ".next", ".nuxt", "vendor", "target",
    ".pytest_cache", ".mypy_cache", ".tox", "coverage", ".idea", ".cache",
    "migrations", ".terraform", "bower_components", "eggs", ".eggs",
}
EXCLUDE_SUFFIX = (".min.js", ".min.css", ".bundle.js", ".lock", ".map", ".pb.go")
EXCLUDE_FILES = {
    "package-lock.json", "yarn.lock", "poetry.lock", "pnpm-lock.yaml",
    "Cargo.lock", "composer.lock", "go.sum",
}

MAX_BYTES = 400_000  # skip anything bigger; almost certainly generated

# Emit a progress line at least this often, even when calls are slow.
HEARTBEAT_S = 60

# Ordering only — everything still gets audited, this just front-loads risk.
HOT = (
    "auth", "login", "session", "token", "password", "crypt", "secret", "key",
    "admin", "payment", "billing", "upload", "api", "route", "handler", "view",
    "controller", "middleware", "query", "db", "sql", "user", "account",
    "permission", "role", "security", "exec", "shell", "deserial",
)

SYSTEM = (
    "You are a security code auditor. You are given one chunk of a source "
    "file. Report ONLY concrete, evidence-backed defects you can point at in "
    "this chunk.\n"
    "Rules:\n"
    "- Do NOT invent issues. If the chunk is clean, return an empty list.\n"
    "- Do NOT report style, formatting, naming, or missing docstrings.\n"
    "- 'line' must be a real line number from the LINE markers shown.\n"
    "- Severity: high = exploitable or data-exposing; medium = real weakness "
    "needing specific conditions; low = hardening / defence-in-depth.\n"
    "\n"
    "Respond with ONLY a JSON object, no prose and no markdown fence:\n"
    '{"findings":[{"severity":"high|medium|low","title":"...","line":<int>,'
    '"description":"...","fix":"..."}]}\n'
    'Return {"findings":[]} when the chunk is clean.\n'
)

FOCUS = {
    "security": (
        "Focus EXCLUSIVELY on security: injection (SQL/command/template/LDAP), "
        "unsafe deserialization, path traversal, SSRF, XSS, auth and access "
        "control gaps, hardcoded or leaked secrets, weak or misused crypto, "
        "unsafe randomness, timing-unsafe comparisons, missing input "
        "validation, and dangerous defaults (debug on, binding 0.0.0.0, "
        "verify=False, permissive CORS)."
    ),
    "correctness": (
        "Focus EXCLUSIVELY on correctness bugs with security consequence: "
        "wrong boundary or off-by-one checks, unhandled errors that fail "
        "open, race conditions and TOCTOU, resource leaks, incorrect "
        "null/None handling, swallowed exceptions hiding failures, and logic "
        "errors in validation or authorization decisions."
    ),
}

_THINK = re.compile(r"<think>.*?</think>|<thinking>.*?</thinking>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a model response.

    Deliberately NOT Ollama's `format` schema parameter. Schema-constrained
    decoding returns an empty string on every reasoning model installed here
    (nemotron-3-nano, qwen3.5, qwen3.6) and invalid JSON on gemma4 — only
    qwen3-coder and llama3.2 survive it. Asking for JSON in the prompt and
    parsing tolerantly works on all of them.
    """
    if not text or not text.strip():
        return None
    t = _THINK.sub("", text)
    m = _FENCE.search(t)
    if m:
        t = m.group(1)
    start, end = t.find("{"), t.rfind("}")
    if start == -1 or end <= start:
        return None
    blob = t[start : end + 1]
    for attempt in (blob, _TRAILING_COMMA.sub(r"\1", blob)):
        try:
            parsed = json.loads(attempt)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Truncated output: a reasoning model can spend its budget thinking and
    # get cut off before closing the array. The individual finding objects
    # are still well-formed, so recover those rather than losing the chunk.
    salvaged = _salvage_objects(t)
    return {"findings": salvaged} if salvaged else None


def _salvage_objects(text: str) -> list[dict]:
    """Extract every balanced {...} that looks like a finding, at any depth.

    Depth matters: the case this exists for is a truncated response, where the
    OUTER brace never closes, so every finding object sits nested inside it.
    Recording only depth-0 objects salvages nothing precisely when salvage is
    needed. The enclosing {"findings": [...]} wrapper is skipped for lacking a
    title, so a well-formed response cannot double-count.
    """
    out: list[dict] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "{":
            stack.append(i)
        elif ch == "}" and stack:
            start = stack.pop()
            try:
                obj = json.loads(text[start : i + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("title"):
                out.append(obj)
    return out


# ---------------------------------------------------------------- config


# Backend names this project can route to that are not the local Ollama
# server. A dimension prefixed with one of these is a cloud route even when
# the backend is not currently declared.
REMOTE_BACKENDS = {"openrouter"}


def load_config(strict: bool = False) -> tuple[str, dict[str, str], list[str]]:
    """Return (ollama_url, {dimension: model}, unused_remote_backend_names).

    The guarantee enforced here is "every model this run sends code to is
    local", checked against the dimensions actually used — not "no cloud
    backend is mentioned anywhere". Those differ, and the distinction matters:
    plenty of people want OpenRouter for local_summarize while keeping audits
    on-premises, and this script cannot reach a cloud host anyway (it POSTs
    only to backends.ollama.url and imports nothing from ollama_mcp).

    `strict` restores the blunt check — refuse if a remote backend is so much
    as declared — for anyone who wants the config itself to be evidence.
    """
    if not ROUTES.exists():
        sys.exit(f"FATAL: no routing config at {ROUTES}")

    cfg = json.loads(ROUTES.read_text())
    backends = cfg.get("backends", {})

    remote = sorted(n for n in backends if n != "ollama")
    if remote and strict:
        sys.exit(
            "FATAL: --strict and remote backend(s) declared: "
            f"{', '.join(remote)}.\nRemove them from {ROUTES}, or drop "
            "--strict to check the audit's own models instead."
        )

    url = backends.get("ollama", {}).get("url", "http://localhost:11434")
    default_model = backends.get("ollama", {}).get("default_model", "")

    dims = dict(
        cfg.get("swarm", {}).get("review_dimensions", {}).get("local_review_diff", {})
    )
    dims = {k: v for k, v in dims.items() if k in FOCUS}
    if not dims:
        if not default_model:
            sys.exit("FATAL: no review dimensions and no default model configured.")
        dims = {"security": default_model}

    # Match only a backend-name prefix. A bare "/" means nothing on its own —
    # Ollama tags pulled from HuggingFace legitimately contain slashes, e.g.
    # hf.co/bartowski/Some-Model-GGUF:Q4_K_M, and rejecting those would be
    # wrong.
    routable_remote = (set(backends) | REMOTE_BACKENDS) - {"ollama"}
    for dim, model in sorted(dims.items()):
        prefix = model.split("/", 1)[0]
        if prefix in routable_remote:
            sys.exit(
                f"FATAL: refusing to run — dimension '{dim}' routes to the "
                f"remote backend '{prefix}' ({model}).\nThis audit is "
                f"local-only by contract. Point it at a local model in {ROUTES}."
            )

    return url.rstrip("/"), dims, remote


# ---------------------------------------------------------------- files


def discover(root: Path, include_tests: bool) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        for fn in filenames:
            p = Path(dirpath) / fn
            low = fn.lower()
            if low in EXCLUDE_FILES or low.endswith(EXCLUDE_SUFFIX):
                continue
            if p.suffix.lower() not in SOURCE_EXT and low not in CONFIG_NAMES:
                continue
            if not include_tests and (
                "test" in low or f"{os.sep}tests{os.sep}" in str(p).lower()
            ):
                continue
            try:
                if p.stat().st_size > MAX_BYTES or p.stat().st_size == 0:
                    continue
            except OSError:
                continue
            out.append(p)

    def rank(p: Path) -> tuple[int, str]:
        s = str(p).lower()
        return (0 if any(h in s for h in HOT) else 1, str(p))

    return sorted(out, key=rank)


def chunk(path: Path, max_lines: int) -> list[tuple[int, int, str]]:
    """Split into (start_line, end_line, numbered_text). Never mid-function."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    if not lines:
        return []

    bounds: list[int] = []
    for i, ln in enumerate(lines):
        st = ln.lstrip()
        if not st:
            continue
        indent = len(ln) - len(st)
        if indent == 0 and (
            st.startswith(("def ", "class ", "async def ", "func ", "type ", "@"))
            or (" function " in f" {st} ")
            or st.startswith(("export ", "const ", "public ", "private ", "impl "))
        ):
            bounds.append(i)

    out, start = [], 0
    n = len(lines)
    while start < n:
        target = min(start + max_lines, n)
        if target < n:
            nice = [b for b in bounds if start + max_lines // 2 < b <= target]
            if nice:
                target = nice[-1]
        seg = "\n".join(f"{start + j + 1:5d}| {l}" for j, l in enumerate(lines[start:target]))
        out.append((start + 1, target, seg))
        start = target
    return out


# ---------------------------------------------------------------- inference


def review(
    url: str, model: str, dim: str,
    rel: str, lo: int, hi: int, body: str, timeout: float, seed: int = 42,
    keep_alive: str = "30m",
) -> tuple[list[dict], str | None]:
    prompt = (
        f"FILE: {rel}\nLINES: {lo}-{hi}\n\n"
        f"{FOCUS[dim]}\n\nSource chunk (each line prefixed with its real line "
        f"number):\n\n{body}\n"
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM,
        "stream": False,
        # Hold the model in memory across the run AND between runs. Output is
        # byte-identical while a model stays loaded, but each fresh load
        # settles into a slightly different stable state (measured: same
        # findings on the same lines, occasionally different title wording or
        # one severity notch). Avoiding the reload is what makes two separate
        # audits of unchanged code comparable.
        "keep_alive": keep_alive,
        "options": {
            # Greedy + fixed seed. Measured: nemotron-3-nano returned three
            # different answers to an identical prompt at temperature 0.1 and
            # became byte-identical under these settings. Without this, a
            # re-run of the same audit reports a different set of findings.
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": seed,
            # Pinned so context size never varies with server defaults, and
            # to stop a 350-line chunk reserving tens of GB of KV cache.
            "num_ctx": 16384,
            # Generous: reasoning models spend a large share of this budget on
            # thinking tokens before emitting any JSON at all.
            "num_predict": 3500,
        },
    }
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        if "error" in data:
            return [], str(data["error"])[:160]
        raw = data.get("response") or ""
    except urllib.error.HTTPError as e:
        return [], f"HTTP {e.code}: {e.read()[:120].decode('utf-8', 'replace')}"
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        return [], "Timeout" if isinstance(reason, (socket.timeout, TimeoutError)) \
            else f"URLError: {reason}"
    except ValueError as e:
        return [], f"bad HTTP body: {e}"

    parsed = extract_json(raw)
    if parsed is None:
        # An empty body is the reasoning-model failure mode and must be
        # reported as NOT AUDITED, never silently treated as "clean".
        return [], f"unparseable model output ({len(raw)} chars)"

    findings = []
    for f in parsed.get("findings") or []:
        if not isinstance(f, dict) or not f.get("title"):
            continue
        try:
            line = int(f.get("line") or lo)
        except (TypeError, ValueError):
            line = lo
        findings.append({
            "severity": str(f.get("severity", "low")).lower(),
            "title": str(f["title"])[:200],
            "line": max(lo, min(line, hi)),   # clamp hallucinated line numbers
            "description": str(f.get("description", ""))[:1200],
            "fix": str(f.get("fix", ""))[:600],
            "dimension": dim,
            "model": model,
            "file": rel,
        })
    return findings, None


# ---------------------------------------------------------------- report


# Words that carry no discriminating signal in a finding title — without
# stripping these, "SQL Injection Vulnerability" and "Command Injection
# Vulnerability" look similar purely because both end in "vulnerability".
_TITLE_STOP = {
    "vulnerability", "vulnerabilities", "issue", "issues", "risk", "risky",
    "potential", "possible", "unsafe", "insecure", "improper", "missing",
    "code", "source", "the", "and", "for", "with", "when", "via", "using",
    "from", "into", "user", "input", "data", "function", "method", "call",
}
_MERGE_LINE_WINDOW = 4
_MERGE_THRESHOLD = 0.6


def _title_words(title: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", title.lower())
        if len(w) > 2 and w not in _TITLE_STOP
    }


def _cluster_findings(group: list[dict]) -> list[list[dict]]:
    """Group findings that describe the same defect.

    Two models rarely phrase a title identically ("SQL Injection
    Vulnerability" vs "SQL injection in user lookup query") or land on the
    exact same line, so match on nearby lines plus overlap of meaningful
    title words, normalised by the shorter title.
    """
    clusters: list[dict] = []
    for f in sorted(group, key=lambda x: x["line"]):
        words = _title_words(f["title"])
        for c in clusters:
            if abs(c["line"] - f["line"]) > _MERGE_LINE_WINDOW:
                continue
            shared = len(c["words"] & words)
            if not shared:
                continue
            if shared / max(1, min(len(c["words"]), len(words))) >= _MERGE_THRESHOLD:
                c["items"].append(f)
                c["words"] |= words
                break
        else:
            clusters.append({"line": f["line"], "words": set(words), "items": [f]})
    return [c["items"] for c in clusters]


def build_report(root: Path, findings: list[dict], stats: dict, dims: dict) -> str:
    by_file: dict[str, list[dict]] = {}
    for f in findings:
        by_file.setdefault(f["file"], []).append(f)

    rank = {"high": 0, "medium": 1, "low": 2}
    merged: list[dict] = []
    for group in by_file.values():
        for cluster in _cluster_findings(group):
            best = dict(sorted(cluster, key=lambda x: rank.get(x["severity"], 3))[0])
            best["reported_by"] = sorted({g["dimension"] for g in cluster})
            best["models"] = sorted({g["model"] for g in cluster})
            if len(cluster) > 1:
                best["aliases"] = [
                    g["title"] for g in cluster if g["title"] != best["title"]
                ]
            merged.append(best)

    order = {"high": 0, "medium": 1, "low": 2}
    # Group by file first, files ordered by their worst finding. Sorting by
    # severity globally splits one file across several sections of the report.
    worst: dict[str, int] = {}
    for f in merged:
        r = order.get(f["severity"], 3)
        worst[f["file"]] = min(worst.get(f["file"], 9), r)
    merged.sort(
        key=lambda f: (worst[f["file"]], f["file"], order.get(f["severity"], 3), f["line"])
    )
    counts = {s: sum(1 for f in merged if f["severity"] == s) for s in ("high", "medium", "low")}
    both = sum(1 for f in merged if len(f["reported_by"]) > 1)

    L = [
        "# Local Security Audit",
        "",
        f"- **Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- **Scope:** `{root}`",
        f"- **Models:** " + ", ".join(f"{d}=`{m}`" for d, m in sorted(dims.items()))
        + f" (local Ollama @ `{stats['url']}`)",
        f"- **Coverage:** {stats['files']} files, {stats['chunks']} chunks, "
        f"{stats['calls']} model calls in {stats['elapsed']}",
        "",
        "> **Unverified machine output.** These findings come from local ~30B "
        "models and are triage, not proof. Expect false positives; confirm each "
        "one against the real file before acting. Absence of a finding is not "
        "evidence of absence.",
        "",
        "## Summary",
        "",
        "| Severity | Count |",
        "| -------- | ----- |",
        f"| High     | {counts['high']} |",
        f"| Medium   | {counts['medium']} |",
        f"| Low      | {counts['low']} |",
        f"| **Total**| **{len(merged)}** |",
        "",
        f"{both} finding(s) were independently reported by more than one "
        "dimension — a useful confidence signal.",
        "",
    ]

    if stats["failed"]:
        L += ["## NOT AUDITED", "",
              "These chunks failed and are **not** covered by this report:", ""]
        for item in stats["failed"][:60]:
            L.append(f"- `{item['file']}` lines {item['range']} ({item['dim']}): {item['error']}")
        if len(stats["failed"]) > 60:
            L.append(f"- ...and {len(stats['failed']) - 60} more")
        L.append("")

    L += ["## Findings", ""]
    if not merged:
        L.append("No findings reported.")
    else:
        cur = None
        for f in merged:
            if f["file"] != cur:
                cur = f["file"]
                L += [f"### `{cur}`", ""]
            tag = "+".join(f["reported_by"])
            conf = " **[CONFIRMED BY BOTH]**" if len(f["reported_by"]) > 1 else ""
            L.append(f"#### [{f['severity'].upper()}] {f['title']} — line {f['line']}{conf}")
            L.append(f"*Reported by: {tag} ({', '.join(f['models'])})*")
            if f.get("aliases"):
                L.append(f"*Also described as: {'; '.join(f['aliases'])}*")
            L.append("")
            L.append(f["description"])
            if f.get("fix"):
                L += ["", f"**Suggested fix:** {f['fix']}"]
            L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=".")
    ap.add_argument("--out", default="SECURITY-AUDIT.md")
    ap.add_argument("--chunk-lines", type=int, default=350)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--keep-alive", default="30m",
                    help="how long Ollama holds models in memory")
    ap.add_argument("--seed", type=int, default=42,
                    help="decoding seed; fixed by default so re-runs match")
    ap.add_argument("--limit", type=int, default=0, help="audit only first N files")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="also refuse if a remote backend is merely declared")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(a.path).resolve()
    if not root.exists():
        sys.exit(f"FATAL: {root} does not exist")

    url, dims, remote = load_config(strict=a.strict)
    print(f"[cfg] local-only verified — every model below is on {url}", flush=True)
    print("[cfg] dimensions: " + ", ".join(f"{d}={m}" for d, m in dims.items()),
          flush=True)
    if remote:
        print(
            f"[cfg] note: {', '.join(remote)} backend(s) are declared in "
            f"{ROUTES.name} but this\n"
            "      audit does not use them. They still apply to the MCP "
            "server's own tools —\n"
            "      check grading.backend there, which defaults to openrouter "
            "when unset."
        )

    files = discover(root, a.include_tests)
    if a.limit:
        files = files[: a.limit]

    jobs = []
    for p in files:
        rel = str(p.relative_to(root))
        for lo, hi, body in chunk(p, a.chunk_lines):
            for dim in dims:
                jobs.append((rel, lo, hi, body, dim))

    n_chunks = len({(j[0], j[1]) for j in jobs})
    print(f"[plan] {len(files)} files -> {n_chunks} chunks -> {len(jobs)} model calls",
          flush=True)
    # ~21s per call measured against warm 30B-class models. Yours will differ;
    # the live ETA below replaces this estimate once calls start completing.
    est = len(jobs) * 21 / max(a.concurrency, 1) / 60
    print(f"[plan] rough estimate: {est:.0f} min at concurrency {a.concurrency}",
          flush=True)
    if a.dry_run:
        for p in files[:40]:
            print("   ", p.relative_to(root))
        if len(files) > 40:
            print(f"    ...and {len(files) - 40} more")
        return 0

    ckpt = root / ".local-audit-checkpoint.jsonl"
    done: set[str] = set()
    findings: list[dict] = []
    failed: list[dict] = []
    if a.resume and ckpt.exists():
        for ln in ckpt.read_text().splitlines():
            try:
                rec = json.loads(ln)
            except json.JSONDecodeError:
                continue
            done.add(rec["key"])
            findings.extend(rec.get("findings", []))
            if rec.get("error"):
                failed.append(rec["failinfo"])
        print(f"[resume] skipping {len(done)} completed calls", flush=True)

    jobs = [j for j in jobs if f"{j[0]}:{j[1]}:{j[4]}" not in done]
    if not jobs:
        print("[done] nothing left to do", flush=True)

    # Load every model before timing starts. A model's first response after
    # load can differ from its later ones even under a fixed seed, so
    # warming up is what makes run 1 match run 2 rather than merely runs
    # 2..N matching each other. num_ctx must match the real calls or this
    # triggers a second load and defeats the purpose.
    if jobs:
        for model in sorted(set(dims.values())):
            t_w = time.time()
            _, werr = review(
                url, model, next(iter(dims)), "warmup", 1, 1, "pass", a.timeout, a.seed
            )
            state = werr if werr else "ok"
            print(f"[warmup] {model}: {state} ({time.time() - t_w:.0f}s)", flush=True)

    t0 = time.time()
    last_report = [t0]  # list so the worker closure can rebind it
    completed = 0
    total = len(jobs)
    lock = threading.Lock()
    fh = ckpt.open("a")

    def run(job) -> None:
        nonlocal completed
        rel, lo, hi, body, dim = job
        key = f"{rel}:{lo}:{dim}"
        fs, err = review(url, dims[dim], dim, rel, lo, hi, body, a.timeout,
                         a.seed, a.keep_alive)
        if err and "Timeout" in err:  # one retry at half size
            half = body.split("\n")
            fs, err = review(
                url, dims[dim], dim, rel, lo, lo + len(half) // 2,
                "\n".join(half[: len(half) // 2]), a.timeout, a.seed, a.keep_alive,
            )
        with lock:
            completed += 1
            rec = {"key": key, "findings": fs, "error": err}
            if err:
                rec["failinfo"] = {
                    "file": rel, "range": f"{lo}-{hi}", "dim": dim, "error": err
                }
                failed.append(rec["failinfo"])
            findings.extend(fs)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            now = time.time()
            # Time-based fallback alongside the call count. Ten calls can be
            # several minutes apart on a slow model or with large chunks, and
            # a long silence is indistinguishable from a hang.
            if (
                completed % 10 == 0
                or completed == total
                or now - last_report[0] >= HEARTBEAT_S
            ):
                last_report[0] = now
                el = now - t0
                rate = completed / max(el, 1)
                eta = (total - completed) / max(rate, 1e-6) / 60
                print(
                    f"[{completed}/{total}] {len(findings)} findings, "
                    f"{len(failed)} failed, ETA {eta:.0f} min",
                    flush=True,
                )

    if jobs:
        with ThreadPoolExecutor(max_workers=max(a.concurrency, 1)) as pool:
            list(pool.map(run, jobs))

    fh.close()
    elapsed = time.time() - t0
    stats = {
        "url": url, "files": len(files), "chunks": n_chunks,
        "calls": total, "failed": failed,
        "elapsed": f"{elapsed / 60:.0f}m {elapsed % 60:.0f}s",
    }

    out = root / a.out
    out.write_text(build_report(root, findings, stats, dims))
    (root / ".local-audit-findings.json").write_text(json.dumps(findings, indent=2))

    print(f"\n[done] {len(findings)} raw findings -> {out}", flush=True)
    print(f"[done] {len(failed)} failed calls", flush=True)
    if not failed and total:
        ckpt.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[abort] interrupted — rerun with --resume to continue", flush=True)
        sys.exit(130)
