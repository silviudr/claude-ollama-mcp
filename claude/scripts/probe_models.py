#!/usr/bin/env python3
"""Probe local Ollama models for audit suitability.

Model behaviour varies enormously and silently. On the machine this was
developed against, four of six models returned an EMPTY STRING under Ollama's
`format` JSON-schema parameter — and an empty response parses as "no
findings", so a completely broken scan is indistinguishable from a clean one.
Another model returned three different answers to an identical prompt.

Rather than trust a table written for someone else's hardware, this probes
YOUR models and prints a routes.json snippet naming the ones that passed.

Usage:
    probe_models.py [--url URL] [--models a,b] [--timeout 300]

Stdlib only — no install required.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from local_audit import extract_json
except ImportError:
    sys.exit(
        "FATAL: local_audit.py must sit beside this script "
        "(both are installed into ~/.claude/scripts by claude/install.sh)."
    )

ROUTES = Path(
    os.environ.get(
        "OLLAMA_MCP_ROUTES", Path.home() / ".config" / "ollama_mcp" / "routes.json"
    )
)

# Three unmissable defects. A model that cannot find these cannot audit code.
CANARY = '''   10| API_TOKEN = "sk-live-4f9a2b7c1e8d3a6f"
   15|     cur.execute("SELECT * FROM users WHERE name = '" + username + "'")
   23|     out = subprocess.check_output("ping -c 1 " + host, shell=True)'''
EXPECT = {
    "secret": ("token", "secret", "hardcod", "credential"),
    "sqli": ("sql", "injection", "concaten", "parameter"),
    "cmdi": ("command", "shell", "subprocess", "injection"),
}

JSON_SYSTEM = (
    "You are a security code auditor. Report only concrete defects.\n"
    "Respond with ONLY a JSON object, no prose and no markdown fence:\n"
    '{"findings":[{"severity":"high|medium|low","title":"...","line":<int>,'
    '"description":"..."}]}\n'
)
PROMPT = f"FILE: app.py\nFind security issues.\n\n{CANARY}\n"

SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "severity": {"type": "string"},
                    "title": {"type": "string"},
                    "line": {"type": "integer"},
                    "description": {"type": "string"},
                },
                "required": ["severity", "title", "line", "description"],
            },
        }
    },
    "required": ["findings"],
}

DETERMINISTIC_OPTS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "seed": 42,
    "num_ctx": 16384,
    "num_predict": 3500,
}


def call(url: str, model: str, timeout: float, system: str,
         opts: dict, fmt: dict | None = None) -> tuple[str, dict, str | None]:
    payload = {
        "model": model, "prompt": PROMPT, "system": system,
        "stream": False, "keep_alive": "30m", "options": opts,
    }
    if fmt is not None:
        payload["format"] = fmt
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        return "", {}, f"HTTP {e.code}"
    except (urllib.error.URLError, socket.timeout, TimeoutError) as e:
        return "", {}, f"unreachable/timeout ({getattr(e, 'reason', e)})"
    except ValueError as e:
        return "", {}, f"bad body ({e})"
    if "error" in data:
        return "", {}, str(data["error"])[:80]
    return data.get("response") or "", data, None


def coverage(findings: list[dict]) -> set[str]:
    """Which of the three planted defects the model actually described."""
    blob = " ".join(
        f"{f.get('title', '')} {f.get('description', '')}" for f in findings
    ).lower()
    return {name for name, kws in EXPECT.items() if any(k in blob for k in kws)}


def probe(url: str, model: str, timeout: float) -> dict:
    r: dict = {"model": model}
    t0 = time.time()

    # 0. Warm up and discard. A model's first response after loading can differ
    #    from its later ones even under a fixed seed, so comparing call 1
    #    against call 2 on a cold host reports every model as
    #    non-deterministic — including ones that are perfectly reproducible
    #    once resident.
    _, _, warm_err = call(url, model, timeout, JSON_SYSTEM, DETERMINISTIC_OPTS)
    if warm_err:
        r.update(broken=warm_err, prompt_json=False, schema=False,
                 deterministic=False, found=set(), elapsed=time.time() - t0)
        return r

    # 1. Prompt-driven JSON — the mode the audit actually uses.
    text, meta, err = call(url, model, timeout, JSON_SYSTEM, DETERMINISTIC_OPTS)
    r["elapsed"] = time.time() - t0
    if err:
        r.update(broken=err, prompt_json=False, schema=False,
                 deterministic=False, found=set())
        return r
    parsed = extract_json(text)
    findings = (parsed or {}).get("findings") or []
    r["prompt_json"] = parsed is not None
    r["found"] = coverage(findings)
    r["n_findings"] = len(findings)
    dur = (meta.get("eval_duration") or 0) / 1e9
    r["tok_s"] = (meta.get("eval_count") or 0) / dur if dur > 0 else 0.0

    # 2. Determinism — identical request twice, byte-compare.
    text2, _, err2 = call(url, model, timeout, JSON_SYSTEM, DETERMINISTIC_OPTS)
    r["deterministic"] = (not err2) and text2 == text

    # 3. Ollama's schema-constrained decoding. Reported separately because
    #    failure here is silent: an empty string looks like a clean file.
    stext, _, serr = call(url, model, timeout, JSON_SYSTEM,
                          DETERMINISTIC_OPTS, fmt=SCHEMA)
    if serr:
        r["schema"] = f"error: {serr}"
    elif not stext.strip():
        r["schema"] = "EMPTY"
    else:
        try:
            json.loads(stext)
            r["schema"] = "ok"
        except json.JSONDecodeError:
            r["schema"] = "bad json"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="")
    ap.add_argument("--models", default="", help="comma-separated subset")
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()

    url = a.url
    if not url and ROUTES.exists():
        try:
            cfg = json.loads(ROUTES.read_text())
            url = cfg.get("backends", {}).get("ollama", {}).get("url", "")
        except (json.JSONDecodeError, OSError):
            pass
    url = (url or "http://localhost:11434").rstrip("/")
    print(f"Probing {url}\n")

    if a.models:
        models = [m.strip() for m in a.models.split(",") if m.strip()]
    else:
        try:
            with urllib.request.urlopen(f"{url}/api/tags", timeout=15) as r:
                models = sorted(
                    m["name"] for m in json.loads(r.read().decode())["models"]
                )
        except Exception as e:  # noqa: BLE001 - any failure here is fatal anyway
            sys.exit(f"FATAL: cannot list models at {url}: {e}")
    if not models:
        sys.exit("FATAL: no models installed.")

    print(f"{len(models)} model(s), 4 calls each. The first is a discarded "
          f"warmup,\nso an unloaded model costs one extra load.\n")
    results = []
    for m in models:
        print(f"  probing {m} ...", end="", flush=True)
        res = probe(url, m, a.timeout)
        results.append(res)
        print(f" {res['elapsed']:.0f}s")

    print("\n" + "=" * 78)
    print(f"{'MODEL':<24}{'FINDS':<8}{'JSON':<7}{'DETERM':<9}{'SCHEMA':<11}{'TOK/S':>6}")
    print("=" * 78)
    for r in results:
        if r.get("broken"):
            print(f"{r['model'][:23]:<24}{'BROKEN — ' + r['broken'][:40]}")
            continue
        print(
            f"{r['model'][:23]:<24}"
            f"{len(r['found'])}/3     "
            f"{'yes' if r['prompt_json'] else 'NO ':<7}"
            f"{'yes' if r['deterministic'] else 'NO ':<9}"
            f"{str(r['schema'])[:10]:<11}"
            f"{r['tok_s']:>6.0f}"
        )
    print("=" * 78)

    usable = [
        r for r in results
        if not r.get("broken") and r["prompt_json"] and len(r["found"]) >= 2
    ]
    usable.sort(key=lambda r: (-len(r["found"]), -r["deterministic"], -r["tok_s"]))

    print("\nNotes")
    print("  FINDS  : planted defects described (secret / SQLi / command inj).")
    print("           Below 2/3, the model is not worth an audit slot.")
    print("  JSON   : usable via prompt-driven JSON — required.")
    print("  DETERM : identical output for an identical request. 'NO' means")
    print("           re-auditing unchanged code yields a different report.")
    print("  SCHEMA : Ollama's `format` parameter. 'EMPTY' means it silently")
    print("           returns nothing — which reads as a clean file. The audit")
    print("           never uses this mode; shown so you know not to either.")

    if not usable:
        print("\nNo model scored 2/3 or better. Pull a stronger one — a ~30B")
        print("coder or reasoning model is the realistic floor for this task.")
        return 1

    if any(not r["deterministic"] for r in usable):
        print("\nWARNING: some usable models are non-deterministic even at")
        print("temperature 0 with a fixed seed. Prefer a deterministic one for")
        print("the audit, or expect findings to shift between identical runs.")

    sec = usable[0]["model"]
    cor = usable[1]["model"] if len(usable) > 1 else sec
    print("\nSuggested routes.json snippet (two dimensions, two models):\n")
    print(json.dumps({
        "swarm": {
            "enabled": True,
            "concurrency": {"ollama": 2},
            "review_dimensions": {
                "local_review_diff": {"security": sec, "correctness": cor}
            },
        }
    }, indent=2))
    if len(usable) == 1:
        print("\nOnly one model qualified, so both dimensions use it. Two")
        print("DIFFERENT models is strongly preferred: their mistakes are less")
        print("correlated, so agreement between them actually means something.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
