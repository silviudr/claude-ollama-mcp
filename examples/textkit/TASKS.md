# textkit — task plan

A small Python library of **pure, self-contained text utilities**. Every task
below is sized to be one Ollama call via the `ollama-local` MCP server. The
orchestrator (Claude Code) only stitches outputs into files and runs tests.

## Philosophy reminder

- Every spec is paste-ready into one `local_*` tool — no project context needed.
- `local_draft_boilerplate` for mechanical scaffolds.
- `local_implement_small` for one pure function (≤ ~50 lines, stdlib only).
- If a row needs cross-file reasoning, it does **not** belong here.

## How to drive this

In a Claude Code session inside `~/projects/textkit/`, say *"work TASKS.md
top-to-bottom — for each row call the listed tool with the spec verbatim, then
mark it done."* Claude should route the call, paste back the output, and
check the box. Tail `~/.cache/ollama_mcp.jsonl` to watch the routing happen.

---

## Phase 1 — boilerplate (`local_draft_boilerplate`)

| # | File | Spec to pass to the tool | Done |
|---|------|---------------------------|------|
| 1 | `.gitignore` | "Generate a `.gitignore` for a Python project that uses `venv`, `pytest`, and the `src/` layout. Include `__pycache__`, `.venv`, `dist`, `build`, `*.egg-info`, `.pytest_cache`, `.coverage`, `.mypy_cache`, `.ruff_cache`. One file, no comments." | ☑ |
| 2 | `pyproject.toml` | "Generate a minimal `pyproject.toml` for a pure-Python library named `textkit`, version `0.1.0`, Python `>=3.11`, no runtime dependencies, build-system `setuptools`, package discovery via `src/` layout (package dir is `src/textkit`). Add an optional `dev` extra with `pytest>=8`. No tool configuration sections." | ☑ |
| 3 | `pytest.ini` | "Generate a `pytest.ini` that sets `testpaths = tests`, `pythonpath = src`, and `addopts = -q`." | ☑ |
| 4 | `.github/workflows/ci.yml` | "Generate a GitHub Actions workflow that runs on push and pull_request to `main`. One job on `ubuntu-latest`, matrix on Python `3.11` and `3.12`. Steps: checkout, setup-python, `pip install -e .[dev]`, `pytest`. Name the workflow `ci`." | ☑ |
| 5 | `README.md` | "Generate a short README skeleton for a Python library called `textkit` — a collection of small, pure-function text utilities. Include sections: title, one-line description, Install (`pip install -e .`), Usage (single import example placeholder), Development (`pip install -e .[dev]` and `pytest`). No badges, no emoji." | ☑ |

## Phase 2 — implementations (`local_implement_small`)

Each function lives in its own file under `src/textkit/`. Stdlib only. No
imports from other `textkit` modules. Pure: same input → same output, no I/O.

| #  | File | Spec to pass to the tool | Done |
|----|------|---------------------------|------|
| 6  | `src/textkit/slugify.py` | "Write a Python function `slugify(s: str) -> str`. Lowercase the input, strip Unicode accents (use `unicodedata.normalize('NFKD', ...)` and drop combining marks), replace any run of non-alphanumeric characters with a single `-`, and strip leading/trailing `-`. Stdlib only. Example: `slugify('Héllo, World!')` → `'hello-world'`." | ☑ |
| 7  | `src/textkit/dedent_trim.py` | "Write a Python function `dedent_trim(s: str) -> str`. Remove the longest common leading-whitespace prefix from all non-empty lines (use `textwrap.dedent`), then strip leading and trailing blank lines (but preserve internal blank lines and trailing whitespace within lines). Stdlib only." | ☑ |
| 8  | `src/textkit/wrap_by_width.py` | "Write a Python function `wrap_by_width(s: str, width: int) -> list[str]`. Word-wrap the input to lines of at most `width` characters, breaking on spaces only. Words longer than `width` go on their own line uncut. Returns an empty list for empty input. Raise `ValueError` if `width < 1`. Stdlib only." | ☑ |
| 9  | `src/textkit/redact_secrets.py` | "Write a Python function `redact_secrets(s: str) -> str`. Replace anything that looks like a secret with the literal string `***`. Match: (a) substrings of 24+ characters made of `[A-Za-z0-9_\\-]` (likely tokens/keys), (b) `sk-` followed by 20+ such characters (OpenAI-style), (c) values inside quoted JSON-style pairs whose key contains `password`, `secret`, `token`, or `api_key` (case-insensitive). Use `re`. Stdlib only." | ☑ |
| 10 | `src/textkit/table_to_markdown.py` | "Write a Python function `table_to_markdown(rows: list[dict]) -> str`. Render a list of dicts as a GitHub-Flavored Markdown table. Columns are the union of all keys, in first-seen order. Missing values render as empty string. Pipe characters in cell values are escaped as `\\|`. Returns `''` for an empty list. Stdlib only." | ☑ |
| 11 | `src/textkit/parse_kv.py` | "Write a Python function `parse_kv(s: str) -> dict[str, str]`. Parse a string of `key=value` pairs separated by whitespace. Values may be wrapped in double quotes to allow embedded spaces (use `shlex.split`). Bare tokens with no `=` are skipped. Later keys override earlier ones. Stdlib only. Example: `parse_kv('a=1 b=\"two words\" c=3')` → `{'a': '1', 'b': 'two words', 'c': '3'}`." | ☑ |
| 12 | `src/textkit/human_bytes.py` | "Write a Python function `human_bytes(n: int) -> str`. Format a non-negative integer byte count using binary units (1024-based): `B`, `KiB`, `MiB`, `GiB`, `TiB`, `PiB`. One decimal place for non-`B` units, no decimal for `B`. Examples: `human_bytes(0)` → `'0 B'`, `human_bytes(1536)` → `'1.5 KiB'`. Raise `ValueError` for negative input. Stdlib only." | ☑ |
| 13 | `src/textkit/truncate_middle.py` | "Write a Python function `truncate_middle(s: str, max_len: int) -> str`. If `len(s) <= max_len`, return `s` unchanged. Otherwise return `s` shortened to `max_len` characters by removing the middle and inserting `'…'` (single ellipsis char). Keep roughly equal halves of the original on each side. Raise `ValueError` if `max_len < 1`. Stdlib only." | ☑ |
| 14 | `src/textkit/count_lines.py` | "Write a Python function `count_lines(s: str) -> int`. Count the number of lines in the string. Empty string returns `0`. A trailing newline does not add an extra empty line (so `'a\\nb'` and `'a\\nb\\n'` both return `2`). Stdlib only." | ☑ |
| 15 | `src/textkit/to_snake.py` | "Write a Python function `to_snake(s: str) -> str`. Convert `camelCase`, `PascalCase`, or `kebab-case` (and any mix, including space-separated) to `snake_case`. Insert `_` before each uppercase letter that follows a lowercase letter or digit, lowercase the result, and replace runs of `-` or whitespace with a single `_`. Collapse repeated underscores and strip leading/trailing `_`. Use `re`. Stdlib only. Examples: `to_snake('XMLHttpRequest')` → `'xml_http_request'`, `to_snake('my-var name')` → `'my_var_name'`." | ☑ |

## Phase 3 — tests (`local_draft_boilerplate`)

One test file per function, each a self-contained spec. Done as a batch once
Phase 2 is complete so the orchestrator can run `pytest` end-to-end.

| #  | File | Spec to pass to the tool | Done |
|----|------|---------------------------|------|
| 16 | `tests/test_slugify.py` | "Write a pytest test file for a function `slugify(s: str) -> str` imported from `textkit.slugify`. Cases: `'Héllo, World!'` → `'hello-world'`; `'  spaces   here  '` → `'spaces-here'`; `'---!!!---'` → `''`; `'AlreadySlug'` → `'alreadyslug'`; empty string → `''`. One `pytest.mark.parametrize` block." | ☑ |
| 17 | `tests/test_dedent_trim.py` | "Write a pytest test file for `dedent_trim(s: str) -> str` from `textkit.dedent_trim`. Cases: input `'\\n    a\\n    b\\n\\n'` → `'a\\nb'`; input `'\\n  a\\n    b\\n'` → `'a\\n  b'`; empty string → `''`; all blank lines → `''`. One `parametrize` block." | ☑ |
| 18 | `tests/test_wrap_by_width.py` | "Write a pytest test file for `wrap_by_width(s, width)` from `textkit.wrap_by_width`. Cover: normal wrap of `'one two three four'` at width 8 → `['one two', 'three', 'four']`; oversize word kept whole; empty input → `[]`; `width=0` raises `ValueError`. Use `parametrize` for the value cases and a separate test for the raise." | ☑ |
| 19 | `tests/test_human_bytes.py` | "Write a pytest test file for `human_bytes(n)` from `textkit.human_bytes`. Cases: `0` → `'0 B'`, `512` → `'512 B'`, `1024` → `'1.0 KiB'`, `1536` → `'1.5 KiB'`, `1048576` → `'1.0 MiB'`. Negative input raises `ValueError`. Use `parametrize`." | ☑ |
| 20 | `tests/test_to_snake.py` | "Write a pytest test file for `to_snake(s)` from `textkit.to_snake`. Cases: `'XMLHttpRequest'` → `'xml_http_request'`; `'camelCase'` → `'camel_case'`; `'my-var name'` → `'my_var_name'`; `'already_snake'` → `'already_snake'`; `''` → `''`. One `parametrize` block." | ☑ |

(Tests for the remaining Phase 2 functions can be added in the same shape if
the routing rate looks healthy — kept off the initial plan to keep the test
focused.)

## Phase 4 — orchestrator glue (no Ollama)

These are *deliberately* not delegated — they need repo context.

- Create `src/textkit/__init__.py` that re-exports every function from Phase 2.
- Create the venv, `pip install -e .[dev]`, run `pytest`, fix any failures.
- If any Phase-2 function fails its test, **fix it directly** (don't re-route)
  — the local model already had its shot, and iterating on it from here would
  destroy the routing-rate signal.

---

## What success looks like

After working this list, check `~/.cache/ollama_mcp.jsonl`:

```bash
jq -s 'group_by(.tool) | map({tool: .[0].tool, n: length, ok: map(select(.ok)) | length})' \
  ~/.cache/ollama_mcp.jsonl
```

You should see roughly:
- `local_draft_boilerplate`: 5 (Phase 1) + 5 (Phase 3) = 10 calls
- `local_implement_small`: 10 calls
- Failure rate near zero
- Routing rate (local calls / total tool calls in the session) well above 50%

If the routing rate is low, the docstrings still aren't pulling weight — edit
`~/projects/ollama-mcp/ollama_mcp.py` and retry on the next phase.
