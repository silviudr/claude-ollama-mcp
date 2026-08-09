# Local model findings

Measured while building the local-only audit pipeline (`claude/`). Everything
here was observed, not assumed — but it was observed on **one machine with six
models**, so treat it as a warning about what to check, not a table to copy.
`claude/scripts/probe_models.py` re-derives all of it against your own models.

Test rig: Ollama 0.32.5, six models in the 3B–36B range (`qwen3-coder:30b`,
`nemotron-3-nano:30b`, `qwen3.5:35b`, `qwen3.6:35b`, `gemma4:31b`,
`llama3.2:3b`), ~78 tok/s for the 30B-class models, ~18s cold load.

---

## 1. Ollama's JSON-schema mode is unusable on most models, and fails silently

Ollama's `format` parameter accepts a JSON schema and constrains decoding to
match it. It looks like the obvious way to get structured findings. Results:

| Model | `format` schema |
| ----- | --------------- |
| `qwen3-coder:30b` | works |
| `llama3.2:3b` | works |
| `gemma4:31b` | invalid JSON, and 158s for one call |
| `qwen3.6:35b` | **empty string** |
| `qwen3.5:35b` | **empty string** |
| `nemotron-3-nano:30b` | **empty string** |

Four of six produce nothing usable. The three reasoning models emit thinking
tokens that constrained decoding discards, leaving an empty response — one
burned 310 tokens and returned `""`.

**The dangerous part is not the failure, it is the shape of the failure.** An
empty response parses as "no findings", which is indistinguishable from a clean
file. The first working version of the audit reported seven findings and looked
fine; every one came from a single model, and the security reviewer had been
silently contributing nothing at all.

**What to do instead:** ask for JSON in the prompt and parse tolerantly — strip
`<think>` blocks and markdown fences, repair trailing commas, and salvage
individual balanced `{...}` objects when a reasoning model is truncated
mid-array. That works on all six. See `extract_json` in
`claude/scripts/local_audit.py`.

**Corollary:** never let an unparseable response count as "clean". The sweeper
records it as `NOT AUDITED` with a reason. A coverage gap you can see is worth
far more than a clean bill of health you cannot trust.

## 2. Determinism needs `temperature: 0`, `top_k: 1`, a fixed `seed`, and no reload

`nemotron-3-nano:30b` returned **three different answers to a byte-identical
prompt** at `temperature: 0.1`. `qwen3-coder:30b` was already stable. With
`temperature: 0`, `top_k: 1`, and `seed: 42`, both became byte-identical.

A second effect survives that fix. Output is identical while a model stays
loaded, but each fresh load settles into a *slightly different* stable state —
plausibly floating-point reduction order shifting with memory layout. Measured
across a reload: 12 of 14 findings identical, the two differences being one
title reworded (`"Insecure Deserialization Vulnerability"` vs `"Insecure
Deserialization"`) and one severity notch (low vs medium). **No finding
appeared or disappeared.** Pass `keep_alive` to hold models in memory when two
runs need to be comparable.

Worth stating plainly: pinning the seed makes wrong findings *reproducibly*
wrong. Determinism buys comparable reports, not correct ones.

## 3. Pin `num_ctx` — the default reservation is enormous

Left at its default, `qwen3-coder:30b` reserved **45.6 GB** of KV cache. At
`num_ctx: 16384` it used **20.2 GB** for identical work. Two models went from
66 GB co-resident to 44 GB, which is the difference between running two review
dimensions and three.

Size it deliberately: a 350-line chunk is roughly 5–6K tokens, plus the system
prompt, plus `num_predict`. 16384 leaves comfortable headroom. Too small
truncates the prompt silently.

## 4. Detection is good, with a consistent blind spot

Against a file with eight planted vulnerabilities, two dimensions
(`security` = nemotron, `correctness` = qwen3-coder) found seven: SQL
injection, command injection, path traversal, pickle deserialization, MD5
password hashing, a hardcoded token, and `debug=True` on `0.0.0.0`.

It never found the eighth: **timing-unsafe token comparison** (`if supplied ==
API_TOKEN`). Not once across many runs. Assume non-constant-time comparison is
invisible to this pipeline and check it another way.

The two dimensions were genuinely complementary rather than redundant — in one
run the security model missed path traversal and the correctness model caught
it. Running both is not belt-and-braces; each covers real ground the other
misses.

## 5. It hallucinates confidently, so verify before believing

On real code the pipeline reported a HIGH "SQL Injection Vulnerability via Tool
Parameter". The cited line was:

```python
where = "WHERE g.tool = ?" if tool else ""
```

A correctly parameterized query. The f-string interpolates only the static
clause; the user value travels through `params`. Complete false positive,
stated with total confidence and an accurate line number.

Every HIGH needs the cited file and line opened and read before it is believed.
The slash command instructs Claude to do exactly that and to report
non-matching claims as false positives rather than passing them along.

## 6. Cross-model agreement beats multi-pass voting

The tempting robustness trick is to run the same chunk N times and keep
findings that recur. It does not work here, for two reasons.

First, mechanically: at `temperature: 0` with `top_k: 1`, decoding is greedy.
N passes return N identical answers — N times the cost, zero information. You
would have to raise the temperature, giving up the determinism from §2.

Second, and more fundamental: **two samples from one model make correlated
mistakes.** The false positive in §5 came from `correctness` alone. Multi-pass
on that same model would very likely have reproduced it every time and reported
it as "confirmed 3/3". The second model simply did not agree, so it never
earned the `CONFIRMED BY BOTH` tag.

Two different model families reviewing the same chunk is strictly better per
unit of compute. If you want more, add a third *dimension* on a third model for
2-of-3 agreement — not more passes of the same one.

## 7. Operational gotchas

- **`routes.json` is re-read on every tool call.** Config edits take effect
  immediately with no MCP server restart. Environment variables in the server's
  registration are the exception — those are fixed at process spawn.
- **Adaptive routing scores are keyed by model name.** Renaming or replacing a
  model orphans its entire history; every new candidate restarts at zero and
  needs `min_samples` before it influences anything. Not broken, just cold —
  but silent about it.
- **Removing a backend beats unrouting it.** The router resolves backends by
  name. With no `openrouter` entry defined, a stray `openrouter/...` candidate
  has nothing to resolve to and fails loudly instead of quietly billing you.
- **Grading is no longer free.** On a cloud free tier, grading every call cost
  nothing. Locally it is a second inference per graded call, roughly doubling
  the cost of the tool being graded. Sample it.
- **Check parallelism before assuming you need to configure it.** Two models
  stayed co-resident and same-model requests genuinely overlapped (708ms wall
  for two requests against a 2218ms serial baseline), so `OLLAMA_NUM_PARALLEL`
  and `OLLAMA_MAX_LOADED_MODELS` were already adequate at their defaults.

## 8. Method

Each claim above came from the same loop: form the hypothesis, run it several
times, compare exact bytes rather than impressions. Two habits did most of the
work.

**Plant known defects.** A canary file with deliberate vulnerabilities converts
"the output looks plausible" into "it found 7 of 8, and always misses this
specific one". The silent-security-model bug in §1 was invisible until findings
were counted per dimension.

**Hash the results and re-run.** Comparing SHA-256 of a normalized finding set
across runs is what separated "the sampler is noisy" (§2, fixable) from "each
model load has its own stable state" (§2, not fixable client-side). Eyeballing
two reports would have shown neither.
