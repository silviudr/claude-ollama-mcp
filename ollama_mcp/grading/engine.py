"""Grading engine — async orchestration of heuristic and semantic grading."""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time

from ..backends import Backend, OllamaBackend, OpenRouterBackend
from ..config import (
    GRADING_ENABLED,
    GRADING_SAMPLE_RATE,
    OLLAMA_URL,
    ROUTES_CONFIG_PATH,
)
from ..storage import log_grade
from .capacity import check_capacity, suggest_grading_config
from .heuristics import run_heuristics
from .semantic import grade_output

logger = logging.getLogger("ollama_mcp.grading")

_capacity_checked = False
_capacity_warning: str | None = None


def _load_grading_config() -> dict:
    """Load the grading section from routes.json."""
    if not ROUTES_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(ROUTES_CONFIG_PATH.read_text())
        return data.get("grading", {}) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _load_full_config() -> dict:
    """Load the entire routes.json."""
    if not ROUTES_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(ROUTES_CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _build_grading_backend(config: dict) -> tuple[Backend, str] | None:
    """Build the backend instance for grading based on config.

    Returns (backend, model) or None if grading backend can't be built.
    """
    backend_name = config.get("backend", "openrouter")
    model = config.get("model", "")

    if backend_name == "openrouter":
        api_key = config.get("api_key")
        api_key_env = config.get("api_key_env", "OPENROUTER_API_KEY")
        if not api_key:
            full = _load_full_config()
            api_key = full.get("backends", {}).get("openrouter", {}).get("api_key")
        try:
            backend = OpenRouterBackend(
                api_key_env=api_key_env,
                default_model=model or "google/gemma-3-27b-it:free",
                api_key=api_key,
            )
            return backend, model or backend.default_model
        except Exception:
            return None

    if backend_name == "ollama":
        url = config.get("url", OLLAMA_URL)
        backend = OllamaBackend(
            url=url,
            default_model=model or "gemma4-32k",
        )
        return backend, model or backend.default_model

    return None


async def _check_capacity_once(config: dict) -> None:
    """Run capacity check on first grading call if using local Ollama."""
    global _capacity_checked, _capacity_warning
    if _capacity_checked:
        return
    _capacity_checked = True

    backend_name = config.get("backend", "openrouter")
    if backend_name != "ollama":
        return

    model = config.get("model", "gemma4-32k")
    url = config.get("url", OLLAMA_URL)
    cap = await check_capacity(url, model)
    suggestion = suggest_grading_config(cap)
    if suggestion:
        _capacity_warning = suggestion
        logger.warning("Grading capacity issue: %s", suggestion)


async def _grade(
    tool_name: str,
    input_text: str,
    output_text: str,
    call_id: int | None,
) -> None:
    """Run heuristic + (sampled) semantic grading and store results."""
    config = _load_grading_config()
    enabled = config.get("enabled", GRADING_ENABLED)
    if not enabled:
        return

    per_tool = config.get("tool_sample_rates", {})
    sample_rate = per_tool.get(tool_name, config.get("sample_rate", GRADING_SAMPLE_RATE))

    await _check_capacity_once(config)

    # Always run heuristics
    ts = time.time()
    h_results = run_heuristics(tool_name, input_text, output_text)
    for r in h_results:
        log_grade({
            "ts": ts,
            "call_id": call_id,
            "tool": tool_name,
            "grade_type": "heuristic",
            "checker": r["checker"],
            "score": r["score"],
            "passed": r["passed"],
            "details": r.get("details"),
            "grader_model": None,
            "grader_backend": None,
            "grader_ms": 0,
        })

    # Semantic grading at sample_rate
    if random.random() >= sample_rate:
        return

    backend_and_model = _build_grading_backend(config)
    if backend_and_model is None:
        return

    backend, model = backend_and_model
    result = await grade_output(backend, tool_name, input_text, output_text, model=model)
    log_grade({
        "ts": time.time(),
        "call_id": call_id,
        "tool": tool_name,
        "grade_type": "semantic",
        **result,
    })


def schedule_grading(
    tool_name: str,
    input_text: str,
    output_text: str,
    call_id: int | None,
) -> None:
    """Fire-and-forget: schedule grading as a background async task."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return

    async def _safe_grade():
        try:
            await _grade(tool_name, input_text, output_text, call_id)
        except Exception as exc:
            logger.debug("grading failed (non-fatal): %s", exc)

    loop.create_task(_safe_grade())


def get_capacity_warning() -> str | None:
    """Return the last capacity warning, if any."""
    return _capacity_warning
