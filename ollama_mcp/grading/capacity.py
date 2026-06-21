"""Query Ollama for GPU capacity to determine if local grading is feasible."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger("ollama_mcp.grading")


async def get_running_models(ollama_url: str) -> list[dict]:
    """Query Ollama /api/ps for currently loaded models.

    Returns a list of dicts with keys: name, size, size_vram, expires_at.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            r = await c.get(f"{ollama_url}/api/ps")
            r.raise_for_status()
            data = r.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return []
    return data.get("models", [])


async def check_capacity(
    ollama_url: str, grading_model: str
) -> dict:
    """Check whether the local Ollama instance can fit a grading model.

    Returns:
        {
            "can_grade_locally": bool,
            "running_models": [{"name": ..., "size_vram": ...}],
            "grading_model": str,
            "reason": str,
        }
    """
    running = await get_running_models(ollama_url)

    if not running:
        return {
            "can_grade_locally": True,
            "running_models": [],
            "grading_model": grading_model,
            "reason": "no models loaded — grading model can load freely",
        }

    loaded_names = [m.get("name", "") for m in running]
    total_vram = sum(m.get("size_vram", 0) for m in running)

    if any(grading_model in name for name in loaded_names):
        return {
            "can_grade_locally": True,
            "running_models": _summarise(running),
            "grading_model": grading_model,
            "reason": "grading model already loaded",
        }

    if len(running) >= 2:
        return {
            "can_grade_locally": False,
            "running_models": _summarise(running),
            "grading_model": grading_model,
            "reason": (
                f"{len(running)} models already loaded "
                f"(~{total_vram / 1e9:.1f} GB VRAM) — "
                f"loading '{grading_model}' risks eviction"
            ),
        }

    return {
        "can_grade_locally": True,
        "running_models": _summarise(running),
        "grading_model": grading_model,
        "reason": "1 model loaded — may fit a second for grading",
        "warning": (
            "If your GPU has limited VRAM, loading a second model may "
            "evict the task model. Consider using OpenRouter for grading."
        ),
    }


def suggest_grading_config(capacity: dict) -> str | None:
    """Return a human-readable suggestion if local grading won't work."""
    if capacity["can_grade_locally"]:
        return None

    return (
        f"Local grading with '{capacity['grading_model']}' is not recommended: "
        f"{capacity['reason']}.\n\n"
        "Suggested fix — use OpenRouter (free) for grading:\n"
        '  "grading": {\n'
        '    "enabled": true,\n'
        '    "backend": "openrouter",\n'
        '    "model": "google/gemma-4-31b-it:free"\n'
        "  }\n\n"
        "Or set OLLAMA_MCP_GRADING=0 to disable grading entirely."
    )


def _summarise(models: list[dict]) -> list[dict]:
    return [
        {
            "name": m.get("name", "?"),
            "size_vram_gb": round(m.get("size_vram", 0) / 1e9, 1),
        }
        for m in models
    ]
