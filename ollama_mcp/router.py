"""Model routing — resolve which Ollama model to use per tool."""

from __future__ import annotations

import json

from .config import MODEL, ROUTES_CONFIG_PATH


def _load_routes() -> dict:
    if not ROUTES_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(ROUTES_CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_model(tool_name: str) -> str:
    config = _load_routes()
    routes = config.get("routes", {})
    if tool_name in routes:
        return routes[tool_name]
    return config.get("default", MODEL)


def get_routes_info() -> str:
    config = _load_routes()
    routes = config.get("routes", {})
    default = config.get("default", MODEL)

    if not routes:
        return (
            f"No routing config found. All tools use the default model: {MODEL}\n"
            f"To configure, create {ROUTES_CONFIG_PATH} with:\n"
            "{\n"
            '  "default": "gemma4-32k",\n'
            '  "routes": {\n'
            '    "local_review_diff": "deepseek-coder",\n'
            '    "local_generate_tests": "qwen2.5-coder"\n'
            "  }\n"
            "}"
        )

    lines = [f"Default model: {default}", "", "Per-tool routes:"]
    for tool, model in sorted(routes.items()):
        lines.append(f"  {tool} → {model}")
    return "\n".join(lines)
