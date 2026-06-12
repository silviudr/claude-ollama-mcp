"""Model and backend routing — resolve which backend+model to use per tool."""

from __future__ import annotations

import json

from .backends import Backend, OllamaBackend, OpenRouterBackend
from .config import MODEL, OLLAMA_URL, ROUTES_CONFIG_PATH

_BACKEND_BUILDERS = {
    "ollama": lambda cfg: OllamaBackend(
        url=cfg.get("url", OLLAMA_URL),
        default_model=cfg.get("default_model", MODEL),
    ),
    "openrouter": lambda cfg: OpenRouterBackend(
        api_key_env=cfg.get("api_key_env", "OPENROUTER_API_KEY"),
        default_model=cfg.get("default_model", "google/gemma-3-27b-it:free"),
        url=cfg.get("url", "https://openrouter.ai/api/v1"),
        api_key=cfg.get("api_key"),
    ),
}


def _load_raw() -> dict:
    if not ROUTES_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(ROUTES_CONFIG_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _build_backend(name: str, cfg: dict) -> Backend:
    backend_type = cfg.get("type", name)
    builder = _BACKEND_BUILDERS.get(backend_type)
    if builder is None:
        raise ValueError(f"Unknown backend type: {backend_type}")
    return builder(cfg)


def _parse_config(raw: dict) -> tuple[dict[str, Backend], str, dict[str, tuple[str, str]]]:
    """Parse config into (backends, default_backend_name, routes).

    Routes map tool_name -> (backend_name, model).
    """
    backends: dict[str, Backend] = {}
    routes: dict[str, tuple[str, str]] = {}

    if "backends" in raw:
        default_name = raw.get("default_backend", "ollama")

        for name, cfg in raw.get("backends", {}).items():
            backends[name] = _build_backend(name, cfg)

        for tool, route in raw.get("routes", {}).items():
            if isinstance(route, str):
                routes[tool] = (default_name, route)
            elif isinstance(route, dict):
                bname = route.get("backend", default_name)
                routes[tool] = (bname, route.get("model", ""))
    else:
        # Legacy format: pure Ollama
        default_name = "ollama"
        default_model = raw.get("default", MODEL)
        backends["ollama"] = OllamaBackend(url=OLLAMA_URL, default_model=default_model)

        for tool, model in raw.get("routes", {}).items():
            if isinstance(model, str):
                routes[tool] = ("ollama", model)

    # Ensure default Ollama backend always exists
    if "ollama" not in backends:
        backends["ollama"] = OllamaBackend(url=OLLAMA_URL, default_model=MODEL)

    return backends, default_name, routes


def resolve(tool_name: str) -> tuple[Backend, str]:
    """Resolve which backend and model to use for a tool.

    Returns (backend_instance, model_name).
    """
    raw = _load_raw()
    backends, default_name, routes = _parse_config(raw)

    if tool_name in routes:
        backend_name, model = routes[tool_name]
        backend = backends.get(backend_name, backends[default_name])
        return backend, model or backend.default_model

    default_backend = backends[default_name]
    return default_backend, default_backend.default_model


def resolve_model(tool_name: str) -> str:
    """Legacy API: returns just the model name."""
    _, model = resolve(tool_name)
    return model


def get_backends() -> dict[str, Backend]:
    """Return all configured backend instances."""
    raw = _load_raw()
    backends, _, _ = _parse_config(raw)
    return backends


def get_routes_info() -> str:
    """Format current routing config for display."""
    raw = _load_raw()
    backends, default_name, routes = _parse_config(raw)

    lines = ["Backends:"]
    for name in sorted(backends):
        backend = backends[name]
        default_marker = " (default)" if name == default_name else ""
        if isinstance(backend, OllamaBackend):
            lines.append(
                f"  {name}{default_marker}: {backend.url}, "
                f"model: {backend.default_model}"
            )
        elif isinstance(backend, OpenRouterBackend):
            lines.append(
                f"  {name}{default_marker}: openrouter.ai, "
                f"model: {backend.default_model}"
            )

    if not routes:
        db = backends[default_name]
        lines.append("")
        lines.append(
            f"No per-tool routes. All tools use: "
            f"{default_name} / {db.default_model}"
        )
    else:
        lines.append("")
        lines.append("Per-tool routes:")
        for tool in sorted(routes):
            backend_name, model = routes[tool]
            lines.append(f"  {tool} -> {backend_name} / {model}")
        db = backends[default_name]
        lines.append("")
        lines.append(f"Unrouted tools use: {default_name} / {db.default_model}")

    return "\n".join(lines)
