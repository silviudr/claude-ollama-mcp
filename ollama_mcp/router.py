"""Model and backend routing — resolve which backend+model to use per tool."""

from __future__ import annotations

import json
import random

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


class AdaptiveConfig:
    """Parsed adaptive routing configuration."""

    def __init__(
        self,
        enabled: bool = False,
        min_samples: int = 10,
        quality_weight: float = 0.7,
        explore_rate: float = 0.1,
        recency_window: int = 100,
        candidates: dict[str, list[str]] | None = None,
    ):
        self.enabled = enabled
        self.min_samples = min_samples
        self.quality_weight = quality_weight
        self.explore_rate = explore_rate
        self.recency_window = recency_window
        self.candidates = candidates or {}


def _parse_adaptive(raw: dict) -> AdaptiveConfig:
    section = raw.get("adaptive", {})
    if not section or not section.get("enabled", False):
        return AdaptiveConfig()
    return AdaptiveConfig(
        enabled=True,
        min_samples=section.get("min_samples", 10),
        quality_weight=section.get("quality_weight", 0.7),
        explore_rate=section.get("explore_rate", 0.1),
        recency_window=section.get("recency_window", 100),
        candidates=section.get("candidates", {}),
    )


def _resolve_candidate(
    candidate: str,
    backends: dict[str, Backend],
    default_name: str,
) -> tuple[str, str]:
    """Parse a candidate string into (backend_name, model).

    Bare names use the default backend. Prefixed names (e.g.
    "openrouter/google/gemma-3-27b-it:free") use the prefix as the backend.
    """
    for bname in backends:
        prefix = bname + "/"
        if candidate.startswith(prefix):
            return bname, candidate[len(prefix):]
    return default_name, candidate


def _pick_adaptive(
    tool_name: str,
    adaptive: AdaptiveConfig,
    backends: dict[str, Backend],
    default_name: str,
) -> tuple[str, str] | None:
    """Pick a (backend_name, model) via adaptive routing, or None to fall back."""
    candidates = adaptive.candidates.get(tool_name)
    if not candidates or len(candidates) < 2:
        return None

    from .storage import get_model_scores

    bare_models = [
        _resolve_candidate(c, backends, default_name)[1] for c in candidates
    ]
    scores = get_model_scores(
        tool_name, bare_models, recency_window=adaptive.recency_window,
    )
    scores_by_model = {s["model"]: s for s in scores}

    eligible = [
        c for c in candidates
        if scores_by_model.get(
            _resolve_candidate(c, backends, default_name)[1], {},
        ).get("sample_count", 0) >= adaptive.min_samples
    ]

    if not eligible:
        return None

    if random.random() < adaptive.explore_rate:
        non_best = [c for c in candidates if c not in eligible[:1]]
        if non_best:
            pick = random.choice(non_best)
            return _resolve_candidate(pick, backends, default_name)

    best = max(
        eligible,
        key=lambda c: _adaptive_score(
            scores_by_model.get(
                _resolve_candidate(c, backends, default_name)[1], {},
            ),
            adaptive.quality_weight,
        ),
    )
    return _resolve_candidate(best, backends, default_name)


def _adaptive_score(stats: dict, quality_weight: float) -> float:
    if not stats:
        return 0.0
    quality = stats.get("avg_score", 0.0)
    latency = stats.get("avg_latency_ms", 1)
    speed = 1.0 / max(latency, 1)
    return quality_weight * quality + (1 - quality_weight) * speed


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
    adaptive = _parse_adaptive(raw)

    if adaptive.enabled and tool_name in adaptive.candidates:
        pick = _pick_adaptive(tool_name, adaptive, backends, default_name)
        if pick is not None:
            backend_name, model = pick
            backend = backends.get(backend_name, backends[default_name])
            return backend, model or backend.default_model

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
    adaptive = _parse_adaptive(raw)

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

    if adaptive.enabled and adaptive.candidates:
        from .storage import get_model_scores

        lines += ["", "Adaptive routing: enabled"]
        lines.append(
            f"  explore_rate: {adaptive.explore_rate}, "
            f"quality_weight: {adaptive.quality_weight}, "
            f"min_samples: {adaptive.min_samples}, "
            f"recency_window: {adaptive.recency_window}"
        )

        for tool in sorted(adaptive.candidates):
            candidates = adaptive.candidates[tool]
            bare_models = [
                _resolve_candidate(c, backends, default_name)[1]
                for c in candidates
            ]
            scores = get_model_scores(
                tool, bare_models, recency_window=adaptive.recency_window,
            )
            scores_by_model = {s["model"]: s for s in scores}

            lines.append(f"  {tool}:")
            for c in candidates:
                _, model = _resolve_candidate(c, backends, default_name)
                stats = scores_by_model.get(model)
                if stats and stats["sample_count"] >= adaptive.min_samples:
                    combined = _adaptive_score(stats, adaptive.quality_weight)
                    lines.append(
                        f"    {c}: score {stats['avg_score']:.2f}, "
                        f"latency {stats['avg_latency_ms']}ms, "
                        f"combined {combined:.3f} "
                        f"({stats['sample_count']} samples)"
                    )
                elif stats:
                    lines.append(
                        f"    {c}: warming up "
                        f"({stats['sample_count']}/{adaptive.min_samples} samples)"
                    )
                else:
                    lines.append(f"    {c}: no data yet")
    elif adaptive.enabled:
        lines += ["", "Adaptive routing: enabled (no candidates configured)"]

    return "\n".join(lines)
