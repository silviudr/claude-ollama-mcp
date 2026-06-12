"""Ollama HTTP client — compatibility shim.

Delegates to the OllamaBackend class. Prefer using backends directly
via ``from ollama_mcp.backends import OllamaBackend`` or the router's
``resolve()`` function for new code.
"""

from .backends.base import Backend
from .backends.ollama import OllamaBackend
from .config import MODEL, OLLAMA_URL

_default_backend = OllamaBackend(url=OLLAMA_URL, default_model=MODEL)

generate = _default_backend.generate
generate_json = _default_backend.generate_json
list_models = _default_backend.list_models
_strip_fences = Backend._strip_fences
