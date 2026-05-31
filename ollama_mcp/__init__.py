"""Local Ollama MCP server.

Exposes MCP tools that delegate work to a local Ollama model
(default: gemma4-32k, override with OLLAMA_MODEL).
"""

from . import tools as _tools  # noqa: F401 — registers tools on import
from .server import mcp

__all__ = ["mcp"]
