"""Environment-driven configuration."""

import os
from pathlib import Path

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gemma4-32k")
LOG_PATH = Path(
    os.environ.get("OLLAMA_MCP_LOG", str(Path.home() / ".cache" / "ollama_mcp.jsonl"))
)
DB_PATH = Path(
    os.environ.get("OLLAMA_MCP_DB", str(Path.home() / ".cache" / "ollama_mcp.db"))
)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
