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
PRIVACY_CONFIG_PATH = Path(
    os.environ.get(
        "OLLAMA_MCP_PRIVACY",
        str(Path.home() / ".config" / "ollama_mcp" / "privacy.json"),
    )
)
ROUTES_CONFIG_PATH = Path(
    os.environ.get(
        "OLLAMA_MCP_ROUTES",
        str(Path.home() / ".config" / "ollama_mcp" / "routes.json"),
    )
)

# Data analyzer settings
ANALYSIS_SAMPLE_ROWS = int(os.environ.get("OLLAMA_MCP_SAMPLE_ROWS", "50"))
ANALYSIS_MAX_COLS = int(os.environ.get("OLLAMA_MCP_MAX_COLS", "20"))
ANALYSIS_THRESHOLD = float(os.environ.get("OLLAMA_MCP_COMPLEXITY_THRESHOLD", "0.7"))
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
