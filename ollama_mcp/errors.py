"""Structured errors for Ollama failures."""

from .config import MODEL, OLLAMA_URL


class OllamaError(Exception):
    """Base class for all Ollama-related errors."""


class OllamaConnectionError(OllamaError):
    def __init__(self):
        super().__init__(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            f"Is it running? Try: ollama serve"
        )


class OllamaModelNotFound(OllamaError):
    def __init__(self, model: str = MODEL):
        super().__init__(
            f"Model '{model}' not found in Ollama. "
            f"Run: ollama pull {model}"
        )


class OllamaTimeout(OllamaError):
    def __init__(self, timeout_s: int):
        super().__init__(
            f"Ollama request timed out after {timeout_s}s. "
            f"The model may be overloaded or the prompt too large."
        )


class OllamaMalformedResponse(OllamaError):
    def __init__(self, detail: str):
        super().__init__(f"Unexpected response from Ollama: {detail}")


class OllamaServerError(OllamaError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(
            f"Ollama returned HTTP {status_code}: {detail}"
        )
