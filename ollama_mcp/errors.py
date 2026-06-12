"""Structured errors for backend failures."""

from .config import MODEL, OLLAMA_URL


class BackendError(Exception):
    """Base class for all backend-related errors."""


# --- Ollama errors ---


class OllamaError(BackendError):
    """Base class for Ollama-specific errors."""


class OllamaConnectionError(OllamaError):
    def __init__(self, url: str = OLLAMA_URL):
        super().__init__(
            f"Cannot reach Ollama at {url}. "
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


# --- OpenRouter errors ---


class OpenRouterError(BackendError):
    """Base class for OpenRouter-specific errors."""


class OpenRouterConnectionError(OpenRouterError):
    def __init__(self):
        super().__init__(
            "Cannot reach OpenRouter at openrouter.ai. "
            "Check your internet connection."
        )


class OpenRouterAuthError(OpenRouterError):
    def __init__(self, detail: str = "Invalid or missing API key"):
        super().__init__(
            f"OpenRouter authentication failed: {detail}. "
            f"Get a key at https://openrouter.ai/keys"
        )


class OpenRouterRateLimitError(OpenRouterError):
    def __init__(self):
        super().__init__(
            "OpenRouter rate limit exceeded. "
            "Wait a moment or upgrade your plan at https://openrouter.ai"
        )


class OpenRouterModelNotFound(OpenRouterError):
    def __init__(self, model: str):
        super().__init__(
            f"Model '{model}' not available on OpenRouter. "
            f"Check available models at https://openrouter.ai/models"
        )


class OpenRouterTimeout(OpenRouterError):
    def __init__(self, timeout_s: int):
        super().__init__(
            f"OpenRouter request timed out after {timeout_s}s. "
            f"The model may be overloaded or the prompt too large."
        )


class OpenRouterServerError(OpenRouterError):
    def __init__(self, status_code: int, detail: str):
        super().__init__(
            f"OpenRouter returned HTTP {status_code}: {detail}"
        )
