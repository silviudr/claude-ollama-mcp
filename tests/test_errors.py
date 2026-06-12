from ollama_mcp.errors import (
    BackendError,
    OllamaConnectionError,
    OllamaError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
    OpenRouterAuthError,
    OpenRouterConnectionError,
    OpenRouterError,
    OpenRouterModelNotFound,
    OpenRouterRateLimitError,
    OpenRouterServerError,
    OpenRouterTimeout,
)


# --- Hierarchy ---


def test_ollama_errors_inherit_from_backend_error():
    for cls in [
        OllamaConnectionError,
        OllamaModelNotFound,
        OllamaTimeout,
        OllamaMalformedResponse,
        OllamaServerError,
    ]:
        assert issubclass(cls, OllamaError)
        assert issubclass(cls, BackendError)


def test_openrouter_errors_inherit_from_backend_error():
    for cls in [
        OpenRouterConnectionError,
        OpenRouterAuthError,
        OpenRouterRateLimitError,
        OpenRouterModelNotFound,
        OpenRouterServerError,
        OpenRouterTimeout,
    ]:
        assert issubclass(cls, OpenRouterError)
        assert issubclass(cls, BackendError)


def test_backend_error_catches_both():
    try:
        raise OllamaConnectionError()
    except BackendError:
        pass

    try:
        raise OpenRouterConnectionError()
    except BackendError:
        pass


# --- Ollama error messages ---


def test_connection_error_message():
    err = OllamaConnectionError()
    assert "ollama serve" in str(err)


def test_connection_error_custom_url():
    err = OllamaConnectionError("http://remote:11434")
    assert "remote:11434" in str(err)


def test_model_not_found_message():
    err = OllamaModelNotFound("llama3")
    assert "llama3" in str(err)
    assert "ollama pull" in str(err)


def test_timeout_message():
    err = OllamaTimeout(120)
    assert "120s" in str(err)


def test_malformed_response_message():
    err = OllamaMalformedResponse("no JSON")
    assert "no JSON" in str(err)


def test_server_error_message():
    err = OllamaServerError(503, "service unavailable")
    assert "503" in str(err)
    assert "service unavailable" in str(err)


# --- OpenRouter error messages ---


def test_openrouter_connection_error():
    err = OpenRouterConnectionError()
    assert "openrouter.ai" in str(err)


def test_openrouter_auth_error_default():
    err = OpenRouterAuthError()
    assert "openrouter.ai/keys" in str(err)


def test_openrouter_auth_error_custom():
    err = OpenRouterAuthError("MY_KEY is not set")
    assert "MY_KEY" in str(err)


def test_openrouter_rate_limit():
    err = OpenRouterRateLimitError()
    assert "rate limit" in str(err).lower()


def test_openrouter_model_not_found():
    err = OpenRouterModelNotFound("google/gemma-3-27b-it")
    assert "gemma-3-27b-it" in str(err)
    assert "openrouter.ai/models" in str(err)


def test_openrouter_timeout():
    err = OpenRouterTimeout(60)
    assert "60s" in str(err)


def test_openrouter_server_error():
    err = OpenRouterServerError(502, "bad gateway")
    assert "502" in str(err)
    assert "bad gateway" in str(err)
