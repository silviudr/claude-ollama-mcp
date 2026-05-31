from ollama_mcp.errors import (
    OllamaConnectionError,
    OllamaError,
    OllamaMalformedResponse,
    OllamaModelNotFound,
    OllamaServerError,
    OllamaTimeout,
)


def test_all_inherit_from_base():
    for cls in [
        OllamaConnectionError,
        OllamaModelNotFound,
        OllamaTimeout,
        OllamaMalformedResponse,
        OllamaServerError,
    ]:
        assert issubclass(cls, OllamaError)


def test_connection_error_message():
    err = OllamaConnectionError()
    assert "ollama serve" in str(err)


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
