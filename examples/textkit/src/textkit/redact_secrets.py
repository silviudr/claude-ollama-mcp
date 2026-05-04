import re


def redact_secrets(s: str) -> str:
    # Redact JSON-style key:value pairs with sensitive keys
    json_pattern = r'("(?:[^"]*?(?:password|secret|token|api_key)[^"]*?)":\s*)"([^"]*?)"'
    s = re.sub(json_pattern, r'\1"***"', s, flags=re.IGNORECASE)

    # Redact OpenAI-style keys and long tokens
    token_pattern = r'(sk-[A-Za-z0-9_\-]{20,}|[A-Za-z0-9_\-]{24,})'
    s = re.sub(token_pattern, '***', s)

    return s
