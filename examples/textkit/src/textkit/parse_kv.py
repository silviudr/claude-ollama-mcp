import shlex


def parse_kv(s: str) -> dict[str, str]:
    result = {}
    tokens = shlex.split(s)
    for token in tokens:
        if '=' in token:
            key, value = token.split('=', 1)
            result[key] = value
    return result
