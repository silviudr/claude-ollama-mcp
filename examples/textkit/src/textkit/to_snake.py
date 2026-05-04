import re


def to_snake(s: str) -> str:
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    s = re.sub(r'([A-Z])([A-Z][a-z])', r'\1_\2', s)
    s = re.sub(r'[ \-]+', '_', s)
    s = s.lower()
    s = re.sub(r'_{2,}', '_', s)
    return s.strip('_')
