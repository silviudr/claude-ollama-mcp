"""A small utility module for testing local_generate_tests."""


def clamp(value: float, low: float, high: float) -> float:
    """Restrict value to the range [low, high]."""
    if low > high:
        raise ValueError(f"low ({low}) must be <= high ({high})")
    return max(low, min(high, value))


def chunk(items: list, size: int) -> list[list]:
    """Split a list into chunks of the given size."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [items[i : i + size] for i in range(0, len(items), size)]


def flatten(nested: list) -> list:
    """Flatten one level of nesting."""
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result


def deduplicate(items: list) -> list:
    """Remove duplicates while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """Divide a by b, returning default if b is zero."""
    if b == 0:
        return default
    return a / b
