def human_bytes(n: int) -> str:
    if n < 0:
        raise ValueError("Input byte count must be non-negative.")

    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]

    i = 0
    value = float(n)
    while value >= 1024 and i < len(units) - 1:
        value /= 1024.0
        i += 1

    if units[i] == "B":
        return f"{int(value)} B"
    else:
        return f"{value:.1f} {units[i]}"
