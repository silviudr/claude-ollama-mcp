def wrap_by_width(s: str, width: int) -> list[str]:
    if width < 1:
        raise ValueError("Width must be at least 1.")

    if not s:
        return []

    words = s.split()
    if not words:
        return []

    lines = []
    current_line = []

    for word in words:
        if len(word) > width:
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
            lines.append(word)
            continue

        if not current_line:
            current_line.append(word)
        else:
            potential_length = len(" ".join(current_line)) + 1 + len(word)
            if potential_length <= width:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]

    if current_line:
        lines.append(" ".join(current_line))

    return lines
