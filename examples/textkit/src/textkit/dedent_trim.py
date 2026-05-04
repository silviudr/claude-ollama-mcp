import textwrap


def dedent_trim(s: str) -> str:
    s_dedented = textwrap.dedent(s)
    lines = s_dedented.splitlines()

    if not lines:
        return ""

    start_index = 0
    for i, line in enumerate(lines):
        if line.strip() != "":
            start_index = i
            break
    else:
        return ""

    end_index = len(lines) - 1
    for j in range(len(lines) - 1, -1, -1):
        if lines[j].strip() != "":
            end_index = j
            break

    trimmed_lines = lines[start_index : end_index + 1]
    return "\n".join(trimmed_lines)
