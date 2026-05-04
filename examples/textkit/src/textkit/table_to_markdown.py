def table_to_markdown(rows: list[dict]) -> str:
    if not rows:
        return ""

    keys = []
    seen_keys = set()
    for row in rows:
        for key in row:
            if key not in seen_keys:
                keys.append(key)
                seen_keys.add(key)

    if not keys:
        return ""

    def escape_pipes(text):
        return str(text).replace("|", r"\|")

    header = "| " + " | ".join(keys) + " |"
    separator = "| " + " | ".join(["---"] * len(keys)) + " |"

    body_rows = []
    for row_dict in rows:
        cells = []
        for key in keys:
            value = row_dict.get(key, "")
            cells.append(escape_pipes(value))
        body_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + body_rows)
