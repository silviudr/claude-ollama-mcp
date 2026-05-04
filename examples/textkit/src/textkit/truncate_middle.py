def truncate_middle(s: str, max_len: int) -> str:
    if max_len < 1:
        raise ValueError("max_len must be at least 1.")

    if len(s) <= max_len:
        return s

    remaining_space = max_len - 1
    t_l = remaining_space // 2
    t_r = remaining_space - t_l

    left_part = s[:t_l]
    right_part = s[len(s) - t_r:]

    return left_part + '…' + right_part
