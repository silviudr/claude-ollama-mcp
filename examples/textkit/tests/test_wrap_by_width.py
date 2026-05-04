import pytest
from textkit.wrap_by_width import wrap_by_width


@pytest.mark.parametrize("input_string, width, expected", [
    ("one two three four", 8, ["one two", "three", "four"]),
    ("apple supercalifragilistic", 5, ["apple", "supercalifragilistic"]),
    ("", 10, []),
])
def test_wrap_by_width(input_string, width, expected):
    assert wrap_by_width(input_string, width) == expected


def test_wrap_by_width_zero_width_raises():
    with pytest.raises(ValueError):
        wrap_by_width("some text", 0)
