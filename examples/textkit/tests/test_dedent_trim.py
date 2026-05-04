import pytest
from textkit.dedent_trim import dedent_trim


@pytest.mark.parametrize("input_str, expected_output", [
    ('\n    a\n    b\n\n', 'a\nb'),
    ('\n  a\n    b\n', 'a\n  b'),
    ('', ''),
    ('\n \n   \n', ''),
])
def test_dedent_trim(input_str, expected_output):
    assert dedent_trim(input_str) == expected_output
