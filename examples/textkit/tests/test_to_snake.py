import pytest
from textkit.to_snake import to_snake


@pytest.mark.parametrize("input_string, expected_output", [
    ("XMLHttpRequest", "xml_http_request"),
    ("camelCase", "camel_case"),
    ("my-var name", "my_var_name"),
    ("already_snake", "already_snake"),
    ("", ""),
])
def test_to_snake(input_string, expected_output):
    assert to_snake(input_string) == expected_output
