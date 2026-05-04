import pytest
from textkit.slugify import slugify


@pytest.mark.parametrize("input_string, expected_output", [
    ('Héllo, World!', 'hello-world'),
    ('  spaces   here  ', 'spaces-here'),
    ('---!!!---', ''),
    ('AlreadySlug', 'alreadyslug'),
    ('', ''),
])
def test_slugify(input_string, expected_output):
    assert slugify(input_string) == expected_output
