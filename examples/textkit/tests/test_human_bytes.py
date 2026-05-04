import pytest
from textkit.human_bytes import human_bytes


@pytest.mark.parametrize("input_bytes, expected_output", [
    (0, '0 B'),
    (512, '512 B'),
    (1024, '1.0 KiB'),
    (1536, '1.5 KiB'),
    (1048576, '1.0 MiB'),
])
def test_human_bytes(input_bytes, expected_output):
    assert human_bytes(input_bytes) == expected_output


def test_human_bytes_negative_raises():
    with pytest.raises(ValueError):
        human_bytes(-1)
