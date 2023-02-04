import pytest

from nagl.utilities import get_map_func


@pytest.mark.parametrize("n_processes, expected_name", [(1, "imap"), (0, "map")])
def test_get_map_func(n_processes, expected_name):
    with get_map_func(n_processes) as map_func:
        assert map_func.__name__ == expected_name
