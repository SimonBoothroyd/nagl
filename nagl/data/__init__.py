from typing import Literal


def get_file_path(relative_file_path: Literal["normalizations.json"]) -> str:
    """Get the full path to one of the files in the ``data`` directory.

    Args:
        relative_file_path: The path to the file relative to the ``data`` directory

    Returns:
        The absolute path to the file.

    Raises:
        FileNotFoundError
    """

    from openff.utilities import get_data_file_path

    return get_data_file_path(relative_file_path, "nagl")


__all__ = [get_file_path]
