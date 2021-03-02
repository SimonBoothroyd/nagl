import os

import pytest

from nagl.utilities import MissingOptionalDependency, requires_package, temporary_cd
from nagl.utilities.utilities import _CONDA_INSTALLATION_COMMANDS


def compare_paths(path_1: str, path_2: str) -> bool:
    """Checks whether two paths are the same.

    Args
        path_1: The first path.
        path_2: The second path.

    Returns
        True if the paths are equivalent.
    """
    return os.path.normpath(path_1) == os.path.normpath(path_2)


def test_requires_package_found():
    @requires_package("nagl")
    def dummy_function():
        return 42

    assert dummy_function() == 42


def test_requires_package_unknown_missing():
    @requires_package("fake-package-42")
    def dummy_function():
        pass

    with pytest.raises(MissingOptionalDependency) as error_info:
        dummy_function()

    assert "The required fake-package-42 module could not be imported." in str(
        error_info.value
    )


def test_requires_package_known_missing(monkeypatch):

    monkeypatch.setitem(
        _CONDA_INSTALLATION_COMMANDS, "fake-package-42", "conda install ..."
    )

    @requires_package("fake-package-42")
    def dummy_function():
        pass

    with pytest.raises(MissingOptionalDependency) as error_info:
        dummy_function()

    assert "Try installing the package by running `conda install ...`" in str(
        error_info.value
    )


def test_temporary_cd():
    """Tests that temporary cd works as expected"""

    original_directory = os.getcwd()

    # Move to the parent directory
    with temporary_cd(os.pardir):

        current_directory = os.getcwd()
        expected_directory = os.path.abspath(
            os.path.join(original_directory, os.pardir)
        )

        assert compare_paths(current_directory, expected_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to a temporary directory
    with temporary_cd():
        assert not compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    # Move to the same directory
    with temporary_cd(""):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)

    with temporary_cd(os.curdir):
        assert compare_paths(os.getcwd(), original_directory)

    assert compare_paths(os.getcwd(), original_directory)
