import functools
import importlib
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory
from typing import Optional

_CONDA_INSTALLATION_COMMANDS = {
    "openff.toolkit": "conda install -c conda-forge openff-toolkit",
    "dask.distributed": "conda install -c conda-forge distributed",
    "dask_jobqueue": "conda install -c conda-forge dask-jobqueue",
}


class MissingOptionalDependency(ImportError):
    """An exception raised when an optional dependency is required
    but cannot be found.
    """

    def __init__(self, library_name: str, license_issue: bool = False):
        """

        Args:
            library_name: The name of the missing library.
            license_issue: Whether the library was importable but was unusable due to a
                missing license.
        """

        message = f"The required {library_name} module could not be imported."
        conda_command = _CONDA_INSTALLATION_COMMANDS.get(
            library_name.split(".")[0], None
        )

        if license_issue:
            message = f"{message} This is due to a missing license."
        elif conda_command is not None:
            message = (
                f"{message} Try installing the package by running `{conda_command}`."
            )

        super(MissingOptionalDependency, self).__init__(message)

        self.library_name = library_name
        self.license_issue = license_issue


def requires_package(library_path: str):
    def inner_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):

            try:
                importlib.import_module(library_path)
            except (ImportError, ModuleNotFoundError):
                raise MissingOptionalDependency(library_path, False)
            except Exception as e:
                raise e

            return function(*args, **kwargs)

        return wrapper

    return inner_decorator


@contextmanager
def temporary_cd(directory_path: Optional[str] = None):
    """Temporarily move the current working directory to the path
    specified. If no path is given, a temporary directory will be
    created, moved into, and then destroyed when the context manager
    is closed.

    Args
        directory_path: The optional path to change to. If none is specified a random
        temporary directory will be changed to.
    """

    if directory_path is not None and len(directory_path) == 0:
        yield
        return

    old_directory = os.getcwd()

    try:

        if directory_path is None:

            with TemporaryDirectory() as new_directory:
                os.chdir(new_directory)
                yield

        else:

            os.chdir(directory_path)
            yield

    finally:
        os.chdir(old_directory)
