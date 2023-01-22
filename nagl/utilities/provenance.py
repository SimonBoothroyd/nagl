import importlib
import typing

from openff.utilities.provenance import get_ambertools_version

import nagl


def _get_optional_dependency_version(import_path: str) -> typing.Optional[str]:
    """Attempts to retrieve the version of an optional dependency

    Args:
        import_path: The import path of the dependency.

    Returns:
        The version of the dependency if it can be imported, otherwise none.
    """

    try:
        dependency = importlib.import_module(import_path)
        # noinspection PyUnresolvedReferences
        return dependency.__version__
    except ImportError:
        return None


def get_labelling_software_provenance() -> typing.Dict[str, str]:
    """Returns the versions of the core dependencies used when labelling a set of
    molecules."""

    software_provenance = {
        "nagl": nagl.__version__,
        "openff-toolkit": _get_optional_dependency_version("openff.toolkit"),
        "openeye": _get_optional_dependency_version("openeye"),
        "rdkit": _get_optional_dependency_version("rdkit"),
        "ambertools": get_ambertools_version(),
    }

    return {
        package_name: version
        for package_name, version in software_provenance.items()
        if version is not None
    }
