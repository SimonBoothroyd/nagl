"""Utilities for generating default provenance information."""
import functools
import re
import subprocess
import typing


@functools.lru_cache()
def get_ambertools_version() -> typing.Optional[str]:
    """Attempts to retrieve the version of the currently installed AmberTools."""

    result = subprocess.run(["conda", "list"], capture_output=True, text=True)
    lines = result.stdout.split("\n")

    package_versions = {}

    for output_line in lines[3:-1]:
        package_name, package_version, *_ = re.split(" +", output_line)
        package_versions[package_name] = package_version

    return package_versions.get("ambertools", None)


def default_software_provenance() -> typing.Dict[str, str]:
    """Returns the versions of the core dependencies used when labelling a set of
    molecules."""

    import rdkit
    import torch

    import nagl

    software_provenance = {
        "nagl": nagl.__version__,
        "rdkit": rdkit.__version__,
        "ambertools": get_ambertools_version(),
        "torch": torch.__version__,
    }

    return {
        package_name: version
        for package_name, version in software_provenance.items()
        if version is not None
    }
