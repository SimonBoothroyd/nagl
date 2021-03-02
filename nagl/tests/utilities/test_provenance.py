import pkg_resources
import pytest

from nagl.utilities.provenance import (
    _get_ambertools_version,
    _get_optional_dependency_version,
    get_labelling_software_provenance,
)


def test_get_optional_dependency_version():
    assert _get_optional_dependency_version("my_fake_package") is None


def test_get_ambertools_version():

    pytest.importorskip("parmed")

    version = _get_ambertools_version()
    assert version is not None and isinstance(version, str)


def test_get_ambertools_version_missing(monkeypatch):
    def get_distribution(*_):
        raise pkg_resources.DistributionNotFound("AmberTools")

    monkeypatch.setattr(pkg_resources, "get_distribution", get_distribution)

    version = _get_ambertools_version()
    assert version is None


def test_get_labelling_software_provenance():
    assert "nagl" in get_labelling_software_provenance()
