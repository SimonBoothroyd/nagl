from nagl.utilities.provenance import (
    _get_optional_dependency_version,
    default_software_provenance,
)


def test_get_optional_dependency_version():
    assert _get_optional_dependency_version("my_fake_package") is None


def test_default_software_provenance():
    assert "nagl" in default_software_provenance()
