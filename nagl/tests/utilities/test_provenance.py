from nagl.utilities.provenance import (
    default_software_provenance,
    get_ambertools_version,
)


def test_default_software_provenance():
    assert "nagl" in default_software_provenance()


def test_get_ambertools_version():
    assert get_ambertools_version() is not None
