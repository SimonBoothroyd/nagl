import numpy
import pytest
from openff.toolkit.topology import Molecule

from nagl.labelling import _get_map_func, compute_charges_func, label_molecules


def test_compute_charges():

    molecule = Molecule.from_smiles("C")

    func = compute_charges_func("am1bcc", n_conformers=1)

    charges_per_method = func(molecule)
    assert set(charges_per_method) == {"smiles", "charges-am1bcc"}

    charges = charges_per_method["charges-am1bcc"]
    assert charges.shape == (5, 1)
    assert not numpy.allclose(charges, 0.0)


@pytest.mark.parametrize("n_processes, expected_name", [(1, "imap"), (0, "map")])
def test_get_map_func(n_processes, expected_name):

    with _get_map_func(n_processes) as map_func:
        assert map_func.__name__ == expected_name


def test_label_molecules():

    table, errors = label_molecules(
        ["C", "ClC=CCl"],
        compute_charges_func("am1", n_conformers=1),
        guess_stereo=False,
    )

    assert len(table) == 1
    assert table.column_names == ["smiles", "charges-am1"]

    assert len(errors) == 1
    assert "UndefinedStereo" in errors[0]
