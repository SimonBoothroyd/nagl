import numpy
import pytest
from openff.toolkit.topology import Molecule

from nagl.labelling import compute_batch_charges, compute_charges


@pytest.mark.parametrize("molecule", ["C", Molecule.from_smiles("C")])
def test_compute_charges(molecule):

    charges_per_method = compute_charges(molecule, "am1bcc")
    assert set(charges_per_method) == {"smiles", "am1bcc"}

    charges = charges_per_method["am1bcc"]
    assert charges.shape == (5,)
    assert not numpy.allclose(charges, 0.0)


def test_compute_batch_charges():

    [(record_1, error_1), (record_2, error_2)] = compute_batch_charges(
        ["C", "ClC=CCl"], "am1", guess_stereo=False
    )

    assert {*record_1} == {"smiles", "am1"}
    assert error_1 is None

    assert record_2 is None
    assert "UndefinedStereo" in error_2
