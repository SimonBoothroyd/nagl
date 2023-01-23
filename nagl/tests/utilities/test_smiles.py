import pytest
from openff.toolkit.utils import UndefinedStereochemistryError

from nagl.tests import does_not_raise
from nagl.utilities.smiles import smiles_to_molecule


@pytest.mark.parametrize(
    "smiles, guess_stereo, expected_raises",
    [
        ("CO", False, does_not_raise()),
        ("C(F)(Cl)(Br)", False, pytest.raises(UndefinedStereochemistryError)),
        ("C(F)(Cl)(Br)", True, does_not_raise()),
    ],
)
def test_smiles_to_molecule(smiles, guess_stereo, expected_raises):
    """Tests that the `smiles_to_molecule` behaves as expected."""

    with expected_raises:
        smiles_to_molecule(smiles, guess_stereo)
