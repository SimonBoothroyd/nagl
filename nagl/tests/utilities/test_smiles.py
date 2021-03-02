import pytest
from openff.toolkit.utils import UndefinedStereochemistryError

from nagl.tests import does_not_raise
from nagl.utilities.smiles import map_indexed_smiles, smiles_to_molecule


@pytest.mark.parametrize(
    "smiles, guess_stereochemistry, expected_raises",
    [
        ("CO", False, does_not_raise()),
        ("C(F)(Cl)(Br)", False, pytest.raises(UndefinedStereochemistryError)),
        ("C(F)(Cl)(Br)", True, does_not_raise()),
    ],
)
def test_smiles_to_molecule(smiles, guess_stereochemistry, expected_raises):
    """Tests that the `smiles_to_molecule` behaves as expected."""

    with expected_raises:
        smiles_to_molecule(smiles, guess_stereochemistry)


@pytest.mark.parametrize(
    "smiles_a,smiles_b,expected",
    [
        ("[Cl:1][H:2]", "[Cl:2][H:1]", {0: 1, 1: 0}),
        ("[Cl:2][H:1]", "[Cl:1][H:2]", {0: 1, 1: 0}),
    ],
)
def test_map_indexed_smiles(smiles_a, smiles_b, expected):
    assert map_indexed_smiles(smiles_a, smiles_b) == expected
