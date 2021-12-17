import pytest
from openff.toolkit.topology import Molecule
from openff.utilities import temporary_cd

from nagl.utilities.toolkits import (
    get_atom_symmetries,
    normalize_molecule,
    smiles_to_inchi_key,
    stream_from_file,
    stream_to_file,
)


def test_read_write_streams():

    molecules = [Molecule.from_smiles("C"), Molecule.from_smiles("CO")]

    with temporary_cd():

        with stream_to_file("molecules.sdf") as writer:

            for molecule in molecules:
                writer(molecule)

        loaded_molecules = [*stream_from_file("molecules.sdf")]

    assert len(molecules) == len(loaded_molecules)
    assert {molecule.to_smiles() for molecule in molecules} == {
        molecule.to_smiles() for molecule in loaded_molecules
    }


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("Cl", "VEXZGXHMUGYJMC-UHFFFAOYNA-N"),
        ("[H]Cl", "VEXZGXHMUGYJMC-UHFFFAOYNA-N"),
        ("[Cl:2][H:1]", "VEXZGXHMUGYJMC-UHFFFAOYNA-N"),
        ("C", "VNWKTOKETHGBQD-UHFFFAOYNA-N"),
        ("[CH4]", "VNWKTOKETHGBQD-UHFFFAOYNA-N"),
    ],
)
def test_smiles_to_inchi_key(smiles, expected):
    assert smiles_to_inchi_key(smiles) == expected


def test_get_atom_symmetries():

    molecule = Molecule.from_mapped_smiles("[H:1][C:2]([H:3])([H:4])[O:5][H:6]")

    atom_symmetries = get_atom_symmetries(molecule)

    assert len({atom_symmetries[i] for i in (0, 2, 3)}) == 1
    assert len({atom_symmetries[i] for i in (1, 4, 5)}) == 3


def test_normalize_molecule():

    expected_molecule = Molecule.from_smiles("CS(=O)(=O)C")

    molecule = Molecule.from_smiles("C[S+2]([O-])([O-])C")
    assert not Molecule.are_isomorphic(molecule, expected_molecule)[0]

    output_molecule = normalize_molecule(molecule)
    assert Molecule.are_isomorphic(output_molecule, expected_molecule)[0]
