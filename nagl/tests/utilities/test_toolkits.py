import pytest
from openff.toolkit.topology import Molecule
from openff.utilities import temporary_cd

from nagl.utilities.toolkits import (
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
