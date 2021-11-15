from openff.toolkit.topology import Molecule
from openff.utilities import temporary_cd

from nagl.utilities.toolkits import stream_from_file, stream_to_file


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
