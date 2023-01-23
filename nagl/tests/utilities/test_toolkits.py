from openff.toolkit.topology import Molecule
from openff.utilities import temporary_cd

from nagl.utilities.toolkits import normalize_molecule, stream_from_file, stream_to_file


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


def test_normalize_molecule():

    expected_molecule = Molecule.from_smiles("CS(=O)(=O)C")

    molecule = Molecule.from_smiles("C[S+2]([O-])([O-])C")
    assert not Molecule.are_isomorphic(molecule, expected_molecule)[0]

    output_molecule = normalize_molecule(molecule)
    assert Molecule.are_isomorphic(output_molecule, expected_molecule)[0]
