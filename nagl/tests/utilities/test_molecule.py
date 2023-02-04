import pytest
from rdkit import Chem

from nagl.utilities.molecule import (
    molecule_from_mapped_smiles,
    molecule_from_smiles,
    molecule_to_mapped_smiles,
    normalize_molecule,
    stream_from_file,
    stream_to_file,
)


def test_read_write_streams(tmp_cwd):

    molecules = [molecule_from_smiles("C"), molecule_from_smiles("CO")]

    with stream_to_file(tmp_cwd / "molecules.sdf") as writer:

        for molecule in molecules:
            writer(molecule)

    loaded_molecules = [*stream_from_file(tmp_cwd / "molecules.sdf")]

    assert len(molecules) == len(loaded_molecules)
    assert {Chem.MolToSmiles(molecule) for molecule in molecules} == {
        Chem.MolToSmiles(molecule) for molecule in loaded_molecules
    }


def test_normalize_molecule():

    expected_smiles = "[H]C([H])([H])S(=O)(=O)C([H])([H])[H]"

    opts = Chem.SmilesWriteParams()
    opts.allHsExplicit = False

    molecule = molecule_from_smiles("C[S+2]([O-])([O-])C")
    assert Chem.MolToSmiles(molecule, opts) != expected_smiles

    output_molecule = normalize_molecule(molecule)
    assert Chem.MolToSmiles(output_molecule, opts) == expected_smiles


@pytest.mark.parametrize(
    "smiles, guess_stereo, expected_smiles",
    [
        ("CO", False, "[H]OC([H])([H])[H]"),
        ("FC(Cl)Br", False, "[H]C(F)(Cl)Br"),
        ("FC(Cl)Br", True, "[H][C@](F)(Cl)Br"),
    ],
)
def test_molecule_from_smiles(smiles, guess_stereo, expected_smiles):

    molecule = molecule_from_smiles(smiles, guess_stereo)
    assert isinstance(molecule, Chem.Mol)

    assert Chem.MolToSmiles(molecule) == expected_smiles


def test_molecule_from_smiles_failed():

    with pytest.raises(ValueError, match="could not parse X"):
        molecule_from_smiles("X")


@pytest.mark.parametrize(
    "smiles, expected_atomic_num",
    [
        ("[F:1][C:2]([H:3])([Cl:4])[Br:5]", [9, 6, 1, 17, 35]),
        ("[F:3][C:2]([H:1])([Cl:4])[Br:5]", [1, 6, 9, 17, 35]),
    ],
)
def test_molecule_from_mapped_smiles(smiles, expected_atomic_num):

    molecule = molecule_from_mapped_smiles(smiles)
    assert isinstance(molecule, Chem.Mol)

    assert [atom.GetAtomicNum() for atom in molecule.GetAtoms()] == expected_atomic_num


@pytest.mark.parametrize("smiles", ["C", "[CH4:1]", "[Cl:1]Cl"])
def test_molecule_from_mapped_smiles_failed(smiles):

    with pytest.raises(ValueError, match="all atoms must have a map index"):
        molecule_from_mapped_smiles(smiles)


def test_molecule_to_mapped_smiles():

    expected_smiles = "[H:1][C:2]([F:3])([Cl:4])[Br:5]"

    molecule = molecule_from_mapped_smiles(expected_smiles)
    smiles = molecule_to_mapped_smiles(molecule)

    assert smiles == expected_smiles
    assert isinstance(molecule, Chem.Mol)
