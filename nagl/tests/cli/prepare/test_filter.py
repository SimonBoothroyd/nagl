import os

from rdkit import Chem

from nagl.cli.prepare.filter import filter_cli
from nagl.utilities.molecule import (
    molecule_from_smiles,
    stream_from_file,
    stream_to_file,
)


def test_filter_cli(rdkit_methane, tmp_cwd, runner):

    # Create an SDF file to filter.
    with stream_to_file(tmp_cwd / "molecules.sdf") as writer:

        writer(molecule_from_smiles("C1(=C(C(=C(C(=C1Cl)Cl)Cl)Cl)Cl)[O-].[Na+]"))
        writer(molecule_from_smiles("CCC(C)(C)C(F)(F)CCCCC(F)(F)C(C)(C)CC"))

    arguments = ["--input", "molecules.sdf", "--output", "filtered.sdf", "--strip-ions"]

    result = runner.invoke(filter_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("filtered.sdf")

    filtered_molecules = [
        molecule for molecule in stream_from_file(tmp_cwd / "filtered.sdf")
    ]
    assert len(filtered_molecules) == 1

    filtered_molecule = filtered_molecules[0]

    expected_smiles = Chem.MolToSmiles(
        Chem.MolFromSmiles("[O-][c]1[c]([Cl])[c]([Cl])[c]([Cl])[c]([Cl])[c]1[Cl]")
    )
    assert Chem.MolToSmiles(filtered_molecule) == expected_smiles
