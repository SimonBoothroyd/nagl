import os

from rdkit import Chem

from nagl.cli.prepare.enumerate import enumerate_cli
from nagl.utilities.molecule import (
    molecule_from_smiles,
    stream_from_file,
    stream_to_file,
)


def test_enumerate_cli(rdkit_methane, tmp_cwd, runner):

    # Create an SDF file to enumerate.
    buteneol = molecule_from_smiles(r"C/C=C(/C)\O")

    with stream_to_file(tmp_cwd / "molecules.sdf") as writer:

        writer(buteneol)
        writer(buteneol)

    arguments = ["--input", "molecules.sdf", "--output", "tautomers.sdf", "--tautomers"]

    result = runner.invoke(enumerate_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("tautomers.sdf")

    tautomers = [molecule for molecule in stream_from_file(tmp_cwd / "tautomers.sdf")]
    assert len(tautomers) == 3

    actual_smiles = {
        Chem.MolToSmiles(Chem.RemoveHs(tautomer)) for tautomer in tautomers
    }

    expected_smiles = {"C/C=C(/C)O", "C=C(O)CC", "CCC(C)=O"}
    assert actual_smiles == expected_smiles
