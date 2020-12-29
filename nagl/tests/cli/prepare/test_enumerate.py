import os

from openeye import oechem
from openforcefield.topology import Molecule

from nagl.cli.prepare.enumerate import enumerate_cli


def test_enumerate_cli(methane: Molecule, runner):

    # Create an SDF file to enumerate.
    Molecule.from_smiles(r"C/C=C(/C)\O").to_file("molecules.sdf", "sdf")

    arguments = [
        "--input",
        "molecules.sdf",
        "--output",
        "tautomers.sdf",
    ]

    result = runner.invoke(enumerate_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("tautomers.sdf")

    input_stream = oechem.oemolistream("tautomers.sdf")
    tautomers = [oechem.OEMol(oe_molecule) for oe_molecule in input_stream.GetOEMols()]
    input_stream.close()

    assert len(tautomers) == 2

    assert {oechem.OEMolToSmiles(tautomer) for tautomer in tautomers} == {
        r"C/C=C(/C)\O",
        "CCC(=O)C",
    }
