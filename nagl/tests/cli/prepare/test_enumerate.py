import os

from openff.toolkit.topology import Molecule
from openff.toolkit.utils import RDKitToolkitWrapper

from nagl.cli.prepare.enumerate import enumerate_cli
from nagl.utilities.toolkits import stream_from_file, stream_to_file


def test_enumerate_cli(openff_methane: Molecule, runner):

    # Create an SDF file to enumerate.
    buteneol = Molecule.from_smiles(r"C/C=C(/C)\O")

    with stream_to_file("molecules.sdf") as writer:

        writer(buteneol)
        writer(buteneol)

    arguments = ["--input", "molecules.sdf", "--output", "tautomers.sdf"]

    result = runner.invoke(enumerate_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("tautomers.sdf")

    tautomers = [molecule for molecule in stream_from_file("tautomers.sdf")]
    assert len(tautomers) == 4

    assert {
        tautomer.to_smiles(
            explicit_hydrogens=False, toolkit_registry=RDKitToolkitWrapper()
        )
        for tautomer in tautomers
    } == {"C/C=C(/C)O", "C=C(O)CC", "CCC(C)=O", "CC=C(C)O"}
