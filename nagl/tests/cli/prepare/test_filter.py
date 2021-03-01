import os

from openff.toolkit.topology import Molecule
from openff.toolkit.utils import RDKitToolkitWrapper

from nagl.cli.prepare.filter import filter_cli
from nagl.utilities.toolkits import stream_from_file, stream_to_file


def test_filter_cli(methane: Molecule, runner):

    # Create an SDF file to filter.
    with stream_to_file("molecules.sdf") as writer:

        writer(Molecule.from_smiles("C1(=C(C(=C(C(=C1Cl)Cl)Cl)Cl)Cl)[O-].[Na+]"))
        writer(Molecule.from_smiles("CCC(C)(C)C(F)(F)CCCCC(F)(F)C(C)(C)CC"))

    arguments = ["--input", "molecules.sdf", "--output", "filtered.sdf", "--strip-ions"]

    result = runner.invoke(filter_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("filtered.sdf")

    filtered_molecules = [molecule for molecule in stream_from_file("filtered.sdf")]
    assert len(filtered_molecules) == 1

    filtered_molecule = filtered_molecules[0]

    assert (
        filtered_molecule.to_smiles(toolkit_registry=RDKitToolkitWrapper())
        == "[O-][c]1[c]([Cl])[c]([Cl])[c]([Cl])[c]([Cl])[c]1[Cl]"
    )
