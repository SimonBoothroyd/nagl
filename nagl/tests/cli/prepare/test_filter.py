import os

from openeye import oechem
from openforcefield.topology import Molecule

from nagl.cli.prepare.filter import filter_cli


def test_filter_cli(methane: Molecule, runner):

    # Create an SDF file to filter.
    output_stream = oechem.oemolostream("molecules.sdf")
    oechem.OEWriteMolecule(
        output_stream,
        Molecule.from_smiles("C1(=C(C(=C(C(=C1Cl)Cl)Cl)Cl)Cl)[O-].[Na+]").to_openeye(),
    )
    oechem.OEWriteMolecule(
        output_stream,
        Molecule.from_smiles("CCC(C)(C)C(F)(F)CCCCC(F)(F)C(C)(C)CC").to_openeye(),
    )
    output_stream.close()

    arguments = ["--input", "molecules.sdf", "--output", "filtered.sdf", "--strip-ions"]

    result = runner.invoke(filter_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("filtered.sdf")

    input_stream = oechem.oemolistream("filtered.sdf")
    filtered_molecules = [
        oechem.OEMol(oe_molecule) for oe_molecule in input_stream.GetOEMols()
    ]
    input_stream.close()

    assert len(filtered_molecules) == 1

    filtered_molecule = Molecule.from_openeye(filtered_molecules[0])
    assert filtered_molecule.to_smiles() == "c1(c(c(c(c(c1Cl)Cl)Cl)Cl)Cl)[O-]"
