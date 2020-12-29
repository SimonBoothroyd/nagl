import os
import pickle

import numpy
from openforcefield.topology import Molecule
from simtk import unit

from nagl.cli.label import label_cli


def test_label_cli(methane: Molecule, runner):

    # Create an SDF file to label.
    methane.to_file("methane.sdf", "sdf")

    arguments = [
        "--input",
        "methane.sdf",
        "--output",
        "labelled.pkl",
    ]

    result = runner.invoke(label_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("labelled.pkl")

    with open("labelled.pkl", "rb") as file:
        methane_labelled = pickle.load(file)

    assert isinstance(methane_labelled, Molecule)
    assert methane_labelled.n_atoms == methane.n_atoms
    assert methane_labelled.n_bonds == methane.n_bonds

    assert not all(
        numpy.isclose(charge.value_in_unit(unit.elementary_charge), 0.0)
        for charge in methane_labelled.partial_charges
    )
    assert not all(
        numpy.isclose(bond.fractional_bond_order, 0.0)
        for bond in methane_labelled.bonds
    )
