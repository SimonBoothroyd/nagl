import os

import numpy
from openff.toolkit.topology import Molecule

from nagl.cli.label import label_cli
from nagl.storage.storage import MoleculeStore


def test_label_cli(methane: Molecule, runner):

    # Create an SDF file to label.
    methane.to_file("methane.sdf", "sdf")

    arguments = [
        "--input",
        "methane.sdf",
        "--output",
        "labelled.sqlite",
    ]

    result = runner.invoke(label_cli, arguments)

    if result.exit_code != 0:
        raise result.exception

    assert os.path.isfile("labelled.sqlite")

    store = MoleculeStore("labelled.sqlite")

    assert len(store) == 1

    molecule_record = store.retrieve()[0]
    assert (
        Molecule.from_smiles(molecule_record.smiles).to_smiles() == methane.to_smiles()
    )

    assert len(molecule_record.conformers) == 1

    conformer_record = molecule_record.conformers[0]

    assert len(conformer_record.partial_charges) == 2
    assert len(conformer_record.bond_orders) == 1

    for partial_charge_set in conformer_record.partial_charges:

        assert not all(
            numpy.isclose(charge, 0.0) for charge in partial_charge_set.values
        )

    for bond_order_set in conformer_record.bond_orders:

        assert not all(
            numpy.isclose(value, 0.0) for (_, _, value) in bond_order_set.values
        )
