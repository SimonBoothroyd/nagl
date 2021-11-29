import pytest
from openff.toolkit.topology import Molecule

from nagl.labelling import label_molecule, label_molecules
from nagl.storage import MoleculeRecord


@pytest.mark.parametrize("molecule", ["C", Molecule.from_smiles("C")])
def test_label_molecule(molecule):

    labelled_record = label_molecule(
        molecule,
        guess_stereochemistry=True,
        partial_charge_methods=["am1bcc"],
        bond_order_methods=["am1"],
    )
    assert isinstance(labelled_record, MoleculeRecord)

    assert len(labelled_record.conformers) == 1
    assert {*labelled_record.conformers[0].partial_charges_by_method} == {"am1bcc"}
    assert {*labelled_record.conformers[0].bond_orders_by_method} == {"am1"}

    if isinstance(molecule, Molecule):
        assert molecule.conformers is None


def test_label_molecules():

    [(record_1, error_1), (record_2, error_2)] = label_molecules(
        ["C", "ClC=CCl"],
        guess_stereochemistry=False,
        partial_charge_methods=["am1"],
        bond_order_methods=[],
    )

    assert isinstance(record_1, MoleculeRecord)
    assert {*record_1.conformers[0].partial_charges_by_method} == {"am1"}
    assert error_1 is None

    assert record_2 is None
    assert "UndefinedStereo" in error_2
