import os

import numpy
import pytest
import torch
from openff.toolkit.topology import Molecule
from openff.units import unit

from nagl.datasets import DGLMoleculeDataLoader, DGLMoleculeDataset
from nagl.features import AtomConnectivity, BondIsInRing
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
    WibergBondOrderSet,
)


def label_function(molecule: Molecule):
    return {
        "formal_charges": torch.tensor(
            [
                atom.formal_charge.m_as(unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.float,
        ),
    }


def test_data_set_from_molecules(openff_methane):

    data_set = DGLMoleculeDataset.from_molecules(
        [openff_methane], [AtomConnectivity()], [BondIsInRing()], label_function
    )
    assert len(data_set) == 1
    assert data_set.n_features == 4

    dgl_molecule, labels = data_set[0]

    assert isinstance(dgl_molecule, DGLMolecule)
    assert dgl_molecule.n_atoms == 5

    assert "formal_charges" in labels
    label = labels["formal_charges"]

    assert label.numpy().shape == (5,)


@pytest.mark.parametrize(
    "partial_charge_method, bond_order_method",
    [("am1", None), (None, "am1"), ("am1", "am1")],
)
def test_labelled_molecule_to_dict(
    openff_methane, partial_charge_method, bond_order_method
):

    expected_charges = numpy.arange(openff_methane.n_atoms)
    expected_orders = numpy.arange(openff_methane.n_bonds)

    openff_methane.partial_charges = expected_charges * unit.elementary_charge

    for i, bond in enumerate(openff_methane.bonds):
        bond.fractional_bond_order = expected_orders[i]

    labels = DGLMoleculeDataset._labelled_molecule_to_dict(
        openff_methane, partial_charge_method, bond_order_method
    )

    if partial_charge_method is not None:
        assert "am1-charges" in labels
        assert numpy.allclose(expected_charges, labels["am1-charges"])
    else:
        assert "am1-charges" not in labels

    if bond_order_method is not None:
        assert "am1-wbo" in labels
        assert numpy.allclose(expected_orders, labels["am1-wbo"])
    else:
        assert "am1-wbo" not in labels


def test_data_set_from_molecule_stores(tmpdir):

    molecule_store = MoleculeStore(os.path.join(tmpdir, "store.sqlite"))
    molecule_store.store(
        MoleculeRecord(
            smiles="[Cl:1]-[H:2]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    partial_charges=[
                        PartialChargeSet(method="am1", values=[0.1, -0.1])
                    ],
                    bond_orders=[
                        WibergBondOrderSet(method="am1", values=[(0, 1, 1.1)])
                    ],
                )
            ],
        )
    )

    data_set = DGLMoleculeDataset.from_molecule_stores(
        molecule_store, "am1", "am1", [AtomConnectivity()], [BondIsInRing()]
    )

    assert len(data_set) == 1
    assert data_set.n_features == 4

    dgl_molecule, labels = data_set[0]

    assert isinstance(dgl_molecule, DGLMolecule)
    assert dgl_molecule.n_atoms == 2

    assert "am1-charges" in labels
    assert labels["am1-charges"].numpy().shape == (2,)

    assert "am1-wbo" in labels
    assert labels["am1-wbo"].numpy().shape == (1,)


def test_data_set_loader():

    data_loader = DGLMoleculeDataLoader(
        dataset=DGLMoleculeDataset.from_molecules(
            [Molecule.from_smiles("C"), Molecule.from_smiles("C[O-]")],
            [AtomConnectivity()],
            [],
            label_function,
        ),
    )

    entries = [*data_loader]

    for dgl_molecule, labels in entries:

        assert isinstance(
            dgl_molecule, DGLMoleculeBatch
        ) and dgl_molecule.n_atoms_per_molecule == (5,)
        assert "formal_charges" in labels
