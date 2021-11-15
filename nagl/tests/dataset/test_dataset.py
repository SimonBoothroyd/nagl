import os

import numpy
import pytest
import torch
from openff.toolkit.topology import Molecule
from simtk import unit

from nagl.dataset.dataset import (
    MoleculeGraphDataLoader,
    MoleculeGraphDataset,
    molecule_to_graph,
)
from nagl.dataset.features import AtomConnectivity, BondIsInRing
from nagl.storage.storage import (
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
                atom.formal_charge.value_in_unit(unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.float,
        ),
    }


def test_molecule_to_graph(methane):

    molecule: Molecule = Molecule.from_smiles("C[O-]")
    molecule_graph = molecule_to_graph(molecule, [AtomConnectivity()], [BondIsInRing()])

    assert numpy.allclose(
        molecule_graph.ndata["formal_charge"].numpy(),
        numpy.array([0.0, -1.0, 0.0, 0.0, 0.0]),
    )

    node_features = molecule_graph.ndata["feat"].numpy()

    assert node_features.shape == (5, 4)
    assert not numpy.allclose(node_features, numpy.zeros_like(node_features))

    forward_features = molecule_graph.edges["forward"].data["feat"].numpy()
    reverse_features = molecule_graph.edges["reverse"].data["feat"].numpy()

    assert forward_features.shape == reverse_features.shape
    assert forward_features.shape == (4, 2)

    assert numpy.allclose(forward_features, reverse_features)

    assert numpy.allclose(
        forward_features[:, 1], numpy.zeros_like(forward_features[:, 1])
    )
    assert numpy.allclose(
        forward_features[:, 0], numpy.ones_like(forward_features[:, 0])
    )


def test_data_set_from_molecules(methane):

    data_set = MoleculeGraphDataset.from_molecules(
        [methane], [AtomConnectivity()], [BondIsInRing()], label_function
    )
    assert len(data_set) == 1
    assert data_set.n_features == 4

    molecule_graph, features, labels = data_set[0]

    assert molecule_graph is not None
    assert molecule_graph.number_of_nodes() == 5

    assert features.numpy().shape == (5, 4)

    assert "formal_charges" in labels
    label = labels["formal_charges"]

    assert label.numpy().shape == (5,)


@pytest.mark.parametrize(
    "partial_charge_method, bond_order_method",
    [("am1", None), (None, "am1"), ("am1", "am1")],
)
def test_labelled_molecule_to_dict(methane, partial_charge_method, bond_order_method):

    expected_charges = numpy.arange(methane.n_atoms)
    expected_orders = numpy.arange(methane.n_bonds)

    methane.partial_charges = expected_charges * unit.elementary_charge

    for i, bond in enumerate(methane.bonds):
        bond.fractional_bond_order = expected_orders[i]

    labels = MoleculeGraphDataset._labelled_molecule_to_dict(
        methane, partial_charge_method, bond_order_method
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

    data_set = MoleculeGraphDataset.from_molecule_stores(
        molecule_store, "am1", "am1", [AtomConnectivity()], [BondIsInRing()]
    )

    assert len(data_set) == 1
    assert data_set.n_features == 4

    molecule_graph, features, labels = data_set[0]

    assert molecule_graph is not None
    assert molecule_graph.number_of_nodes() == 2

    assert features.numpy().shape == (2, 4)

    assert "am1-charges" in labels
    assert labels["am1-charges"].numpy().shape == (2,)

    assert "am1-wbo" in labels
    assert labels["am1-wbo"].numpy().shape == (1,)


def test_data_set_loader():

    data_loader = MoleculeGraphDataLoader(
        dataset=MoleculeGraphDataset.from_molecules(
            [Molecule.from_smiles("C"), Molecule.from_smiles("C[O-]")],
            [AtomConnectivity()],
            [],
            label_function,
        ),
    )

    entries = [*data_loader]

    for graph, features, labels in entries:

        assert graph is not None and graph.number_of_nodes() == 5
        assert features.numpy().shape == (5, 4)
        assert "formal_charges" in labels
