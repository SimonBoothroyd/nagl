import numpy
import torch
from openff.toolkit.topology import Molecule
from simtk import unit

from nagl.dataset.dataset import (
    MoleculeGraphDataLoader,
    MoleculeGraphDataset,
    molecule_to_graph,
)
from nagl.dataset.features import AtomConnectivity, BondIsInRing


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
    assert len(molecule_graph) == 5

    assert features.numpy().shape == (5, 4)

    assert "formal_charges" in labels
    label = labels["formal_charges"]

    assert label.numpy().shape == (5,)


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

        assert graph is not None and len(graph) == 5
        assert features.numpy().shape == (5, 4)
        assert "formal_charges" in labels
