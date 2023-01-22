import copy

import dgl
import numpy
import pytest
import torch
import torch.nn

from nagl.nn.pooling import AtomPoolingLayer, BondPoolingLayer, get_pooling_layer


def test_pool_atom_features(dgl_methane):

    dgl_methane.graph.ndata["h"] = torch.from_numpy(numpy.arange(5))
    atom_features = AtomPoolingLayer().forward(dgl_methane)

    assert numpy.allclose(dgl_methane.graph.ndata["h"].numpy(), atom_features.numpy())


def test_pool_bond_features(dgl_methane):

    molecule_a = dgl_methane

    molecule_b = copy.deepcopy(molecule_a)
    molecule_b._graph = dgl.reverse(molecule_a.graph, copy_edata=True)

    bond_pool_layer = BondPoolingLayer(torch.nn.Linear(12, 2))

    molecule_a.graph.ndata["h"] = torch.from_numpy(
        numpy.arange(30).reshape(-1, 6)
    ).float()
    molecule_b.graph.ndata["h"] = torch.clone(molecule_a.graph.ndata["h"])

    bond_features_a = bond_pool_layer.forward(molecule_a).detach().numpy()
    bond_features_b = bond_pool_layer.forward(molecule_b).detach().numpy()

    assert not numpy.allclose(bond_features_a, 0.0)
    assert numpy.allclose(bond_features_a, bond_features_b)


@pytest.mark.parametrize(
    "type_, expected_class", [("atom", AtomPoolingLayer), ("bond", BondPoolingLayer)]
)
def test_get_pooling_layer(type_, expected_class):
    assert get_pooling_layer(type_) == expected_class
