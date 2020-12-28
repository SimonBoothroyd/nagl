import numpy
import torch
import torch.nn

from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures


def test_pool_atom_features(methane_graph):

    methane_graph.ndata["h"] = torch.from_numpy(numpy.arange(5))
    atom_features = PoolAtomFeatures().forward(methane_graph)

    assert numpy.allclose(methane_graph.ndata["h"].numpy(), atom_features.numpy())


def test_pool_bond_features(methane_graph):

    bond_pool_layer = PoolBondFeatures(torch.nn.Identity(8))

    methane_graph.ndata["h"] = torch.from_numpy(numpy.arange(5).reshape(-1, 1))

    bond_features = bond_pool_layer.forward(methane_graph).numpy()
    assert numpy.allclose(bond_features[:, 0], bond_features[:, 1])
