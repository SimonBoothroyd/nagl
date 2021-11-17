import numpy
import torch
import torch.nn

from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures


def test_pool_atom_features(dgl_methane):

    dgl_methane.graph.ndata["h"] = torch.from_numpy(numpy.arange(5))
    atom_features = PoolAtomFeatures().forward(dgl_methane)

    assert numpy.allclose(dgl_methane.graph.ndata["h"].numpy(), atom_features.numpy())


def test_pool_bond_features(dgl_methane):

    bond_pool_layer = PoolBondFeatures(torch.nn.Identity(8))

    dgl_methane.graph.ndata["h"] = torch.from_numpy(numpy.arange(5).reshape(-1, 1))

    bond_features = bond_pool_layer.forward(dgl_methane).numpy()

    assert not numpy.allclose(bond_features, 0.0)
    assert numpy.allclose(bond_features[:, 0], bond_features[:, 1])
