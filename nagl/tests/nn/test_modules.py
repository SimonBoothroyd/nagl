import dgl.nn.pytorch
import numpy
import torch.nn

from nagl.nn import Sequential
from nagl.nn.gcn import GCNStack
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import AtomPoolingLayer
from nagl.nn.postprocess import PartialChargeLayer


class TestConvolutionModule:
    def test_init(self):

        module = ConvolutionModule(
            "SAGEConv", 2, [2, 2], [torch.nn.SELU(), torch.nn.LeakyReLU()]
        )
        assert isinstance(module.gcn_layers, GCNStack)
        assert len(module.gcn_layers) == 2

        assert isinstance(module.gcn_layers[0], dgl.nn.pytorch.SAGEConv)
        assert isinstance(module.gcn_layers[0].activation, torch.nn.SELU)
        assert numpy.isclose(module.gcn_layers[0].feat_drop.p, 0.0)

        assert isinstance(module.gcn_layers[1], dgl.nn.pytorch.SAGEConv)
        assert isinstance(module.gcn_layers[1].activation, torch.nn.LeakyReLU)
        assert numpy.isclose(module.gcn_layers[1].feat_drop.p, 0.0)


class TestReadoutModule:
    def test_init(self):
        module = ReadoutModule(
            AtomPoolingLayer(), Sequential(1, [1]), PartialChargeLayer()
        )
        assert isinstance(module.pooling_layer, AtomPoolingLayer)
        assert isinstance(module.readout_layers, Sequential)
        assert isinstance(module.postprocess_layer, PartialChargeLayer)
