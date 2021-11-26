from nagl.nn import SequentialLayers
from nagl.nn.gcn import GCNStack
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import PoolAtomFeatures
from nagl.nn.process import ComputePartialCharges


class TestConvolutionModule:
    def test_init(self):

        module = ConvolutionModule("SAGEConv", 2, [2, 2], ["ReLU", "ReLU"])
        assert isinstance(module.gcn_layers, GCNStack)
        assert len(module.gcn_layers) == 2


class TestReadoutModule:
    def test_init(self):
        module = ReadoutModule(
            PoolAtomFeatures(), SequentialLayers(1, [1]), ComputePartialCharges()
        )
        assert isinstance(module.pooling_layer, PoolAtomFeatures)
        assert isinstance(module.readout_layers, SequentialLayers)
        assert isinstance(module.postprocess_layer, ComputePartialCharges)
