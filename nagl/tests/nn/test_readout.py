from nagl.nn import Sequential
from nagl.nn.pooling import AtomPoolingLayer
from nagl.nn.postprocess import PartialChargeLayer
from nagl.nn.readout import ReadoutModule


class TestReadoutModule:
    def test_init(self):
        module = ReadoutModule(
            AtomPoolingLayer(), Sequential(1, [1]), PartialChargeLayer()
        )
        assert isinstance(module.pooling_layer, AtomPoolingLayer)
        assert isinstance(module.forward_layers, Sequential)
        assert isinstance(module.postprocess_layer, PartialChargeLayer)
