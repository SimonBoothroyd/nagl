import torch.nn

from nagl.models import DGLMoleculeModel
from nagl.nn import Sequential
from nagl.nn.convolution import SAGEConvStack
from nagl.nn.pooling import AtomPoolingLayer, BondPoolingLayer
from nagl.nn.postprocess import PartialChargeLayer
from nagl.nn.readout import ReadoutModule


class TestDGLMoleculeModel:
    def test_init(self):
        model = DGLMoleculeModel(
            convolution_module=SAGEConvStack(in_feats=1, hidden_feats=[2, 2]),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=AtomPoolingLayer(),
                    forward_layers=Sequential(
                        in_feats=2, hidden_feats=[2], activation=[torch.nn.Identity()]
                    ),
                    postprocess_layer=PartialChargeLayer(),
                ),
                "bond": ReadoutModule(
                    pooling_layer=BondPoolingLayer(
                        layers=Sequential(in_feats=4, hidden_feats=[4])
                    ),
                    forward_layers=Sequential(in_feats=4, hidden_feats=[8]),
                ),
            },
        )

        assert isinstance(model.convolution_module, SAGEConvStack)
        assert len(model.convolution_module) == 2

        assert all(x in model.readout_modules for x in ["atom", "bond"])

        assert isinstance(model.readout_modules["atom"].pooling_layer, AtomPoolingLayer)
        assert isinstance(model.readout_modules["bond"].pooling_layer, BondPoolingLayer)

    def test_forward(self, dgl_methane):
        model = DGLMoleculeModel(
            convolution_module=SAGEConvStack(in_feats=4, hidden_feats=[4]),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=AtomPoolingLayer(),
                    forward_layers=Sequential(in_feats=4, hidden_feats=[2]),
                    postprocess_layer=PartialChargeLayer(),
                ),
            },
        )

        output = model.forward(dgl_methane)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)
