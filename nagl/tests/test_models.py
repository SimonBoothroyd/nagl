import torch.nn

from nagl.models import MoleculeGCNModel
from nagl.nn import Sequential
from nagl.nn.gcn import GCNStack
from nagl.nn.modules import ConvolutionModule, ReadoutModule
from nagl.nn.pooling import AtomPoolingLayer, BondPoolingLayer
from nagl.nn.postprocess import PartialChargeLayer


class TestMoleculeGCNModel:
    def test_init(self):

        model = MoleculeGCNModel(
            convolution_module=ConvolutionModule(
                "SAGEConv", in_feats=1, hidden_feats=[2, 2]
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=AtomPoolingLayer(),
                    readout_layers=Sequential(
                        in_feats=2, hidden_feats=[2], activation=[torch.nn.Identity()]
                    ),
                    postprocess_layer=PartialChargeLayer(),
                ),
                "bond": ReadoutModule(
                    pooling_layer=BondPoolingLayer(
                        layers=Sequential(in_feats=4, hidden_feats=[4])
                    ),
                    readout_layers=Sequential(in_feats=4, hidden_feats=[8]),
                ),
            },
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, GCNStack)
        assert len(model.convolution_module.gcn_layers) == 2

        assert all(x in model.readout_modules for x in ["atom", "bond"])

        assert isinstance(model.readout_modules["atom"].pooling_layer, AtomPoolingLayer)
        assert isinstance(model.readout_modules["bond"].pooling_layer, BondPoolingLayer)

    def test_forward(self, dgl_methane):

        model = MoleculeGCNModel(
            convolution_module=ConvolutionModule(
                "SAGEConv", in_feats=4, hidden_feats=[4]
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=AtomPoolingLayer(),
                    readout_layers=Sequential(in_feats=4, hidden_feats=[2]),
                    postprocess_layer=PartialChargeLayer(),
                ),
            },
        )

        output = model.forward(dgl_methane)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)
