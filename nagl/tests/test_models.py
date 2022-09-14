import torch.nn

from nagl.models import ConvolutionModule, MoleculeGCNModel, ReadoutModule
from nagl.nn import SequentialLayers
from nagl.nn.gcn import GCNStack
from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures
from nagl.nn.postprocess import ComputePartialCharges


class TestMoleculeGCNModel:
    def test_init(self):

        model = MoleculeGCNModel(
            convolution_module=ConvolutionModule(
                "SAGEConv", in_feats=1, hidden_feats=[2, 2]
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers(
                        in_feats=2, hidden_feats=[2], activation=[torch.nn.Identity()]
                    ),
                    postprocess_layer=ComputePartialCharges(),
                ),
                "bond": ReadoutModule(
                    pooling_layer=PoolBondFeatures(
                        layers=SequentialLayers(in_feats=4, hidden_feats=[4])
                    ),
                    readout_layers=SequentialLayers(in_feats=4, hidden_feats=[8]),
                ),
            },
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, GCNStack)
        assert len(model.convolution_module.gcn_layers) == 2

        assert all(x in model.readout_modules for x in ["atom", "bond"])

        # check the activation function in the readout layer
        assert isinstance(
            model.readout_modules["atom"].readout_layers[1], torch.nn.Identity
        )
        assert isinstance(
            model.readout_modules["bond"].readout_layers[1], torch.nn.ReLU
        )
        assert isinstance(model.readout_modules["atom"].pooling_layer, PoolAtomFeatures)
        assert isinstance(model.readout_modules["bond"].pooling_layer, PoolBondFeatures)

    def test_forward(self, dgl_methane):

        model = MoleculeGCNModel(
            convolution_module=ConvolutionModule(
                "SAGEConv", in_feats=4, hidden_feats=[4]
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers(in_feats=4, hidden_feats=[2]),
                    postprocess_layer=ComputePartialCharges(),
                ),
            },
        )

        output = model.forward(dgl_methane)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)
