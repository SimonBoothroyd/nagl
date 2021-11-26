import numpy
import pytest
import torch
import torch.optim

from nagl.lightning import MoleculeGCNLightningModel
from nagl.models import ConvolutionModule, ReadoutModule
from nagl.nn import SequentialLayers
from nagl.nn.gcn import GCNStack
from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures
from nagl.nn.process import ComputePartialCharges


@pytest.fixture()
def mock_atom_model() -> MoleculeGCNLightningModel:

    return MoleculeGCNLightningModel(
        convolution_module=ConvolutionModule("SAGEConv", in_feats=4, hidden_feats=[4]),
        readout_modules={
            "atom": ReadoutModule(
                pooling_layer=PoolAtomFeatures(),
                readout_layers=SequentialLayers(in_feats=4, hidden_feats=[2]),
                postprocess_layer=ComputePartialCharges(),
            ),
        },
        learning_rate=0.01,
    )


class TestMoleculeGCNLightningModel:
    def test_init(self):

        model = MoleculeGCNLightningModel(
            convolution_module=ConvolutionModule(
                "SAGEConv", in_feats=1, hidden_feats=[2, 2]
            ),
            readout_modules={
                "atom": ReadoutModule(
                    pooling_layer=PoolAtomFeatures(),
                    readout_layers=SequentialLayers(
                        in_feats=2, hidden_feats=[2], activation=["Identity"]
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
            learning_rate=0.01,
        )

        assert model.convolution_module is not None
        assert isinstance(model.convolution_module, ConvolutionModule)

        assert isinstance(model.convolution_module.gcn_layers, GCNStack)
        assert len(model.convolution_module.gcn_layers) == 2

        assert all(x in model.readout_modules for x in ["atom", "bond"])

        assert isinstance(model.readout_modules["atom"].pooling_layer, PoolAtomFeatures)
        assert isinstance(model.readout_modules["bond"].pooling_layer, PoolBondFeatures)

        assert numpy.isclose(model.learning_rate, 0.01)

    def test_forward(self, mock_atom_model, dgl_methane):

        output = mock_atom_model.forward(dgl_methane)
        assert "atom" in output

        assert output["atom"].shape == (5, 1)

    @pytest.mark.parametrize(
        "method_name", ["training_step", "validation_step", "test_step"]
    )
    def test_step(self, mock_atom_model, method_name, dgl_methane, monkeypatch):
        def mock_forward(_):
            return {"atom": torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])}

        monkeypatch.setattr(mock_atom_model, "forward", mock_forward)

        loss = getattr(mock_atom_model, method_name)(
            (dgl_methane, {"atom": torch.tensor([[2.0, 3.0, 4.0, 5.0, 6.0]])}), 0
        )
        assert torch.isclose(loss, torch.tensor([1.0]))

    def test_configure_optimizers(self, mock_atom_model):

        optimizer = mock_atom_model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Adam)
        assert torch.isclose(torch.tensor(optimizer.defaults["lr"]), torch.tensor(0.01))
