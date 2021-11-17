from nagl.models import ConvolutionConfig, MoleculeGCNModel, ReadoutConfig
from nagl.nn import SequentialConfig, SequentialLayers
from nagl.nn.gcn import SAGEConvStack
from nagl.nn.pooling import PoolAtomFeatures, PoolBondFeatures
from nagl.nn.process import ComputePartialCharges


def test_init_mol_graph():

    model = MoleculeGCNModel(
        convolution_config=ConvolutionConfig(in_feats=1, hidden_feats=[2, 2]),
        readout_configs={
            "atom": ReadoutConfig(
                pooling_layer=PoolAtomFeatures.Config(),
                readout_layers=SequentialConfig(
                    in_feats=2, hidden_feats=[2], activation=["Identity"]
                ),
                postprocess_layer=ComputePartialCharges.Config(),
            ),
            "bond": ReadoutConfig(
                pooling_layer=PoolBondFeatures.Config(
                    layers=SequentialConfig(in_feats=4, hidden_feats=[4])
                ),
                readout_layers=SequentialConfig(in_feats=4, hidden_feats=[8]),
            ),
        },
    )

    assert model._convolution is not None
    assert isinstance(model._convolution, SAGEConvStack)

    assert all(x in model._pooling_layers for x in ["atom", "bond"])

    assert isinstance(model._pooling_layers["atom"], PoolAtomFeatures)
    assert isinstance(model._pooling_layers["bond"], PoolBondFeatures)

    assert all(x in model._readouts for x in ["atom", "bond"])
    assert all(isinstance(x, SequentialLayers) for x in model._readouts.values())

    assert "atom" in model._postprocess_layers
    assert "bond" not in model._postprocess_layers


def test_mol_graph_forward(dgl_methane):

    model = MoleculeGCNModel(
        convolution_config=ConvolutionConfig(in_feats=4, hidden_feats=[4]),
        readout_configs={
            "atom": ReadoutConfig(
                pooling_layer=PoolAtomFeatures.Config(),
                readout_layers=SequentialConfig(in_feats=4, hidden_feats=[2]),
                postprocess_layer=ComputePartialCharges.Config(),
            ),
        },
    )

    output = model.forward(dgl_methane)
    assert "atom" in output

    assert output["atom"].shape == (5, 1)
