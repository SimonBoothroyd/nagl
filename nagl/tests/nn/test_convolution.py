import numpy
import pytest
import torch.nn
from dgl.nn.pytorch import SAGEConv

from nagl.nn.convolution import SAGEConvStack, get_convolution_layer


class TestGCNStack:
    def test_init(self):
        conv_stack = SAGEConvStack(
            in_feats=1,
            hidden_feats=[2, 3],
            activation=[torch.nn.SELU(), torch.nn.LeakyReLU()],
        )
        conv_stack.reset_parameters()

        assert all(isinstance(x, SAGEConv) for x in conv_stack)
        assert len(conv_stack) == 2

        assert isinstance(conv_stack[0].activation, torch.nn.SELU)
        assert numpy.isclose(conv_stack[0].feat_drop.p, 0.0)
        assert conv_stack[0].fc_self.in_features == 1
        assert conv_stack[0].fc_self.out_features == 2

        assert isinstance(conv_stack[1].activation, torch.nn.LeakyReLU)
        assert numpy.isclose(conv_stack[1].feat_drop.p, 0.0)
        assert conv_stack[1].fc_self.in_features == 2
        assert conv_stack[1].fc_self.out_features == 3

    def test_forward(self, dgl_methane):
        conv_stack = SAGEConvStack(in_feats=4, hidden_feats=[2])
        h = conv_stack.forward(dgl_methane.graph, dgl_methane.atom_features)

        assert h.detach().numpy().shape == (5, 2)


class TestSAGEConvStack:
    def test_init(self):
        conv_stack = SAGEConvStack(
            in_feats=2,
            hidden_feats=[3],
            activation=[torch.nn.LeakyReLU()],
            dropout=[0.5],
            aggregator=["lstm"],
        )
        assert len(conv_stack) == 1
        assert all(isinstance(x, SAGEConv) for x in conv_stack)

        assert numpy.isclose(conv_stack[0].feat_drop.p, 0.5)
        assert conv_stack[0].lstm.input_size == 2
        assert conv_stack[0].lstm.hidden_size == 2
        assert conv_stack[0].fc_neigh.out_features == 3
        assert isinstance(conv_stack[0].activation, torch.nn.LeakyReLU)

    def test_init_invalid(self):
        with pytest.raises(
            ValueError,
            match="`hidden_feats`, `activation`, `dropout` and `aggregator` must",
        ):
            SAGEConvStack(in_feats=1, hidden_feats=[2, 3], dropout=[0.5])


@pytest.mark.parametrize("type_, expected_class", [("SaGeConv", SAGEConvStack)])
def test_get_convolution_layer(type_, expected_class):
    assert get_convolution_layer(type_) == expected_class
