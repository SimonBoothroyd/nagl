import numpy
import pytest
import torch
import torch.nn
import torch.nn.functional
from dgl.nn.pytorch import SAGEConv

from nagl.nn.gcn import SAGEConvStack


def test_init_gcn_stack():

    conv_stack = SAGEConvStack(in_feats=1, hidden_feats=[2, 3])
    conv_stack.reset_parameters()

    assert len(conv_stack) == 2

    assert all(isinstance(x, SAGEConv) for x in conv_stack)

    assert numpy.isclose(conv_stack[0].feat_drop.p, 0.0)
    assert conv_stack[0].fc_self.in_features == 1
    assert conv_stack[0].fc_self.out_features == 2

    assert numpy.isclose(conv_stack[1].feat_drop.p, 0.0)
    assert conv_stack[1].fc_self.in_features == 2
    assert conv_stack[1].fc_self.out_features == 3


def test_init_sequential_layers_inputs():

    conv_stack = SAGEConvStack(
        in_feats=2,
        hidden_feats=[3],
        activation=[torch.nn.functional.leaky_relu],
        dropout=[0.5],
        aggregator_type=["lstm"],
    )
    assert len(conv_stack) == 1
    assert all(isinstance(x, SAGEConv) for x in conv_stack)

    assert numpy.isclose(conv_stack[0].feat_drop.p, 0.5)
    assert conv_stack[0].lstm.input_size == 2
    assert conv_stack[0].lstm.hidden_size == 2
    assert conv_stack[0].fc_neigh.out_features == 3
    assert conv_stack[0].activation == torch.nn.functional.leaky_relu


def test_init_sequential_layers_invalid():

    with pytest.raises(ValueError) as error_info:
        SAGEConvStack(in_feats=1, hidden_feats=[2, 3], dropout=[0.5])

    assert "`hidden_feats`, `activation`, `dropout` and `aggregator_type` must " in str(
        error_info.value
    )


def test_gcn_stack_forward(dgl_methane):

    homograph = dgl_methane.homograph

    conv_stack = SAGEConvStack(in_feats=4, hidden_feats=[2])
    h = conv_stack.forward(homograph, dgl_methane.atom_features)

    assert h.detach().numpy().shape == (5, 2)
