import numpy
import pytest
import torch
import torch.nn

from nagl.nn import SequentialLayers


def test_init_sequential_layers_default():

    sequential_layers = SequentialLayers(
        in_feats=1,
        hidden_feats=[2],
    )

    assert len(sequential_layers.layers) == 3

    assert isinstance(sequential_layers.layers[0], torch.nn.Linear)
    assert isinstance(sequential_layers.layers[1], torch.nn.ReLU)
    assert isinstance(sequential_layers.layers[2], torch.nn.Dropout)
    assert numpy.isclose(sequential_layers.layers[2].p, 0.0)


def test_init_sequential_layers_inputs():

    sequential_layers = SequentialLayers(
        in_feats=1,
        hidden_feats=[2, 1],
        activation=[torch.nn.ReLU(), torch.nn.LeakyReLU()],
        dropout=[0.0, 0.5],
    )

    assert len(sequential_layers.layers) == 6

    assert isinstance(sequential_layers.layers[0], torch.nn.Linear)
    assert isinstance(sequential_layers.layers[1], torch.nn.ReLU)
    assert isinstance(sequential_layers.layers[2], torch.nn.Dropout)
    assert numpy.isclose(sequential_layers.layers[2].p, 0.0)

    assert isinstance(sequential_layers.layers[3], torch.nn.Linear)
    assert isinstance(sequential_layers.layers[4], torch.nn.LeakyReLU)
    assert isinstance(sequential_layers.layers[5], torch.nn.Dropout)
    assert numpy.isclose(sequential_layers.layers[5].p, 0.5)


def test_init_sequential_layers_invalid():

    with pytest.raises(ValueError) as error_info:

        SequentialLayers(
            in_feats=1,
            hidden_feats=[2],
            activation=[torch.nn.ReLU(), torch.nn.LeakyReLU()],
            dropout=[0.0, 0.5],
        )

    assert "The `hidden_feats`, `activation`, and `dropout` lists must be the" in str(
        error_info.value
    )
