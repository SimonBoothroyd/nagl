import numpy
import pytest
import torch.nn

from nagl.nn.layers import Sequential


class TestSequential:
    def test_init_default(self):
        sequential = Sequential(in_feats=1, hidden_feats=[2])

        assert len(sequential) == 3

        assert isinstance(sequential[0], torch.nn.Linear)
        assert isinstance(sequential[1], torch.nn.ReLU)
        assert isinstance(sequential[2], torch.nn.Dropout)
        assert numpy.isclose(sequential[2].p, 0.0)

    def test_init(self):
        sequential_layers = Sequential(
            in_feats=1,
            hidden_feats=[2, 1],
            activation=[torch.nn.ReLU(), torch.nn.LeakyReLU()],
            dropout=[0.0, 0.5],
        )

        assert len(sequential_layers) == 6

        assert isinstance(sequential_layers[0], torch.nn.Linear)
        assert isinstance(sequential_layers[1], torch.nn.ReLU)
        assert isinstance(sequential_layers[2], torch.nn.Dropout)
        assert numpy.isclose(sequential_layers[2].p, 0.0)

        assert isinstance(sequential_layers[3], torch.nn.Linear)
        assert isinstance(sequential_layers[4], torch.nn.LeakyReLU)
        assert isinstance(sequential_layers[5], torch.nn.Dropout)
        assert numpy.isclose(sequential_layers[5].p, 0.5)

    def test_init_invalid_inputs(self):
        with pytest.raises(
            ValueError, match="`hidden_feats`, `activation`, and `dropout` must be"
        ):
            Sequential(
                in_feats=1,
                hidden_feats=[2],
                activation=[torch.nn.ReLU(), torch.nn.LeakyReLU()],
                dropout=[0.0, 0.5],
            )
