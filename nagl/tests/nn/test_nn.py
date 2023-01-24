import pytest
import torch.nn

import nagl.nn


@pytest.mark.parametrize(
    "type_, expected_class",
    [
        ("Identity", torch.nn.Identity),
        ("Tanh", torch.nn.Tanh),
        ("ReLU", torch.nn.ReLU),
        ("relu", torch.nn.ReLU),
        ("LeakyReLU", torch.nn.LeakyReLU),
        ("SELU", torch.nn.SELU),
        ("ELU", torch.nn.ELU),
        ("GELU", torch.nn.GELU),
    ],
)
def test_get_activation_func(type_, expected_class):
    assert nagl.nn.get_activation_func(type_) == expected_class
