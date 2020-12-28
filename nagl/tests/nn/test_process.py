import numpy
import torch

from nagl.nn.process import ComputePartialCharges


def test_atomic_parameters_to_charges_neutral(methane_graph):

    partial_charges = ComputePartialCharges.atomic_parameters_to_charges(
        electronegativity=torch.tensor([30.8, 27.4, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 73.9, 73.9, 73.9, 73.9]),
        total_charge=0.0,
    ).numpy()

    assert numpy.isclose(partial_charges.sum(), 0.0)
    assert numpy.allclose(partial_charges[1:], partial_charges[1])


def test_atomic_parameters_to_charges_charged(methane_graph):

    partial_charges = ComputePartialCharges.atomic_parameters_to_charges(
        electronegativity=torch.tensor([30.8, 49.3, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 25.0, 73.9, 73.9, 73.9]),
        total_charge=-1.0,
    ).numpy()

    assert numpy.isclose(partial_charges.sum(), -1.0)
    assert not numpy.allclose(partial_charges[1:], partial_charges[1])
    assert numpy.allclose(partial_charges[2:], partial_charges[2])


def test_compute_charges_forward(methane_graph):
    input = torch.tensor(
        [[30.8, 78.4], [27.4, 73.9], [27.4, 73.9], [27.4, 73.9], [27.4, 73.9]]
    )
    partial_charges = ComputePartialCharges().forward(methane_graph, input)

    assert numpy.isclose(partial_charges.sum(), 0.0)
    assert numpy.allclose(partial_charges[1:], partial_charges[1])
