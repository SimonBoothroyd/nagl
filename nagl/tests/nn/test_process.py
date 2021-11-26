import numpy
import torch
from openff.toolkit.topology import Molecule

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn.postprocess import ComputePartialCharges


def test_atomic_parameters_to_charges_neutral():

    partial_charges = ComputePartialCharges.atomic_parameters_to_charges(
        electronegativity=torch.tensor([30.8, 27.4, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 73.9, 73.9, 73.9, 73.9]),
        total_charge=0.0,
    ).numpy()

    assert numpy.isclose(partial_charges.sum(), 0.0)
    assert numpy.allclose(partial_charges[1:], partial_charges[1])


def test_atomic_parameters_to_charges_charged():

    partial_charges = ComputePartialCharges.atomic_parameters_to_charges(
        electronegativity=torch.tensor([30.8, 49.3, 27.4, 27.4, 27.4]),
        hardness=torch.tensor([78.4, 25.0, 73.9, 73.9, 73.9]),
        total_charge=-1.0,
    ).numpy()

    assert numpy.isclose(partial_charges.sum(), -1.0)
    assert not numpy.allclose(partial_charges[1:], partial_charges[1])
    assert numpy.allclose(partial_charges[2:], partial_charges[2])


def test_compute_charges_forward(dgl_methane):
    inputs = torch.tensor(
        [
            [30.8, 78.4],
            [27.4, 73.9],
            [27.4, 73.9],
            [27.4, 73.9],
            [27.4, 73.9],
        ]
    )
    partial_charges = ComputePartialCharges().forward(dgl_methane, inputs)

    assert numpy.isclose(partial_charges.sum(), 0.0)
    assert numpy.allclose(partial_charges[1:], partial_charges[1])


def test_compute_charges_forward_batched():

    batch = DGLMoleculeBatch(
        DGLMolecule.from_openff(
            Molecule.from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]"),
            [],
            [],
            enumerate_resonance=True,
        ),
        DGLMolecule.from_smiles("[H]Cl", [], []),
    )

    inputs = torch.tensor(
        [
            # [H]C(=O)O- form 1
            [30.0, 80.0],
            [35.0, 75.0],
            [40.0, 70.0],
            [50.0, 65.0],
            # [H]C(=O)O- form 2
            [30.0, 80.0],
            [35.0, 75.0],
            [50.0, 65.0],
            [40.0, 70.0],
            # [H]Cl
            [55.0, 60.0],
            [60.0, 55.0],
        ]
    )
    partial_charges = ComputePartialCharges().forward(batch, inputs)
    assert partial_charges.shape == (6, 1)

    assert numpy.isclose(partial_charges.sum(), -1.0)
    # The carboxylate oxygen charges should be identical.
    assert numpy.allclose(partial_charges[2], partial_charges[3])
