import dgl
import numpy
import pytest
import torch
from openff.toolkit.topology import Molecule

from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.nn.postprocess import ComputePartialCharges
from nagl.resonance import enumerate_resonance_forms


@pytest.fixture()
def dgl_carboxylate():

    molecule: Molecule = Molecule.from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]")

    resonance_forms = enumerate_resonance_forms(
        molecule, lowest_energy_only=True, as_dicts=False
    )

    graphs = [
        DGLMolecule._molecule_to_dgl(resonance_form, [], [])
        for resonance_form in resonance_forms
    ]

    graph = dgl.batch(graphs)

    graph.set_batch_num_nodes(graph.batch_num_nodes().sum().reshape((-1,)))
    graph.set_batch_num_edges(
        {
            e_type: graph.batch_num_edges(e_type).sum().reshape((-1,))
            for e_type in graph.canonical_etypes
        }
    )

    return DGLMolecule(graph, len(graphs))


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


def test_compute_charges_forward_batched(dgl_carboxylate):

    batch = DGLMoleculeBatch(dgl_carboxylate, DGLMolecule.from_smiles("[H]Cl", [], []))

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
