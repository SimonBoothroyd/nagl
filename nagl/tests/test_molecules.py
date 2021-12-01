import dgl
import numpy
import pytest
import torch
from openff.toolkit.topology import Molecule

from nagl.features import AtomConnectivity, BondIsInRing
from nagl.molecules import DGLMolecule, DGLMoleculeBatch, _hetero_to_homo_graph


@pytest.mark.parametrize(
    "dgl_molecule",
    [
        DGLMolecule.from_smiles("C", [], []),
        DGLMolecule.from_smiles("C", [AtomConnectivity()], []),
        DGLMolecule.from_smiles("C", [], [BondIsInRing()]),
        DGLMolecule.from_smiles("C", [AtomConnectivity()], [BondIsInRing()]),
    ],
)
def test_hetero_to_homo_graph(dgl_molecule):

    heterograph: dgl.DGLHeteroGraph = dgl_molecule.graph
    homograph: dgl.DGLHeteroGraph = _hetero_to_homo_graph(heterograph)

    assert homograph.number_of_nodes() == 5
    assert homograph.number_of_edges() == 8  # 4 forward + 4 reverse

    indices_a, indices_b = homograph.edges()

    assert torch.allclose(indices_a[:4], indices_b[4:])
    assert torch.allclose(indices_b[4:], indices_a[:4])


class TestBaseDGLModel:
    def test_graph_property(self, dgl_methane):
        assert isinstance(dgl_methane.graph, dgl.DGLHeteroGraph)

    def test_homograph_property(self, dgl_methane):
        assert isinstance(dgl_methane.graph, dgl.DGLHeteroGraph)
        assert dgl_methane.homograph.is_homogeneous

    def test_atom_features_property(self, dgl_methane):
        assert dgl_methane.atom_features.shape == (5, 4)

    def test_to(self, dgl_methane):

        dgl_methane_to = dgl_methane.to("cpu")

        assert dgl_methane_to != dgl_methane
        assert dgl_methane_to._graph != dgl_methane._graph  # should be a copy.
        assert dgl_methane_to.n_atoms == 5
        assert dgl_methane_to.n_bonds == 4


class TestDGLMolecule:
    def test_n_properties(self):
        """Test that the number of atoms and bonds properties work correctly with
        multiple resonance structures"""

        dgl_molecule = DGLMolecule.from_smiles(
            "[H]C(=O)[O-]", [], [], enumerate_resonance=True
        )

        assert dgl_molecule.n_atoms == 4
        assert dgl_molecule.n_bonds == 3
        assert dgl_molecule.n_representations == 2

    @pytest.mark.parametrize(
        "from_function, input_object",
        [
            (
                DGLMolecule.from_openff,
                Molecule.from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]"),
            ),
            (DGLMolecule.from_smiles, "[H:1][C:2](=[O:3])[O-:4]"),
        ],
    )
    def test_from_xxx(self, from_function, input_object):

        # noinspection PyArgumentList
        dgl_molecule = from_function(
            input_object,
            [AtomConnectivity()],
            [BondIsInRing()],
            enumerate_resonance=True,
        )
        dgl_graph = dgl_molecule.graph

        node_features = dgl_molecule.atom_features
        assert node_features.shape == (8, 4)

        assert numpy.allclose(
            node_features,
            numpy.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
                * 2
            ),
        )

        forward_features = dgl_graph.edges["forward"].data["feat"].numpy()
        reverse_features = dgl_graph.edges["reverse"].data["feat"].numpy()

        assert forward_features.shape == reverse_features.shape
        assert forward_features.shape == (6, 1)

        assert numpy.allclose(forward_features, reverse_features)
        assert numpy.allclose(
            forward_features[:], numpy.zeros_like(forward_features[:])
        )


class TestDGLMoleculeBatch:
    def test_init(self):

        batch = DGLMoleculeBatch(
            DGLMolecule.from_smiles("C", [], []),
            DGLMolecule.from_smiles("CC", [], []),
            DGLMolecule.from_smiles("[H]C(=O)[O-]", [], [], True),
        )

        assert batch.graph.batch_size == 3

        assert batch.n_atoms_per_molecule == (5, 8, 4)
        assert batch.n_representations_per_molecule == (1, 1, 2)
