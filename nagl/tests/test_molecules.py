import dgl
import numpy
import pytest
from openff.toolkit.topology import Molecule

from nagl.features import AtomConnectivity, BondOrder
from nagl.molecules import DGLMolecule, DGLMoleculeBatch


class TestBaseDGLModel:
    def test_graph_property(self, dgl_methane):
        assert isinstance(dgl_methane.graph, dgl.DGLGraph)

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

        dgl_molecule = DGLMolecule.from_smiles("[H]C(=O)[O-]", [], [])

        assert dgl_molecule.n_atoms == 4
        assert dgl_molecule.n_bonds == 3
        assert dgl_molecule.n_representations == 1

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
            [BondOrder()],
        )
        dgl_graph = dgl_molecule.graph

        node_features = dgl_molecule.atom_features
        assert node_features.shape == (4, 4)

        assert numpy.allclose(
            node_features,
            numpy.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        n_bonds = int(dgl_graph.number_of_edges() / 2)

        assert dgl_graph.edata["mask"][:n_bonds].all()
        assert (~dgl_graph.edata["mask"][n_bonds:]).all()

        forward_features = dgl_graph.edata["feat"][dgl_graph.edata["mask"]].numpy()
        reverse_features = dgl_graph.edata["feat"][~dgl_graph.edata["mask"]].numpy()

        assert forward_features.shape == reverse_features.shape
        assert forward_features.shape == (3, 3)

        assert numpy.allclose(forward_features, reverse_features)
        assert not numpy.allclose(
            forward_features[:], numpy.zeros_like(forward_features[:])
        )


class TestDGLMoleculeBatch:
    def test_init(self):

        batch = DGLMoleculeBatch(
            DGLMolecule.from_smiles("C", [], []),
            DGLMolecule.from_smiles("CC", [], []),
        )

        assert batch.graph.batch_size == 2

        assert batch.n_atoms_per_molecule == (5, 8)
        assert batch.n_representations_per_molecule == (1, 1)
