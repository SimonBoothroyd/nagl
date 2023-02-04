import dgl
import numpy
import pytest
from rdkit import Chem

from nagl.features import AtomConnectivity, BondOrder
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.utilities.molecule import molecule_from_mapped_smiles, molecule_from_smiles


class TestBaseDGLModel:
    def test_graph_property(self, dgl_methane):
        assert isinstance(dgl_methane.graph, dgl.DGLGraph)

    def test_features_property(self, dgl_methane):

        assert dgl_methane.atom_features.shape == (5, 4)
        assert numpy.allclose(
            dgl_methane.atom_features.numpy(),
            numpy.array(
                [
                    [0.0, 0.0, 0.0, 1.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

        assert dgl_methane.bond_features.shape == (4, 1)
        assert numpy.allclose(
            dgl_methane.bond_features.numpy(), numpy.array([[0.0], [0.0], [0.0], [0.0]])
        )

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
                DGLMolecule.from_rdkit,
                molecule_from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]"),
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

    @pytest.mark.parametrize("expected_smiles", ["C", "C[O-]", "C=O", "c1ccccc1"])
    def test_to_rdkit(self, expected_smiles):

        expected_smiles = Chem.MolToSmiles(molecule_from_smiles(expected_smiles))

        dgl_molecule = DGLMolecule.from_smiles(expected_smiles, [], [])

        rdkit_molecule = dgl_molecule.to_rdkit()
        assert Chem.MolToSmiles(rdkit_molecule) == expected_smiles


class TestDGLMoleculeBatch:
    def test_init(self):

        batch = DGLMoleculeBatch(
            DGLMolecule.from_smiles("C", [], []),
            DGLMolecule.from_smiles("CC", [], []),
        )

        assert batch.graph.batch_size == 2

        assert batch.n_atoms_per_molecule == (5, 8)
        assert batch.n_representations_per_molecule == (1, 1)

    def test_unbatch(self):

        original_smiles = ["[H]C([H])([H])[H]", "[H]C([H])([H])C([H])([H])[H]"]

        batch = DGLMoleculeBatch(
            *(DGLMolecule.from_smiles(smiles, [], []) for smiles in original_smiles)
        )

        unbatched = batch.unbatch()

        unbatched_smiles = [
            Chem.MolToSmiles(molecule.to_rdkit()) for molecule in unbatched
        ]
        assert unbatched_smiles == original_smiles
