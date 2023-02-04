import numpy
import pyarrow
import torch
from rdkit import Chem

from nagl.datasets import DGLMoleculeDataset, collate_dgl_molecules
from nagl.features import AtomConnectivity, AtomicElement, BondIsInRing
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.utilities.molecule import molecule_from_mapped_smiles


def label_function(molecule: Chem.Mol):
    return {
        "formal_charges": torch.tensor(
            [atom.GetFormalCharge() for atom in molecule.GetAtoms()],
            dtype=torch.float,
        ),
    }


class TestDGLMoleculeDataset:
    def test_from_molecules(self, rdkit_methane):

        data_set = DGLMoleculeDataset.from_molecules(
            [rdkit_methane], [AtomConnectivity()], [BondIsInRing()], label_function
        )
        assert len(data_set) == 1

        dgl_molecule, labels = data_set[0]

        assert isinstance(dgl_molecule, DGLMolecule)
        assert dgl_molecule.n_atoms == 5

        assert "formal_charges" in labels
        label = labels["formal_charges"]

        assert label.numpy().shape == (5,)

    def test_from_unfeaturized(self, tmp_cwd):

        table = pyarrow.table(
            [
                ["[O-:1][H:2]", "[H:1][H:2]"],
                [numpy.arange(2).astype(float), numpy.zeros(2).astype(float)],
                [numpy.arange(2).astype(float) + 2, None],
            ],
            ["smiles", "charges-am1", "charges-am1bcc"],
        )

        parquet_path = tmp_cwd / "labels.parquet"
        pyarrow.parquet.write_table(table, parquet_path)

        data_set = DGLMoleculeDataset.from_unfeaturized(
            parquet_path,
            ["charges-am1bcc"],
            [AtomicElement(values=["H", "O"])],
            [BondIsInRing()],
        )

        assert len(data_set) == 2

        dgl_molecule, labels = data_set[0]

        assert isinstance(dgl_molecule, DGLMolecule)
        assert dgl_molecule.n_atoms == 2

        expected_features = numpy.array([[0.0, 1.0], [1.0, 0.0]])
        assert dgl_molecule.atom_features.shape == (2, 2)
        assert numpy.allclose(dgl_molecule.atom_features.numpy(), expected_features)

        assert {*labels} == {"charges-am1bcc"}
        charges = labels["charges-am1bcc"].numpy()

        assert charges.shape == (2,)
        assert numpy.allclose(charges, numpy.array([2.0, 3.0]))

        _, labels = data_set[1]
        assert labels["charges-am1bcc"] is None

    def test_from_featurized(self, tmp_cwd):

        table = pyarrow.table(
            [
                ["[O-:1][H:2]", "[H:1][H:2]"],
                [
                    numpy.array([[0.0, 1.0], [1.0, 0.0]]).flatten(),
                    numpy.array([[1.0, 0.0], [1.0, 0.0]]).flatten(),
                ],
                [numpy.array([[0.0]]).flatten(), numpy.array([[0.0]]).flatten()],
                [numpy.array([-1.0, 0.0]), numpy.array([0.0, 0.0])],
            ],
            ["smiles", "atom_features", "bond_features", "formal_charges"],
        )

        parquet_path = tmp_cwd / "labels.parquet"
        pyarrow.parquet.write_table(table, parquet_path)

        data_set = DGLMoleculeDataset.from_featurized(parquet_path, None)

        assert len(data_set) == 2

        dgl_molecule, labels = data_set[0]

        assert isinstance(dgl_molecule, DGLMolecule)
        assert dgl_molecule.n_atoms == 2
        assert dgl_molecule.n_bonds == 1

        expected_features = numpy.array([[0.0, 1.0], [1.0, 0.0]])
        assert dgl_molecule.atom_features.shape == expected_features.shape
        assert numpy.allclose(dgl_molecule.atom_features.numpy(), expected_features)

        assert {*labels} == {"formal_charges"}
        charges = labels["formal_charges"].numpy()

        assert charges.shape == (2,)
        assert numpy.allclose(charges, numpy.array([-1.0, 0.0]))

    def test_to_table(self, tmp_cwd):

        dataset = DGLMoleculeDataset.from_molecules(
            [
                molecule_from_mapped_smiles("[O-:1][H:2]"),
                molecule_from_mapped_smiles("[H:1][H:2]"),
            ],
            [AtomicElement(values=["H", "O"])],
            [BondIsInRing()],
            label_function,
        )

        expected_rows = [
            {
                "smiles": "[O-:1][H:2]",
                "atom_features": numpy.array([[0.0, 1.0], [1.0, 0.0]]).flatten(),
                "bond_features": numpy.array([[0.0]]).flatten(),
                "formal_charges": numpy.array([-1.0, 0.0]),
            },
            {
                "smiles": "[H:1][H:2]",
                "atom_features": numpy.array([[1.0, 0.0], [1.0, 0.0]]).flatten(),
                "bond_features": numpy.array([[0.0]]).flatten(),
                "formal_charges": numpy.array([0.0, 0.0]),
            },
        ]
        actual_table = dataset.to_table()
        actual_rows = actual_table.to_pylist()

        for expected_row, actual_row in zip(expected_rows, actual_rows):

            assert {*expected_row} == {*actual_row}
            assert expected_row.pop("smiles") == actual_row.pop("smiles")

            for column in expected_row:

                assert (
                    expected_row[column].shape == numpy.array(actual_row[column]).shape
                )
                assert numpy.allclose(
                    expected_row[column], numpy.array(actual_row[column])
                )


def test_collate_dgl_molecules():

    dataset = DGLMoleculeDataset.from_molecules(
        [
            molecule_from_mapped_smiles("[O-:1][H:2]"),
            molecule_from_mapped_smiles("[H:1][H:2]"),
        ],
        [AtomicElement(values=["H", "O"])],
        [],
        label_function,
    )
    entries = dataset._entries

    dgl_molecule, labels = collate_dgl_molecules(entries)

    assert isinstance(dgl_molecule, DGLMoleculeBatch)
    assert dgl_molecule.n_atoms_per_molecule == (2, 2)
    assert dgl_molecule.graph.batch_size == 2
    assert torch.allclose(dgl_molecule.graph.batch_num_nodes(), torch.tensor([2, 2]))

    expected_features = numpy.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    actual_features = dgl_molecule.atom_features.numpy()

    assert expected_features.shape == actual_features.shape
    assert numpy.allclose(expected_features, actual_features)
