import functools
import logging
import pathlib
import typing

import dgl
import pyarrow.parquet
import torch
from rdkit import Chem
from torch.utils.data import Dataset

from nagl.features import AtomFeature, BondFeature
from nagl.molecules import DGLMolecule, DGLMoleculeBatch, MoleculeToDGLFunc
from nagl.utilities import get_map_func
from nagl.utilities.molecule import (
    molecule_from_mapped_smiles,
    molecule_to_mapped_smiles,
)

logger = logging.getLogger(__name__)


class DGLMoleculeDatasetEntry(typing.NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: typing.Dict[str, torch.Tensor]


class DGLMoleculeDataset(Dataset):
    """A data set which stores a featurized graph representation of a labelled set of
    molecules."""

    def __init__(self, entries: typing.List[DGLMoleculeDatasetEntry]):
        """
        Args:
            entries: The list of entries to add to the data set.
        """
        self._entries: typing.List[DGLMoleculeDatasetEntry] = entries

    @classmethod
    def from_molecules(
        cls: typing.Type["DGLMoleculeDataset"],
        molecules: typing.Collection[Chem.Mol],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        label_function: typing.Callable[[Chem.Mol], typing.Dict[str, torch.Tensor]],
        molecule_to_dgl: typing.Optional[MoleculeToDGLFunc] = None,
        progress_iterator: typing.Optional[typing.Any] = None,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from a specified list of molecule objects.
        Args:
            molecules: The molecules to load into the set.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            molecule_to_dgl: A (optional) callable to use when converting an RDkit
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_rdkit`` class method is used.
            progress_iterator: An iterator (each a ``tqdm``) to loop over all molecules
                using. This is useful if you wish to display a progress bar for example.
        """

        molecule_to_dgl = (
            DGLMolecule.from_rdkit if molecule_to_dgl is None else molecule_to_dgl
        )

        molecules = (
            molecules if progress_iterator is None else progress_iterator(molecules)
        )
        entries = []

        for molecule in molecules:
            label = label_function(molecule)
            dgl_molecule = molecule_to_dgl(molecule, atom_features, bond_features)

            entries.append(DGLMoleculeDatasetEntry(dgl_molecule, label))

        return cls(entries)

    @classmethod
    def _entry_from_unfeaturized(
        cls,
        labels: typing.Dict[str, typing.Any],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        molecule_to_dgl: typing.Optional[MoleculeToDGLFunc] = None,
    ):
        smiles = labels.pop("smiles")

        molecule = molecule_from_mapped_smiles(smiles)
        dgl_molecule = molecule_to_dgl(molecule, atom_features, bond_features)

        for label, value in labels.items():
            if value is None:
                continue

            labels[label] = torch.tensor(value)

        return DGLMoleculeDatasetEntry(dgl_molecule, labels)

    @classmethod
    def from_unfeaturized(
        cls: typing.Type["DGLMoleculeDataset"],
        paths: typing.Union[pathlib.Path, typing.List[pathlib.Path]],
        columns: typing.Optional[typing.List[str]],
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
        molecule_to_dgl: typing.Optional[MoleculeToDGLFunc] = None,
        progress_iterator: typing.Optional[typing.Any] = None,
        n_processes: int = 0,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from unfeaturized data stored in parquet file.

        The file *must* at minimum contain a ``smiles`` column that stores *mapped*
        SMILES patterns, and additionally columns containing the labels.

        Args:
            paths: The path(s) to the parquet file containing the data labels.
            columns: The columns (in addition to ``smiles``) to load from the file.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            molecule_to_dgl: A (optional) callable to use when converting an RDKit
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_rdkit`` class method is used.
            progress_iterator: An iterator (each a ``tqdm``) to loop over all molecules
                using. This is useful if you wish to display a progress bar for example.
            n_processes: The number of processes to distribute the creation over.
        """

        columns = None if columns is None else ["smiles"] + columns

        molecule_to_dgl = (
            DGLMolecule.from_rdkit if molecule_to_dgl is None else molecule_to_dgl
        )

        table = pyarrow.parquet.read_table(paths, columns=columns)

        label_list = table.to_pylist()
        label_list = (
            label_list if progress_iterator is None else progress_iterator(label_list)
        )

        featurize_func = functools.partial(
            cls._entry_from_unfeaturized,
            atom_features=atom_features,
            bond_features=bond_features,
            molecule_to_dgl=molecule_to_dgl,
        )

        with get_map_func(n_processes) as map_func:
            entries = list(map_func(featurize_func, label_list))

        return DGLMoleculeDataset(entries)

    @classmethod
    def _entry_from_featurized(cls, labels: typing.Dict[str, typing.Any]):
        smiles = labels.pop("smiles")

        atom_features = labels.pop("atom_features")
        bond_features = labels.pop("bond_features")

        molecule = molecule_from_mapped_smiles(smiles)

        atom_features = (
            None
            if atom_features is None
            else torch.tensor(atom_features).float().reshape(molecule.GetNumAtoms(), -1)
        )
        bond_features = (
            None
            if bond_features is None
            else torch.tensor(bond_features).float().reshape(molecule.GetNumBonds(), -1)
        )

        dgl_molecule = DGLMolecule.from_rdkit(
            molecule,
            atom_feature_tensor=atom_features,
            bond_feature_tensor=bond_features,
        )

        for label, value in labels.items():
            if value is None:
                continue

            labels[label] = torch.tensor(value)

        return DGLMoleculeDatasetEntry(dgl_molecule, labels)

    @classmethod
    def from_featurized(
        cls: typing.Type["DGLMoleculeDataset"],
        paths: typing.Union[pathlib.Path, typing.List[pathlib.Path]],
        columns: typing.Optional[typing.List[str]],
        progress_iterator: typing.Optional[typing.Any] = None,
        n_processes: int = 0,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from unfeaturized data stored in parquet file.

        The file *must* at minimum contain a ``smiles``, an ``atom_features``, and a
        ``bond_features`` column that stores *mapped* SMILES patterns, atom features and
        bon features respectively. It should additionally have columns containing the
        labels.

        Args:
            paths: The path(s) to the parquet file containing the featurized molecules
                and data labels.
            columns: The columns (in addition to ``smiles``, ``atom_features``,
                ``bond_features``) to load from the file.
            progress_iterator: An iterator (each a ``tqdm``) to loop over all molecules
                using. This is useful if you wish to display a progress bar for example.
            n_processes: The number of processes to distribute the creation over.
        """

        required_columns = ["smiles", "atom_features", "bond_features"]
        columns = None if columns is None else required_columns + columns

        table = pyarrow.parquet.read_table(paths, columns=columns)

        label_list = table.to_pylist()
        label_list = (
            label_list if progress_iterator is None else progress_iterator(label_list)
        )

        with get_map_func(n_processes) as map_func:
            entries = list(map_func(cls._entry_from_featurized, label_list))

        return DGLMoleculeDataset(entries)

    def to_table(self) -> pyarrow.Table:
        """Converts the dataset to a ``pyarrow`` table.

        The table will contain at minimum a ``smiles``, an ``atom_features``, and a
        ``bond_features column that stores the *mapped* SMILES patterns, atom features
        and bond features of each molecule in the set respectively. It will additionally
        have columns containing the labels.
        """

        dgl_molecule: DGLMolecule

        rows = []

        required_columns = ["smiles", "atom_features", "bond_features"]
        label_columns = [] if len(self._entries) == 0 else [*self._entries[0][1]]

        for dgl_molecule, labels in self._entries:
            rdkit_molecule = dgl_molecule.to_rdkit()
            smiles = molecule_to_mapped_smiles(rdkit_molecule)

            atom_features = (
                None
                if dgl_molecule.atom_features is None
                else dgl_molecule.atom_features.detach().numpy().flatten()
            )
            bond_features = (
                None
                if dgl_molecule.bond_features is None
                else dgl_molecule.bond_features.detach().numpy().flatten()
            )

            assert {*labels} == {*label_columns}

            rows.append(
                (
                    smiles,
                    atom_features,
                    bond_features,
                    *[labels[column].numpy().tolist() for column in label_columns],
                )
            )

        table = pyarrow.table([*zip(*rows)], required_columns + label_columns)
        return table

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> DGLMoleculeDatasetEntry:
        return self._entries[index]


def collate_dgl_molecules(
    entries: typing.Union[
        typing.Tuple[DGLMolecule, typing.List[DGLMoleculeDatasetEntry]],
        typing.List[typing.Tuple[dgl.DGLGraph, typing.List[DGLMoleculeDatasetEntry]]],
    ]
) -> typing.Tuple[DGLMoleculeBatch, typing.Dict[str, torch.Tensor]]:
    if isinstance(entries[0], (dgl.DGLGraph, DGLMolecule)):
        entries = [entries]

    molecules, labels = zip(*entries)

    batched_molecules = DGLMoleculeBatch(*molecules)
    batched_labels = {}

    for label_name in labels[0]:
        batched_labels[label_name] = torch.vstack(
            [label[label_name].reshape(-1, 1) for label in labels]
        )

    return batched_molecules, batched_labels
