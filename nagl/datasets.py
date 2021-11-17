import functools
import logging
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Dict,
    List,
    NamedTuple,
    Optional,
    Type,
    Union,
)

import dgl
import numpy
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nagl.features import AtomFeature, BondFeature
from nagl.molecules import DGLMolecule, DGLMoleculeBatch
from nagl.storage.storage import ChargeMethod, MoleculeStore, WBOMethod

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

logger = logging.getLogger(__name__)


class DGLMoleculeDatasetEntry(NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: Dict[str, torch.Tensor]


class DGLMoleculeDataset(Dataset):
    """A data set which stores a featurized graph representation of a labelled set of
    molecules."""

    @property
    def n_features(self) -> int:
        """Returns the number of atom features"""
        return 0 if len(self) == 0 else self[0][0].atom_features.shape[1]

    def __init__(self, entries: List[DGLMoleculeDatasetEntry]):
        """
        Args:
            entries: The list of entries to add to the data set.
        """
        self._entries = entries

    @classmethod
    def from_molecules(
        cls: Type["DGLMoleculeDataset"],
        molecules: Collection["Molecule"],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: Callable[["Molecule"], Dict[str, torch.Tensor]],
        enumerate_resonance: bool = False,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from a specified list of molecule objects.

        Args:
            molecules: The molecules to load into the set.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            enumerate_resonance: Whether to enumerate the lowest energy resonance
                structures of each molecule and store each within the DGL graph
                representation.
        """

        entries = [
            cls._build_entry(
                molecule,
                atom_features=atom_features,
                bond_features=bond_features,
                label_function=label_function,
                enumerate_resonance=enumerate_resonance,
            )
            for molecule in tqdm(molecules)
        ]

        return cls(entries)

    @classmethod
    def from_smiles(
        cls: Type["DGLMoleculeDataset"],
        smiles: Collection[str],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: Callable[["Molecule"], Dict[str, torch.Tensor]],
        enumerate_resonance: bool = False,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from a specified list of SMILES patterns.

        Args:
            smiles: The SMILES representations of the molecules to load into the set.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            enumerate_resonance: Whether to enumerate the lowest energy resonance
                structures of each molecule and store each within the DGL graph
                representation.
        """

        from openff.toolkit.topology import Molecule

        return cls.from_molecules(
            [Molecule.from_smiles(pattern) for pattern in smiles],
            atom_features,
            bond_features,
            label_function,
            enumerate_resonance,
        )

    @classmethod
    def _labelled_molecule_to_dict(
        cls,
        molecule: "Molecule",
        partial_charge_method: Optional[ChargeMethod],
        bond_order_method: Optional[WBOMethod],
    ) -> Dict[str, torch.Tensor]:
        """A convenience method for mapping a pre-labelled molecule to a dictionary
        of label tensors.

        Args:
            molecule: The labelled molecule object.
            partial_charge_method: The method which was used to generate the partial
                charge on each atom, or ``None`` if charge labels should not be included.
            bond_order_method: The method which was used to generate the Wiberg bond
                orders of each bond, or ``None`` if WBO labels should not be included.

        Returns:
            A dictionary of the tensor labels.
        """
        from simtk import unit

        labels = {}

        if partial_charge_method is not None:
            labels[f"{partial_charge_method}-charges"] = torch.tensor(
                [
                    atom.partial_charge.value_in_unit(unit.elementary_charge)
                    for atom in molecule.atoms
                ],
                dtype=torch.float,
            )

        if bond_order_method is not None:

            labels[f"{bond_order_method}-wbo"] = torch.tensor(
                [bond.fractional_bond_order for bond in molecule.bonds],
                dtype=torch.float,
            )

        return labels

    @classmethod
    def from_molecule_stores(
        cls: Type["DGLMoleculeDataset"],
        molecule_stores: Union[MoleculeStore, Collection[MoleculeStore]],
        partial_charge_method: Optional[ChargeMethod],
        bond_order_method: Optional[WBOMethod],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
    ) -> "DGLMoleculeDataset":
        """Creates a data set from a specified set of labelled molecule stores.

        Args:
            molecule_stores: The molecule stores which contain the pre-labelled
                molecules.
            partial_charge_method: The partial charge method to label each atom using.
                If ``None``, atoms won't be labelled with partial charges.
            bond_order_method: The Wiberg bond order method to label each bond using.
                If ``None``, bonds won't be labelled with WBOs.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
        """

        from openff.toolkit.topology import Molecule
        from simtk import unit

        assert partial_charge_method is not None or bond_order_method is not None, (
            "at least one of the ``partial_charge_method`` and  ``bond_order_method`` "
            "must not be ``None``."
        )

        if isinstance(molecule_stores, MoleculeStore):
            molecule_stores = [molecule_stores]

        stored_records = (
            record
            for molecule_store in molecule_stores
            for record in molecule_store.retrieve(
                partial_charge_method, bond_order_method
            )
        )

        entries = []

        for record in tqdm(stored_records):

            molecule: Molecule = Molecule.from_mapped_smiles(record.smiles)

            if partial_charge_method is not None:

                average_partial_charges = (
                    numpy.mean(
                        [
                            charge_set.values
                            for conformer in record.conformers
                            for charge_set in conformer.partial_charges
                            if charge_set.method == partial_charge_method
                        ],
                        axis=0,
                    )
                    * unit.elementary_charge
                )

                molecule.partial_charges = average_partial_charges

            if bond_order_method is not None:

                bond_order_value_tuples = [
                    value_tuple
                    for conformer in record.conformers
                    for bond_order_set in conformer.bond_orders
                    if bond_order_set.method == bond_order_method
                    for value_tuple in bond_order_set.values
                ]

                bond_orders = defaultdict(list)

                for index_a, index_b, value in bond_order_value_tuples:
                    bond_orders[tuple(sorted([index_a, index_b]))].append(value)

                for bond in molecule.bonds:

                    bond.fractional_bond_order = numpy.mean(
                        bond_orders[tuple(sorted([bond.atom1_index, bond.atom2_index]))]
                    )

            entries.append(
                cls._build_entry(
                    molecule,
                    atom_features,
                    bond_features,
                    functools.partial(
                        cls._labelled_molecule_to_dict,
                        partial_charge_method=partial_charge_method,
                        bond_order_method=bond_order_method,
                    ),
                )
            )

        return cls(entries)

    @classmethod
    def _build_entry(
        cls,
        molecule: "Molecule",
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: Callable[["Molecule"], Dict[str, torch.Tensor]],
        enumerate_resonance: bool = False,
    ) -> DGLMoleculeDatasetEntry:
        """Maps a molecule into a labeled, featurized graph representation.

        Args:
            molecule: The molecule.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            enumerate_resonance: Whether to enumerate the lowest energy resonance
                structures of each molecule and store each within the DGL graph
                representation.

        Returns:
            A named tuple containing the featurized molecule graph, a tensor of the atom
            features, and a tensor of the atom labels for the molecule.
        """
        label = label_function(molecule)

        # Map the molecule to a graph and assign features.
        dgl_molecule = DGLMolecule.from_openff(
            molecule, atom_features, bond_features, enumerate_resonance
        )

        return DGLMoleculeDatasetEntry(dgl_molecule, label)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> DGLMoleculeDatasetEntry:
        return self._entries[index]


class DGLMoleculeDataLoader(DataLoader):
    """A custom data loader for batching ``DGLMoleculeDataset`` objects."""

    def __init__(
        self,
        dataset: DGLMoleculeDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate(graph_entries: List[DGLMoleculeDatasetEntry]):

            if isinstance(graph_entries[0], dgl.DGLGraph):
                graph_entries = [graph_entries]

            molecules, labels = zip(*graph_entries)

            batched_molecules = DGLMoleculeBatch(*molecules)
            batched_labels = {}

            for label_name in labels[0]:

                batched_labels[label_name] = torch.vstack(
                    [label[label_name].reshape(-1, 1) for label in labels]
                )

            return batched_molecules, batched_labels

        super(DGLMoleculeDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )
