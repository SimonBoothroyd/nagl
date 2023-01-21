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
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from nagl.features import AtomFeature, BondFeature
from nagl.molecules import DGLMolecule, DGLMoleculeBatch, MoleculeToDGLFunc
from nagl.utilities.toolkits import capture_toolkit_warnings

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

    from nagl.storage import ChargeMethod, MoleculeStore, WBOMethod

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
        molecule_to_dgl: Optional[MoleculeToDGLFunc] = None,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from a specified list of molecule objects.

        Args:
            molecules: The molecules to load into the set.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            molecule_to_dgl: A (optional) callable to use when converting an OpenFF
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_openff`` class method is used.
        """

        entries = [
            cls._build_entry(
                molecule,
                atom_features=atom_features,
                bond_features=bond_features,
                label_function=label_function,
                molecule_to_dgl=molecule_to_dgl,
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
        molecule_to_dgl: Optional[MoleculeToDGLFunc] = None,
    ) -> "DGLMoleculeDataset":
        """Creates a data set from a specified list of SMILES patterns.

        Args:
            smiles: The SMILES representations of the molecules to load into the set.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            molecule_to_dgl: A (optional) callable to use when converting an OpenFF
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_openff`` class method is used.
        """

        from openff.toolkit.topology import Molecule

        return cls.from_molecules(
            [Molecule.from_smiles(pattern) for pattern in smiles],
            atom_features,
            bond_features,
            label_function,
            molecule_to_dgl,
        )

    @classmethod
    def _labelled_molecule_to_dict(
        cls,
        molecule: "Molecule",
        partial_charge_method: Optional["ChargeMethod"],
        bond_order_method: Optional["WBOMethod"],
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
        from openff.units import unit

        labels = {}

        if partial_charge_method is not None:
            labels[f"{partial_charge_method}-charges"] = torch.tensor(
                [
                    atom.partial_charge.m_as(unit.elementary_charge)
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
        molecule_stores: Union["MoleculeStore", Collection["MoleculeStore"]],
        partial_charge_method: Optional["ChargeMethod"],
        bond_order_method: Optional["WBOMethod"],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        molecule_to_dgl: Optional[MoleculeToDGLFunc] = None,
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
            molecule_to_dgl: A (optional) callable to use when converting an OpenFF
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_openff`` class method is used.
        """

        from openff.toolkit.topology import Molecule
        from openff.units import unit

        from nagl.storage import MoleculeStore

        assert partial_charge_method is not None or bond_order_method is not None, (
            "at least one of the ``partial_charge_method`` and  ``bond_order_method`` "
            "must not be ``None``."
        )

        if isinstance(molecule_stores, MoleculeStore):
            molecule_stores = [molecule_stores]

        stored_records = list(
            record
            for molecule_store in molecule_stores
            for record in molecule_store.retrieve(
                [] if partial_charge_method is None else partial_charge_method,
                [] if bond_order_method is None else bond_order_method,
            )
        )

        entries = []

        for record in tqdm(stored_records, desc="featurizing molecules"):

            with capture_toolkit_warnings():

                molecule: Molecule = Molecule.from_mapped_smiles(
                    record.smiles, allow_undefined_stereo=True
                )

            if partial_charge_method is not None:

                molecule.partial_charges = (
                    numpy.array(record.average_partial_charges(partial_charge_method))
                    * unit.elementary_charge
                )

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
                    molecule_to_dgl,
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
        molecule_to_dgl: Optional[MoleculeToDGLFunc] = None,
    ) -> DGLMoleculeDatasetEntry:
        """Maps a molecule into a labeled, featurized graph representation.

        Args:
            molecule: The molecule.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
            molecule_to_dgl: A (optional) callable to use when converting an OpenFF
                ``Molecule`` object to a ``DGLMolecule`` object. By default, the
                ``DGLMolecule.from_openff`` class method is used.

        Returns:
            A named tuple containing the featurized molecule graph, a tensor of the atom
            features, and a tensor of the atom labels for the molecule.
        """
        label = label_function(molecule)

        # Map the molecule to a graph and assign features.
        molecule_to_dgl = (
            DGLMolecule.from_openff if molecule_to_dgl is None else molecule_to_dgl
        )
        dgl_molecule = molecule_to_dgl(molecule, atom_features, bond_features)

        return DGLMoleculeDatasetEntry(dgl_molecule, label)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> DGLMoleculeDatasetEntry:
        return self._entries[index]


class DGLMoleculeDataLoader(DataLoader):
    """A custom data loader for batching ``DGLMoleculeDataset`` objects."""

    def __init__(
        self,
        dataset: Union[DGLMoleculeDataset, ConcatDataset],
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
