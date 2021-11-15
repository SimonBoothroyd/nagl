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
    Union,
)

import dgl
import numpy
import torch
from openff.utilities import requires_package
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from nagl.dataset.features import (
    AtomFeature,
    AtomFeaturizer,
    BondFeature,
    BondFeaturizer,
)
from nagl.storage.storage import ChargeMethod, MoleculeStore, WBOMethod

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

logger = logging.getLogger(__name__)


@requires_package("openff.toolkit")
@requires_package("simtk")
def molecule_to_graph(
    molecule: "Molecule",
    atom_features: List[AtomFeature],
    bond_features: List[BondFeature],
) -> dgl.DGLGraph:
    """Maps an OpenFF molecule object into a ``dgl`` graph complete with atom (node)
    and bond (edge) features.
    """
    from simtk import unit

    # Create the bond tensors.
    indices_a = []
    indices_b = []

    for bond in molecule.bonds:

        indices_a.append(bond.atom1_index)
        indices_b.append(bond.atom2_index)

    indices_a = torch.tensor(indices_a, dtype=torch.int32)
    indices_b = torch.tensor(indices_b, dtype=torch.int32)

    # Map the bond indices to a molecule graph, making sure to make the graph
    # undirected.
    molecule_graph = dgl.heterograph(
        {
            ("atom", "forward", "atom"): (indices_a, indices_b),
            ("atom", "reverse", "atom"): (indices_b, indices_a),
        }
    )

    # Assign the atom (node) features.
    if len(atom_features) > 0:
        molecule_graph.ndata["feat"] = AtomFeaturizer.featurize(molecule, atom_features)

    molecule_graph.ndata["formal_charge"] = torch.tensor(
        [
            atom.formal_charge.value_in_unit(unit.elementary_charge)
            for atom in molecule.atoms
        ]
    )

    # Assign the bond (edge) features.
    if len(bond_features) > 0:

        feature_tensor = BondFeaturizer.featurize(molecule, bond_features)

        molecule_graph.edges["forward"].data["feat"] = feature_tensor
        molecule_graph.edges["reverse"].data["feat"] = feature_tensor

    return molecule_graph


class MoleculeGraphEntry(NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    graph: dgl.DGLGraph
    features: torch.Tensor
    labels: Dict[str, torch.Tensor]


class MoleculeGraphDataset(Dataset):
    """A data set which stores a featurized graph representation of a labelled set of
    molecules."""

    @property
    def n_features(self) -> int:
        """Returns the number of atom features"""
        return 0 if len(self) == 0 else self[0][1].shape[1]

    def __init__(self, entries: List[MoleculeGraphEntry]):
        """
        Args:
            entries: The list of entries to add to the data set.
        """
        self._entries = entries

    @classmethod
    def from_molecules(
        cls: "MoleculeGraphDataset",
        molecules: Collection["Molecule"],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: Callable[["Molecule"], Dict[str, torch.Tensor]],
    ) -> "MoleculeGraphDataset":
        """Creates a data set from a specified list of molecule objects.

        Args:
            molecules: The molecules to load into the set.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.
        """

        entries = [
            cls._build_entry(
                molecule,
                atom_features=atom_features,
                bond_features=bond_features,
                label_function=label_function,
            )
            for molecule in tqdm(molecules)
        ]

        return cls(entries)

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
        cls: "MoleculeGraphDataset",
        molecule_stores: Union[MoleculeStore, Collection[MoleculeStore]],
        partial_charge_method: Optional[ChargeMethod],
        bond_order_method: Optional[WBOMethod],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
    ) -> "MoleculeGraphDataset":
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
    ) -> MoleculeGraphEntry:
        """Maps a molecule into a labeled, featurized graph representation.

        Args:
            molecule: The molecule.
            atom_features: The atom features to compute for each molecule.
            bond_features: The bond features to compute for each molecule.
            label_function: A function which will return a label for a given molecule.
                The function should take a molecule as input, and return and tensor
                with shape=(n_atoms,) containing the label of each atom.

        Returns:
            A named tuple containing the featurized molecule graph, a tensor of the atom
            features, and a tensor of the atom labels for the molecule.
        """
        label = label_function(molecule)

        # Map the molecule to a graph and assign features.
        graph = molecule_to_graph(molecule, atom_features, bond_features)
        features = graph.ndata["feat"].float()

        return MoleculeGraphEntry(graph, features, label)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int) -> MoleculeGraphEntry:
        return self._entries[index]


class MoleculeGraphDataLoader(DataLoader):
    """A custom data loader for batching ``MoleculeGraphDataset`` objects."""

    def __init__(
        self,
        dataset: MoleculeGraphDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
    ):
        def collate(graph_entries: List[MoleculeGraphEntry]):
            graphs, features, labels = zip(*graph_entries)

            batched_graph = dgl.batch(graphs)
            batched_features = torch.vstack(features)
            batched_labels = {}

            for label_name in labels[0]:

                batched_labels[label_name] = torch.vstack(
                    [label[label_name].reshape(-1, 1) for label in labels]
                )

            return batched_graph, batched_features, batched_labels

        super(MoleculeGraphDataLoader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
        )
