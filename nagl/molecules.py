import copy
import typing

import dgl.function
import torch
from rdkit import Chem

from nagl.features import AtomFeature, AtomFeaturizer, BondFeature, BondFeaturizer
from nagl.utilities.molecule import (
    BOND_ORDER_TO_TYPE,
    BOND_TYPE_TO_ORDER,
    molecule_from_smiles,
)

_T = typing.TypeVar("_T")

MoleculeToDGLFunc = typing.Callable[
    [Chem.Mol, typing.List[AtomFeature], typing.List[BondFeature]], "DGLMolecule"
]


class _BaseDGLModel:
    @property
    def graph(self) -> dgl.DGLGraph:
        """Returns the DGL graph representation of the molecule."""
        return self._graph

    @property
    def atom_features(self) -> typing.Optional[torch.Tensor]:
        """Returns a tensor containing the initial atom features with
        shape=(n_atoms, n_atom_features)."""
        return (
            None
            if "feat" not in self._graph.ndata
            else self._graph.ndata["feat"].float()
        )

    @property
    def bond_features(self) -> typing.Optional[torch.Tensor]:
        """Returns a tensor containing the initial bond features with
        shape=(n_atoms, n_bond_features)."""

        return (
            None
            if "feat" not in self._graph.edata
            else self._graph.edata["feat"][self._graph.edata["mask"], :].float()
        )

    def __init__(self, graph: dgl.DGLGraph):
        """

        Args:
            graph: The DGL graph representation of the molecule.
        """

        self._graph: dgl.DGLGraph = graph

    def to(self: _T, device: str) -> _T:
        return_value = copy.copy(self)
        return_value._graph = self._graph.to(device)

        return return_value


class DGLMolecule(_BaseDGLModel):
    """A wrapper around a DGL graph representation of a molecule that stores additional
    metadata such as the number of different representations (e.g. resonance structures)
    and the number of atoms in the molecule."""

    @property
    def n_atoms(self) -> int:
        """The number of atoms in the stored molecule."""
        return int(self._graph.number_of_nodes() / self._n_representations)

    @property
    def n_bonds(self) -> int:
        """The number of bonds in the stored molecule."""
        return int(self._graph.number_of_edges() / 2 / self._n_representations)

    @property
    def n_representations(self) -> int:
        """Returns the number of different 'representations' (e.g. resonance structures)
        this molecule object contains
        """
        return self._n_representations

    def __init__(self, graph: dgl.DGLGraph, n_representations: int):
        """

        Args:
            n_representations: The number of different representations (e.g. resonance
                structures) present in the graph representation as disconnected
                sub-graphs.
        """
        super().__init__(graph)

        self._n_representations = n_representations

    @classmethod
    def from_rdkit(
        cls,
        molecule: Chem.Mol,
        atom_features: typing.Optional[typing.List[AtomFeature]] = None,
        bond_features: typing.Optional[typing.List[BondFeature]] = None,
        atom_feature_tensor: typing.Optional[torch.Tensor] = None,
        bond_feature_tensor: typing.Optional[torch.Tensor] = None,
    ) -> "DGLMolecule":
        """Creates a new DGL graph molecule representation from an RDKit molecule
        object.

        Args:
            molecule: The molecule to store in the graph.
            atom_features: The atom features to compute for the molecule.
            bond_features: The bond features to compute for the molecule.
            atom_feature_tensor: The (optional) pre-computed atom features. This
                option is mutually exclusive with ``atom_features``.
            bond_feature_tensor: The (optional) pre-computed bond features This
                option is mutually exclusive with ``bond_features``.

        Returns:
            The constructed graph.
        """

        assert (
            atom_features is None or atom_feature_tensor is None
        ), "``atom_features`` and ``atom_feature_tensor`` are mutually exclusive."
        assert (
            bond_features is None or bond_feature_tensor is None
        ), "``bond_features`` and ``bond_feature_tensor`` are mutually exclusive."

        molecule = Chem.Mol(molecule)
        Chem.Kekulize(molecule)

        # Create the bond tensors.
        indices_a = []
        indices_b = []

        for bond in molecule.GetBonds():
            indices_a.append(bond.GetBeginAtomIdx())
            indices_b.append(bond.GetEndAtomIdx())

        indices_a = torch.tensor(indices_a, dtype=torch.int32)
        indices_b = torch.tensor(indices_b, dtype=torch.int32)

        undirected_indices_a = torch.cat([indices_a, indices_b])
        undirected_indices_b = torch.cat([indices_b, indices_a])

        # Map the bond indices to a molecule graph, making sure to make the graph
        # undirected.
        graph: dgl.DGLGraph = dgl.graph((undirected_indices_a, undirected_indices_b))

        # Track which edges correspond to an original bond and which were added to
        # make the graph undirected.
        bond_mask = torch.tensor(
            [True] * molecule.GetNumBonds() + [False] * molecule.GetNumBonds(),
            dtype=torch.bool,
        )
        graph.edata["mask"] = bond_mask

        if atom_features is not None and len(atom_features) > 0:
            atom_feature_tensor = AtomFeaturizer.featurize(molecule, atom_features)
        if atom_feature_tensor is not None:
            graph.ndata["feat"] = atom_feature_tensor

        if bond_features is not None and len(bond_features) > 0:
            bond_feature_tensor = BondFeaturizer.featurize(molecule, bond_features)
        if bond_feature_tensor is not None:
            # We need to 'double stack' the features as the graph is bidirectional
            graph.edata["feat"] = torch.cat([bond_feature_tensor, bond_feature_tensor])

        graph.ndata["formal_charge"] = torch.tensor(
            [atom.GetFormalCharge() for atom in molecule.GetAtoms()],
            dtype=torch.int8,
        )
        graph.ndata["atomic_number"] = torch.tensor(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()], dtype=torch.uint8
        )

        bond_orders = torch.tensor(
            [BOND_TYPE_TO_ORDER[bond.GetBondType()] for bond in molecule.GetBonds()],
            dtype=torch.uint8,
        )
        graph.edata["bond_order"] = torch.cat([bond_orders, bond_orders])

        return cls(graph, 1)

    @classmethod
    def from_smiles(
        cls: typing.Type["DGLMolecule"],
        smiles: str,
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
    ) -> "DGLMolecule":
        """Creates a new molecular graph representation from a SMILES pattern.

        Args:
            smiles: The SMILES representation of the molecule to store in the graph.
            atom_features: The atom features to compute for the molecule.
            bond_features: The bond features to compute for the molecule.

        Returns:
            The constructed graph.
        """

        return cls.from_rdkit(
            molecule_from_smiles(smiles),
            atom_features,
            bond_features,
        )

    def to_rdkit(self) -> Chem.Mol:
        """Converts this DGL molecule into an RDkit molecule object."""

        molecule = Chem.RWMol()

        atomic_numbers = self.graph.ndata["atomic_number"].detach().numpy().tolist()
        formal_charges = self.graph.ndata["formal_charge"].detach().numpy().tolist()

        for i, (atomic_num, formal_charge) in enumerate(
            zip(atomic_numbers, formal_charges)
        ):
            atom = Chem.Atom(int(atomic_num))
            atom.SetFormalCharge(int(formal_charge))

            index = molecule.AddAtom(atom)
            assert index == i

        indices_a, indices_b = self.graph.all_edges()

        for index_a, index_b, bond_order in zip(
            indices_a[self.graph.edata["mask"]],
            indices_b[self.graph.edata["mask"]],
            self.graph.edata["bond_order"][self.graph.edata["mask"]],
        ):
            molecule.AddBond(
                int(index_a), int(index_b), BOND_ORDER_TO_TYPE[int(bond_order)]
            )

        molecule = Chem.Mol(molecule)

        Chem.SanitizeMol(molecule)
        Chem.SetAromaticity(molecule, Chem.AROMATICITY_RDKIT)

        return molecule


class DGLMoleculeBatch(_BaseDGLModel):
    """A wrapper around a batch of DGL molecule objects."""

    @property
    def n_atoms_per_molecule(self) -> typing.Tuple[int, ...]:
        """Returns the number of atoms in each unique molecule in the batch."""
        return self._n_atoms

    @property
    def n_representations_per_molecule(self) -> typing.Tuple[int, ...]:
        """Returns the number of different 'representations' (e.g. resonance structures)
        for each unique molecule in the batch.
        """
        return self._n_representations

    def __init__(self, *molecules: DGLMolecule):
        super(DGLMoleculeBatch, self).__init__(
            dgl.batch([molecule.graph for molecule in molecules])
        )

        self._n_atoms = tuple(molecule.n_atoms for molecule in molecules)
        self._n_representations = tuple(
            molecule.n_representations for molecule in molecules
        )

    def unbatch(self) -> typing.List[DGLMolecule]:
        """Split this batch back into individual molecules."""

        return [
            DGLMolecule(graph, n_repr)
            for graph, n_repr in zip(dgl.unbatch(self._graph), self._n_representations)
        ]
