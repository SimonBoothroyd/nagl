import copy
import typing

import dgl.function
import torch

from nagl.features import AtomFeature, AtomFeaturizer, BondFeature, BondFeaturizer

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

_T = typing.TypeVar("_T")

MoleculeToDGLFunc = typing.Callable[
    ["Molecule", typing.List[AtomFeature], typing.List[BondFeature]], "DGLMolecule"
]


class _BaseDGLModel:
    @property
    def graph(self) -> dgl.DGLGraph:
        """Returns the DGL graph representation of the molecule."""
        return self._graph

    @property
    def atom_features(self) -> torch.Tensor:
        """Returns a tensor containing the initial atom features with
        shape=(n_atoms, n_atom_features)."""
        return self._graph.ndata["feat"].float()

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
    def _molecule_to_dgl(
        cls,
        molecule: "Molecule",
        atom_featurizers: typing.List[AtomFeature],
        bond_featurizers: typing.List[BondFeature],
    ) -> dgl.DGLGraph:
        """Maps an OpenFF molecule object into a ``dgl`` graph complete with atom (node)
        and bond (edge) features.
        """

        from openff.units import unit

        # Create the bond tensors.
        indices_a = []
        indices_b = []

        for bond in molecule.bonds:
            indices_a.append(bond.atom1_index)
            indices_b.append(bond.atom2_index)

        indices_a = torch.tensor(indices_a, dtype=torch.int32)
        indices_b = torch.tensor(indices_b, dtype=torch.int32)

        undirected_indices_a = torch.cat([indices_a, indices_b])
        undirected_indices_b = torch.cat([indices_b, indices_a])

        # Map the bond indices to a molecule graph, making sure to make the graph
        # undirected.
        molecule_graph: dgl.DGLGraph = dgl.graph(
            (undirected_indices_a, undirected_indices_b)
        )

        # Track which edges correspond to an original bond and which were added to
        # make the graph undirected.
        bond_mask = torch.tensor(
            [True] * molecule.n_bonds + [False] * molecule.n_bonds, dtype=torch.bool
        )
        molecule_graph.edata["mask"] = bond_mask

        # Assign the atom (node) features.
        if len(atom_featurizers) > 0:

            molecule_graph.ndata["feat"] = AtomFeaturizer.featurize(
                molecule, atom_featurizers
            )

        if len(bond_featurizers) > 0:

            bond_features = BondFeaturizer.featurize(molecule, bond_featurizers)
            # We need to 'double stack' the features as the graph is bidirectional
            molecule_graph.edata["feat"] = torch.cat([bond_features, bond_features])

        molecule_graph.ndata["formal_charge"] = torch.tensor(
            [
                atom.formal_charge.m_as(unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.int8,
        )
        molecule_graph.ndata["atomic_number"] = torch.tensor(
            [atom.atomic_number for atom in molecule.atoms], dtype=torch.uint8
        )

        bond_orders = torch.tensor(
            [bond.bond_order for bond in molecule.bonds], dtype=torch.uint8
        )
        molecule_graph.edata["bond_order"] = torch.cat([bond_orders, bond_orders])

        return molecule_graph

    @classmethod
    def from_openff(
        cls: typing.Type["DGLMolecule"],
        molecule: "Molecule",
        atom_features: typing.List[AtomFeature],
        bond_features: typing.List[BondFeature],
    ) -> "DGLMolecule":
        """Creates a new molecular graph representation from an OpenFF molecule object.

        Args:
            molecule: The molecule to store in the graph.
            atom_features: The atom features to compute for the molecule.
            bond_features: The bond features to compute for the molecule.

        Returns:
            The constructed graph.
        """

        return cls(cls._molecule_to_dgl(molecule, atom_features, bond_features), 1)

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

        from openff.toolkit.topology import Molecule

        return cls.from_openff(
            Molecule.from_smiles(smiles),
            atom_features,
            bond_features,
        )


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
