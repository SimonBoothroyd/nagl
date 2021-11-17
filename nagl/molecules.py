from typing import TYPE_CHECKING, List, Tuple, Type

import dgl.function
import torch

from nagl.features import AtomFeature, AtomFeaturizer, BondFeature, BondFeaturizer
from nagl.utilities.resonance import enumerate_resonance_forms

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


def _hetero_to_homo_graph(graph: dgl.DGLHeteroGraph) -> dgl.DGLGraph:

    try:
        homo_graph = dgl.to_homogeneous(graph, ndata=["feat"], edata=["feat"])
    except KeyError:

        # A nasty workaround to check when we don't have any atom / bond features as
        # DGL doesn't allow easy querying of features dicts for hetereographs with
        # multiple edge / node types.
        try:
            homo_graph = dgl.to_homogeneous(graph, ndata=["feat"], edata=[])
        except KeyError:
            try:
                homo_graph = dgl.to_homogeneous(graph, ndata=[], edata=["feat"])
            except KeyError:
                homo_graph = dgl.to_homogeneous(graph, ndata=[], edata=[])

    return homo_graph


class _BaseDGLModel:
    @property
    def graph(self) -> dgl.DGLHeteroGraph:
        """Returns the DGL graph representation of the molecule."""
        return self._graph

    @property
    def homograph(self) -> dgl.DGLGraph:
        """Returns the homogeneous (i.e. only one node and edge type) graph
        representation of the molecule."""
        return _hetero_to_homo_graph(self._graph)

    @property
    def atom_features(self) -> torch.Tensor:
        """Returns a tensor containing the initial atom features with
        shape=(n_atoms, n_atom_features)."""
        return self._graph.ndata["feat"].float()

    def __init__(self, graph: dgl.DGLHeteroGraph):
        """

        Args:
            graph: The DGL graph representation of the molecule.
        """

        self._graph = graph


class DGLMolecule(_BaseDGLModel):
    """A wrapper around a DGL graph representation of a molecule that stores additional
    metadata such as the number of different representations (e.g. resonance structures)
    and the number of atoms in the molecule."""

    @property
    def n_atoms(self) -> int:
        return int(self._graph.number_of_nodes() / self._n_representations)

    @property
    def n_bonds(self) -> int:
        return int(self._graph.number_of_edges("forward") / self._n_representations)

    @property
    def n_representations(self) -> int:
        return self._n_representations

    def __init__(self, graph: dgl.DGLHeteroGraph, n_representations: int):
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
        molecule_graph: dgl.DGLHeteroGraph = dgl.heterograph(
            {
                ("atom", "forward", "atom"): (indices_a, indices_b),
                ("atom", "reverse", "atom"): (indices_b, indices_a),
            }
        )

        # Assign the atom (node) features.
        if len(atom_features) > 0:
            molecule_graph.ndata["feat"] = AtomFeaturizer.featurize(
                molecule, atom_features
            )

        molecule_graph.ndata["idx"] = torch.tensor(
            [i for i in range(molecule.n_atoms)], dtype=torch.int32
        )
        molecule_graph.ndata["formal_charge"] = torch.tensor(
            [
                atom.formal_charge.value_in_unit(unit.elementary_charge)
                for atom in molecule.atoms
            ],
            dtype=torch.int8,
        )
        molecule_graph.ndata["atomic_number"] = torch.tensor(
            [atom.atomic_number for atom in molecule.atoms], dtype=torch.uint8
        )

        # Assign the bond (edge) features.
        feature_tensor = (
            BondFeaturizer.featurize(molecule, bond_features)
            if len(bond_features) > 0
            else None
        )
        bond_orders = torch.tensor(
            [bond.bond_order for bond in molecule.bonds], dtype=torch.uint8
        )

        for direction in ("forward", "reverse"):

            if feature_tensor is not None:
                molecule_graph.edges[direction].data["feat"] = feature_tensor

            molecule_graph.edges[direction].data["bond_order"] = bond_orders

        return molecule_graph

    @classmethod
    def from_openff(
        cls: Type["DGLMolecule"],
        molecule: "Molecule",
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        enumerate_resonance: bool = False,
    ) -> "DGLMolecule":
        """Creates a new molecular graph representation from an OpenFF molecule object.

        Args:
            molecule: The molecule to store in the graph.
            atom_features: The atom features to compute for the molecule.
            bond_features: The bond features to compute for the molecule.
            enumerate_resonance: Whether to enumerate the lowest energy resonance
                structures of each molecule and store each within the graph
                representation.

        Returns:
            The constructed graph.
        """

        resonance_forms = (
            [molecule]
            if not enumerate_resonance
            else enumerate_resonance_forms(molecule, lowest_energy_only=True)
        )

        graphs = [
            cls._molecule_to_dgl(resonance_form, atom_features, bond_features)
            for resonance_form in resonance_forms
        ]

        graph = dgl.batch(graphs)

        graph.set_batch_num_nodes(graph.batch_num_nodes().sum().reshape((-1,)))
        graph.set_batch_num_edges(
            {
                e_type: graph.batch_num_edges(e_type).sum().reshape((-1,))
                for e_type in graph.canonical_etypes
            }
        )

        return cls(graph, len(graphs))

    @classmethod
    def from_smiles(
        cls: Type["DGLMolecule"],
        smiles: str,
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        enumerate_resonance: bool = False,
    ) -> "DGLMolecule":
        """Creates a new molecular graph representation from a SMILES pattern.

        Args:
            smiles: The SMILES representation of the molecule to store in the graph.
            atom_features: The atom features to compute for the molecule.
            bond_features: The bond features to compute for the molecule.
            enumerate_resonance: Whether to enumerate the lowest energy resonance
                structures of each molecule and store each within the graph
                representation.

        Returns:
            The constructed graph.
        """

        from openff.toolkit.topology import Molecule

        return cls.from_openff(
            Molecule.from_smiles(smiles),
            atom_features,
            bond_features,
            enumerate_resonance,
        )


class DGLMoleculeBatch(_BaseDGLModel):
    """A wrapper around a batch of DGL molecule objects."""

    @property
    def n_atoms_per_molecule(self) -> Tuple[int, ...]:
        """Returns the number of atoms in each unique molecule in the batch."""
        return self._n_atoms

    @property
    def n_representations_per_molecule(self) -> Tuple[int, ...]:
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
