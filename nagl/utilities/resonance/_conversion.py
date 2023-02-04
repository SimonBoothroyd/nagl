import collections
import typing

import dgl
import networkx
import torch
from rdkit import Chem

from nagl.utilities.molecule import BOND_ORDER_TO_TYPE, BOND_TYPE_TO_ORDER

if typing.TYPE_CHECKING:
    from nagl.molecules import DGLMolecule


def rdkit_molecule_to_networkx(molecule: Chem.Mol) -> networkx.Graph:
    """Attempts to create a networkx graph representation from an RDKit molecule
    object.

    Args:
        molecule: The RDKit molecule object.

    Returns:
        The graph representation.
    """

    molecule = Chem.Mol(molecule)
    Chem.Kekulize(molecule)

    nx_graph = networkx.Graph()
    nx_graph.add_nodes_from(
        [
            (
                atom.GetIdx(),
                {
                    "element": atom.GetSymbol(),
                    "formal_charge": atom.GetFormalCharge(),
                    "bond_orders": tuple(
                        sorted(
                            BOND_TYPE_TO_ORDER[bond.GetBondType()]
                            for bond in atom.GetBonds()
                        )
                    ),
                },
            )
            for atom in molecule.GetAtoms()
        ]
    )

    for bond in molecule.GetBonds():
        nx_graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_order=BOND_TYPE_TO_ORDER[bond.GetBondType()],
        )

    return nx_graph


def rdkit_molecule_from_networkx(nx_graph: networkx.Graph) -> Chem.Mol:
    """Attempts to create an RDKit molecule from a networkx graph representation.

    Notes:
        This method will strip all stereochemistry and aromaticity information.

    Args:
        nx_graph: The graph representation.

    Returns:
        The RDKit molecule object.
    """

    molecule = Chem.RWMol()

    for node_index in nx_graph.nodes:
        node = nx_graph.nodes[node_index]

        atom = Chem.Atom(node["element"])
        atom.SetFormalCharge(node["formal_charge"])

        molecule.AddAtom(atom)

    for atom_index_a, atom_index_b in nx_graph.edges:
        molecule.AddBond(
            atom_index_a,
            atom_index_b,
            BOND_ORDER_TO_TYPE[nx_graph[atom_index_a][atom_index_b]["bond_order"]],
        )

    molecule = Chem.Mol(molecule)
    Chem.SanitizeMol(molecule)
    Chem.SetAromaticity(molecule, Chem.AROMATICITY_RDKIT)

    return molecule


def dgl_molecule_to_networkx(molecule: "DGLMolecule") -> networkx.Graph:
    """Attempts to create an networkx graph representation from an DGL molecule
    object.

    Args:
        molecule: The DGL molecule object.

    Returns:
        The graph representation.
    """

    dgl_graph = molecule.graph

    elements = [
        Chem.Atom(int(atomic_number)).GetSymbol()
        for atomic_number in dgl_graph.ndata["atomic_number"]
    ]
    formal_charges = dgl_graph.ndata["formal_charge"]

    per_atom_bond_orders = collections.defaultdict(list)

    indices_a, indices_b = dgl_graph.all_edges()

    for index_a, index_b, bond_order in zip(
        indices_a[dgl_graph.edata["mask"]],
        indices_b[dgl_graph.edata["mask"]],
        dgl_graph.edata["bond_order"][dgl_graph.edata["mask"]],
    ):
        per_atom_bond_orders[int(index_a)].append(bond_order)
        per_atom_bond_orders[int(index_b)].append(bond_order)

    per_atom_bond_orders = {**per_atom_bond_orders}

    nx_graph = networkx.Graph()
    nx_graph.add_nodes_from(
        [
            (
                atom_index,
                {
                    "element": element,
                    "formal_charge": int(formal_charge),
                    "bond_orders": tuple(
                        sorted(int(i) for i in per_atom_bond_orders[atom_index])
                    ),
                },
            )
            for atom_index, (element, formal_charge) in enumerate(
                zip(elements, formal_charges)
            )
        ]
    )

    for index_a, index_b, bond_order in zip(
        indices_a[dgl_graph.edata["mask"]],
        indices_b[dgl_graph.edata["mask"]],
        dgl_graph.edata["bond_order"][dgl_graph.edata["mask"]],
    ):
        nx_graph.add_edge(int(index_a), int(index_b), bond_order=int(bond_order))

    return nx_graph


def dgl_molecule_from_networkx(nx_graph: networkx.Graph) -> "DGLMolecule":
    """Attempts to create a DGL molecule from a networkx graph representation.

    Notes:
        This method will strip all feature information.

    Args:
        nx_graph: The graph representation.

    Returns:
        The DGL heterograph object.
    """

    from nagl.molecules import DGLMolecule

    indices_a, indices_b = zip(*nx_graph.edges)

    indices_a = torch.tensor(indices_a, dtype=torch.int32)
    indices_b = torch.tensor(indices_b, dtype=torch.int32)

    undirected_indices_a = torch.cat([indices_a, indices_b])
    undirected_indices_b = torch.cat([indices_b, indices_a])

    dgl_graph: dgl.DGLGraph = dgl.graph((undirected_indices_a, undirected_indices_b))
    dgl_graph.ndata["formal_charge"] = torch.tensor(
        [nx_graph.nodes[node_index]["formal_charge"] for node_index in nx_graph.nodes],
        dtype=torch.int8,
    )
    dgl_graph.ndata["atomic_number"] = torch.tensor(
        [
            Chem.Atom(nx_graph.nodes[node_index]["element"]).GetAtomicNum()
            for node_index in nx_graph.nodes
        ],
        dtype=torch.uint8,
    )

    bond_mask = torch.tensor(
        [True] * len(indices_a) + [False] * len(indices_a), dtype=torch.bool
    )
    dgl_graph.edata["mask"] = bond_mask

    bond_orders = torch.tensor(
        [
            nx_graph[index_a][index_b]["bond_order"]
            for index_a, index_b in nx_graph.edges
        ],
        dtype=torch.uint8,
    )
    dgl_graph.edata["bond_order"] = torch.cat([bond_orders, bond_orders])

    return DGLMolecule(dgl_graph, 1)
