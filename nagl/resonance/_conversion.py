from collections import defaultdict
from typing import TYPE_CHECKING

import dgl
import networkx
import torch
from openff.toolkit.topology import Molecule

if TYPE_CHECKING:
    from nagl.molecules import DGLMolecule


def openff_molecule_to_networkx(molecule: Molecule) -> networkx.Graph:
    """Attempts to create a networkx graph representation from an OpenFF molecule
    object.

    Args:
        molecule: The OpenFF molecule object.

    Returns:
        The graph representation.
    """

    from openff.units import unit as openff_unit

    nx_graph = networkx.Graph()
    nx_graph.add_nodes_from(
        [
            (
                atom_index,
                {
                    "element": atom.symbol,
                    "formal_charge": atom.formal_charge.m_as(
                        openff_unit.elementary_charge
                    ),
                    "bond_orders": tuple(
                        sorted(bond.bond_order for bond in atom.bonds)
                    ),
                },
            )
            for atom_index, atom in enumerate(molecule.atoms)
        ]
    )

    for bond in molecule.bonds:

        nx_graph.add_edge(
            bond.atom1_index, bond.atom2_index, bond_order=bond.bond_order
        )

    return nx_graph


def openff_molecule_from_networkx(nx_graph: networkx.Graph) -> Molecule:
    """Attempts to create an OpenFF molecule from a networkx graph representation.

    Notes:
        This method will strip all stereochemistry and aromaticity information.

    Args:
        nx_graph: The graph representation.

    Returns:
        The OpenFF molecule object.
    """

    from openff.units.elements import SYMBOLS

    molecule = Molecule()

    symbol_to_num = {v: k for k, v in SYMBOLS.items()}

    for node_index in nx_graph.nodes:
        node = nx_graph.nodes[node_index]

        molecule.add_atom(
            symbol_to_num[node["element"]],
            node["formal_charge"],
            False,
        )

    for atom_index_a, atom_index_b in nx_graph.edges:

        molecule.add_bond(
            atom_index_a,
            atom_index_b,
            nx_graph[atom_index_a][atom_index_b]["bond_order"],
            False,
        )

    return molecule


def dgl_molecule_to_networkx(molecule: "DGLMolecule") -> networkx.Graph:
    """Attempts to create an networkx graph representation from an DGL molecule
    object.

    Args:
        molecule: The DGL molecule object.

    Returns:
        The graph representation.
    """

    from openff.units.elements import SYMBOLS

    dgl_graph = molecule.graph

    elements = [
        SYMBOLS[int(atomic_number)]
        for atomic_number in dgl_graph.ndata["atomic_number"]
    ]
    formal_charges = dgl_graph.ndata["formal_charge"]

    per_atom_bond_orders = defaultdict(list)

    for index_a, index_b, bond_order in zip(
        *dgl_graph.all_edges(etype="forward"),
        dgl_graph.edges["forward"].data["bond_order"],
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
        *dgl_graph.all_edges(etype="forward"),
        dgl_graph.edges["forward"].data["bond_order"],
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

    from openff.units.elements import SYMBOLS

    from nagl.molecules import DGLMolecule

    symbol_to_num = {v: k for k, v in SYMBOLS.items()}

    indices_a, indices_b = zip(*nx_graph.edges)

    indices_a = torch.tensor(indices_a, dtype=torch.int32)
    indices_b = torch.tensor(indices_b, dtype=torch.int32)

    dgl_graph: dgl.DGLHeteroGraph = dgl.heterograph(
        {
            ("atom", "forward", "atom"): (indices_a, indices_b),
            ("atom", "reverse", "atom"): (indices_b, indices_a),
        }
    )
    dgl_graph.ndata["formal_charge"] = torch.tensor(
        [nx_graph.nodes[node_index]["formal_charge"] for node_index in nx_graph.nodes],
        dtype=torch.int8,
    )
    dgl_graph.ndata["atomic_number"] = torch.tensor(
        [
            symbol_to_num[nx_graph.nodes[node_index]["element"]]
            for node_index in nx_graph.nodes
        ],
        dtype=torch.uint8,
    )

    bond_orders = torch.tensor(
        [
            nx_graph[index_a][index_b]["bond_order"]
            for index_a, index_b in nx_graph.edges
        ],
        dtype=torch.uint8,
    )

    for direction in ("forward", "reverse"):
        dgl_graph.edges[direction].data["bond_order"] = bond_orders

    return DGLMolecule(dgl_graph, 1)
