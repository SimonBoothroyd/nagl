import copy
import itertools
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    NamedTuple,
    Set,
    Tuple,
    overload,
)

import dgl
import networkx
import numpy
import torch
from openff.toolkit.topology import Molecule
from openmm.app import Element

if TYPE_CHECKING:
    from nagl.molecules import DGLMolecule


class _ResonanceTypeKey(NamedTuple):
    """A convenient data structure for storing information used to recognize a possible
    resonance atom type by."""

    element: Literal["O", "S", "N"]

    formal_charge: int
    bond_orders: Tuple[int, ...]


class _ResonanceTypeValue(NamedTuple):
    """A convenient data structure for storing information about a possible resonance
    atom type in."""

    type: Literal["A", "D"]

    energy: float

    id: int
    conjugate_id: int


_RESONANCE_TYPES: Dict[_ResonanceTypeKey, _ResonanceTypeValue] = {
    _ResonanceTypeKey("O", 0, (2,)): _ResonanceTypeValue("A", 0.0, 1, 2),
    _ResonanceTypeKey("O", -1, (1,)): _ResonanceTypeValue("D", 5.0, 2, 1),
    #
    _ResonanceTypeKey("S", 0, (2,)): _ResonanceTypeValue("A", 0.0, 3, 4),
    _ResonanceTypeKey("S", -1, (1,)): _ResonanceTypeValue("D", 5.0, 4, 3),
    #
    _ResonanceTypeKey("N", +1, (1, 1, 2)): _ResonanceTypeValue("A", 5.0, 5, 6),
    _ResonanceTypeKey("N", 0, (1, 1, 1)): _ResonanceTypeValue("D", 0.0, 6, 5),
    #
    _ResonanceTypeKey("N", 0, (1, 2)): _ResonanceTypeValue("A", 0.0, 7, 8),
    _ResonanceTypeKey("N", -1, (1, 1)): _ResonanceTypeValue("D", 5.0, 8, 7),
    #
    _ResonanceTypeKey("N", 0, (3,)): _ResonanceTypeValue("A", 0.0, 9, 10),
    _ResonanceTypeKey("N", -1, (2,)): _ResonanceTypeValue("D", 5.0, 10, 9),
}

_RESONANCE_KEYS_BY_ID = {value.id: key for key, value in _RESONANCE_TYPES.items()}


def _openff_molecule_to_networkx(molecule: Molecule) -> networkx.Graph:
    """Attempts to create a networkx graph representation from an OpenFF molecule
    object.

    Args:
        molecule: The OpenFF molecule object.

    Returns:
        The graph representation.
    """

    from simtk import unit as simtk_unit

    nx_graph = networkx.Graph()
    nx_graph.add_nodes_from(
        [
            (
                atom_index,
                {
                    "element": atom.element.symbol,
                    "formal_charge": atom.formal_charge.value_in_unit(
                        simtk_unit.elementary_charge
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


def _openff_molecule_from_networkx(nx_graph: networkx.Graph) -> Molecule:
    """Attempts to create an OpenFF molecule from a networkx graph representation.

    Notes:
        This method will strip all stereochemistry and aromaticity information.

    Args:
        nx_graph: The graph representation.

    Returns:
        The OpenFF molecule object.
    """

    molecule = Molecule()

    for node_index in nx_graph.nodes:
        node = nx_graph.nodes[node_index]

        molecule.add_atom(
            Element.getBySymbol(node["element"]).atomic_number,
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


def _dgl_molecule_to_networkx(molecule: "DGLMolecule") -> networkx.Graph:
    """Attempts to create an networkx graph representation from an DGL molecule
    object.

    Args:
        molecule: The DGL molecule object.

    Returns:
        The graph representation.
    """

    dgl_graph = molecule.graph

    elements = [
        Element.getByAtomicNumber(int(atomic_number)).symbol
        for atomic_number in dgl_graph.ndata["atomic_number"]
    ]
    formal_charges = dgl_graph.ndata["formal_charge"]

    per_atom_bond_orders = defaultdict(list)

    for index_a, index_b, bond_order in zip(
        *dgl_graph.all_edges(etype="forward"),
        dgl_graph.edges["forward"].data["bond_order"]
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
        dgl_graph.edges["forward"].data["bond_order"]
    ):

        nx_graph.add_edge(int(index_a), int(index_b), bond_order=int(bond_order))

    return nx_graph


def _dgl_molecule_from_networkx(nx_graph: networkx.Graph) -> "DGLMolecule":
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
            Element.getBySymbol(nx_graph.nodes[node_index]["element"]).atomic_number
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


@overload
def enumerate_resonance_forms(
    molecule: Molecule, lowest_energy_only: bool = True
) -> List[Molecule]:
    ...


@overload
def enumerate_resonance_forms(
    molecule: "DGLMolecule", lowest_energy_only: bool = True
) -> List["DGLMolecule"]:
    ...


def enumerate_resonance_forms(molecule, lowest_energy_only: bool = True):
    """Recursively attempts to find all resonance structures of an input molecule
    according to the algorithm proposed by Gilson et al [1].

    Notes:
        This method will strip all stereochemistry and aromaticity information.

    Args:
        molecule: The input molecule.
        lowest_energy_only: Whether to only return the resonance forms with the lowest
            'energy' as defined in [1].

    References:
        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
        assignment of accurate partial atomic charges: an electronegativity
        equalization method that accounts for alternate resonance forms." Journal of
        chemical information and computer sciences 43.6 (2003): 1982-1997.

    Returns:
        A list of all resonance forms including the original molecule.
    """

    from nagl.molecules import DGLMolecule

    if isinstance(molecule, Molecule):
        original_nx_graph = _openff_molecule_to_networkx(molecule)
    elif isinstance(molecule, DGLMolecule):
        original_nx_graph = _dgl_molecule_to_networkx(molecule)
    else:
        raise NotImplementedError

    closed_list: Dict[Any, networkx.Graph] = {}
    _enumerate_resonance_forms(original_nx_graph, closed_list)

    donor_acceptor_indices = {*_extract_type_features(original_nx_graph)}

    found_resonance_forms = [*closed_list.values()]

    if lowest_energy_only:

        found_resonance_form_energies = []

        for resonance_graph in found_resonance_forms:

            resonance_keys = [
                _ResonanceTypeKey(**resonance_graph.nodes[atom_index])
                for atom_index in donor_acceptor_indices
            ]

            total_energy = sum(_RESONANCE_TYPES[key].energy for key in resonance_keys)

            found_resonance_form_energies.append(total_energy)

        lowest_energy = min(found_resonance_form_energies)

        found_resonance_forms = [
            resonance_form
            for resonance_form, resonance_energy in zip(
                found_resonance_forms, found_resonance_form_energies
            )
            if numpy.isclose(resonance_energy, lowest_energy)
        ]

    if isinstance(molecule, Molecule):
        resonance_forms = [
            _openff_molecule_from_networkx(resonance_graph)
            for resonance_graph in found_resonance_forms
        ]
    elif isinstance(molecule, DGLMolecule):
        resonance_forms = [
            _dgl_molecule_from_networkx(resonance_graph)
            for resonance_graph in found_resonance_forms
        ]
    else:
        raise NotImplementedError

    return resonance_forms


def _enumerate_resonance_forms(
    nx_graph: networkx.Graph,
    closed_list: Dict[Tuple[Tuple[int, Literal["A", "D"]], ...], networkx.Graph],
):
    """Recursively attempts to find all resonance structures of an input molecule stored
    in a graph representation according to the v-charge algorithm proposed by Gilson et
    al [1].

    Args:
        nx_graph: The input molecular graph.
        closed_list: A dictionary to store the graph representation of all found
            resonance forms in.

    References:
        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
        assignment of accurate partial atomic charges: an electronegativity
        equalization method that accounts for alternate resonance forms." Journal of
        chemical information and computer sciences 43.6 (2003): 1982-1997.

    """

    type_features = _extract_type_features(nx_graph)

    # We can return early if there are no atoms that are known acceptors / donors.
    if len(type_features) == 0:
        return

    donor_acceptor_key = tuple(
        (atom_index, type_features[atom_index]) for atom_index in sorted(type_features)
    )

    if donor_acceptor_key in closed_list:
        return

    closed_list[donor_acceptor_key] = nx_graph

    acceptor_indices = {
        atom_index
        for atom_index, atom_type in type_features.items()
        if atom_type == "A"
    }
    donor_indices = {
        atom_index
        for atom_index, atom_type in type_features.items()
        if atom_type == "D"
    }

    # Try and find all possible electron transfer paths.
    for acceptor_index, donor_index in itertools.product(
        acceptor_indices, donor_indices
    ):

        transfer_paths = _find_transfer_paths(nx_graph, acceptor_index, donor_index)

        if len(transfer_paths) == 0:
            continue

        # These must be a conjugate / donor pair
        for transfer_path in transfer_paths:

            flipped_graph = _perform_electron_transfer(nx_graph, transfer_path)
            _enumerate_resonance_forms(flipped_graph, closed_list)


def _extract_type_features(nx_graph: networkx.Graph) -> Dict[int, Literal["A", "D"]]:
    """Attempts to find any potential acceptor / donor atoms in a molecular graph
    and returns the resonance type key (see the ``_RESONANCE_TYPES`` dictionary)
    associated with each.

    Args:
        nx_graph: The molecular graph.

    Returns:
        A dictionary of the found accept / donor atoms and their respective resonance
        type key.
    """

    nodes_by_feature: Dict[_ResonanceTypeKey, Set[int]] = defaultdict(set)

    for atom_index in nx_graph:

        node_attributes = nx_graph.nodes[atom_index]

        node_features = _ResonanceTypeKey(
            node_attributes["element"],
            node_attributes["formal_charge"],
            node_attributes["bond_orders"],
        )

        if node_features in _RESONANCE_TYPES:
            nodes_by_feature[node_features].add(atom_index)

    nodes_by_feature = {**nodes_by_feature}  # Convert defaultdict -> dict

    return {
        atom_index: _RESONANCE_TYPES[node_features].type
        for node_features, atom_indices in nodes_by_feature.items()
        for atom_index in atom_indices
    }


def _find_transfer_paths(
    nx_graph: networkx.Graph, acceptor_index: int, donor_index: int
) -> List[List[int]]:
    """Attempts to find all possible electron transfer paths, as defined by Gilson et
    al [1], between a donor and an acceptor atom.

    References:

        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
            assignment of accurate partial atomic charges: an electronegativity
            equalization method that accounts for alternate resonance forms." Journal of
            chemical information and computer sciences 43.6 (2003): 1982-1997.

    Args:
        nx_graph: The graph representation of the molecule.
        acceptor_index: The index of the acceptor atom.
        donor_index: The index of the donor atom.

    Returns:
        A list of any 'electron transfer' paths that begin from the donor atom and end
        at the acceptor atom.
    """

    all_paths = networkx.all_simple_paths(nx_graph, donor_index, acceptor_index)

    # Filter out any paths with even lengths as these cannot transfer paths
    all_paths = [path for path in all_paths if len(path) > 0 and len(path) % 2 != 0]

    # Check whether the bonds along the path form a rising-falling sequence.
    transfer_paths = []

    for path in all_paths:

        pairwise_path = networkx.utils.pairwise(path)

        previous_bond_order = None

        for bond_index, (atom_index_a, atom_index_b) in enumerate(pairwise_path):

            bond_order = nx_graph[atom_index_a][atom_index_b]["bond_order"]

            bond_order_delta = (
                None
                if previous_bond_order is None
                else (bond_order - previous_bond_order)
            )

            previous_bond_order = bond_order

            if bond_order_delta is None:
                continue

            if (bond_index % 2 == 1 and bond_order_delta != 1) or (
                bond_index % 2 == 0 and bond_order_delta != -1
            ):
                # The conjugation is broken so we can ignore discard possible path.
                break

        else:
            transfer_paths.append(path)

    return transfer_paths


def _perform_electron_transfer(
    nx_graph: networkx.Graph, transfer_path: List[int]
) -> networkx.Graph:
    """Carries out an electron transfer along the pre-determined transfer path starting
    from a donor and ending in an acceptor.

    Args:
        nx_graph: The original graph representation of the molecule.
        transfer_path: The indices of the atoms in a pre-determined transfer path
            starting from a donor atom and ending in an acceptor.

    Returns:
        A new graph representation of the molecule after the transfer has taken place.
    """

    donor_index, acceptor_index = transfer_path[0], transfer_path[-1]

    flipped_graph = copy.deepcopy(nx_graph)

    for node_index in [donor_index, acceptor_index]:

        node = flipped_graph.nodes[node_index]

        conjugate_key = _RESONANCE_KEYS_BY_ID[
            _RESONANCE_TYPES[_ResonanceTypeKey(**node)].conjugate_id
        ]

        node["element"] = conjugate_key.element
        node["formal_charge"] = conjugate_key.formal_charge
        node["bond_orders"] = conjugate_key.bond_orders

    pairwise_path = networkx.utils.pairwise(transfer_path)

    for bond_index, (atom_index_a, atom_index_b) in enumerate(pairwise_path):

        increment = 1 if (bond_index % 2 == 0) else -1
        flipped_graph[atom_index_a][atom_index_b]["bond_order"] += increment

    return flipped_graph
