import hashlib
import itertools
import json
import pickle
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import networkx
import numpy
from openff.toolkit.topology import Molecule

from nagl.resonance._caching import PathCache
from nagl.resonance._conversion import (
    dgl_molecule_from_networkx,
    dgl_molecule_to_networkx,
    openff_molecule_from_networkx,
    openff_molecule_to_networkx,
)

if TYPE_CHECKING:
    from nagl.molecules import DGLMolecule

_T = TypeVar("_T")


class ResonanceTypeKey(NamedTuple):
    """A convenient data structure for storing information used to recognize a possible
    resonance atom type by."""

    element: Literal["O", "S", "N"]

    formal_charge: int
    bond_orders: Tuple[int, ...]


class ResonanceTypeValue(NamedTuple):
    """A convenient data structure for storing information about a possible resonance
    atom type in."""

    type: Literal["A", "D"]

    energy: float

    id: int
    conjugate_id: int


_RESONANCE_TYPES: Dict[ResonanceTypeKey, ResonanceTypeValue] = {
    ResonanceTypeKey("O", 0, (2,)): ResonanceTypeValue("A", 0.0, 1, 2),
    ResonanceTypeKey("O", -1, (1,)): ResonanceTypeValue("D", 5.0, 2, 1),
    #
    ResonanceTypeKey("S", 0, (2,)): ResonanceTypeValue("A", 0.0, 3, 4),
    ResonanceTypeKey("S", -1, (1,)): ResonanceTypeValue("D", 5.0, 4, 3),
    #
    ResonanceTypeKey("N", +1, (1, 1, 2)): ResonanceTypeValue("A", 5.0, 5, 6),
    ResonanceTypeKey("N", 0, (1, 1, 1)): ResonanceTypeValue("D", 0.0, 6, 5),
    #
    ResonanceTypeKey("N", 0, (1, 2)): ResonanceTypeValue("A", 0.0, 7, 8),
    ResonanceTypeKey("N", -1, (1, 1)): ResonanceTypeValue("D", 5.0, 8, 7),
    #
    ResonanceTypeKey("N", 0, (3,)): ResonanceTypeValue("A", 0.0, 9, 10),
    ResonanceTypeKey("N", -1, (2,)): ResonanceTypeValue("D", 5.0, 10, 9),
}

_RESONANCE_KEYS_BY_ID = {value.id: key for key, value in _RESONANCE_TYPES.items()}


@overload
def enumerate_resonance_forms(
    molecule: Molecule,
    lowest_energy_only: bool = True,
    max_path_length: Optional[int] = None,
    as_dicts: Literal[False] = False,
) -> List[Molecule]:
    ...


@overload
def enumerate_resonance_forms(
    molecule: "DGLMolecule",
    lowest_energy_only: bool = True,
    max_path_length: Optional[int] = None,
    as_dicts: Literal[False] = False,
) -> List["DGLMolecule"]:
    ...


@overload
def enumerate_resonance_forms(
    molecule: Union[Molecule, "DGLMolecule"],
    lowest_energy_only: bool = True,
    max_path_length: Optional[int] = None,
    as_dicts: Literal[True] = True,
) -> List[dict]:
    ...


def enumerate_resonance_forms(
    molecule,
    lowest_energy_only: bool = True,
    max_path_length: Optional[int] = None,
    as_dicts: bool = False,
):
    """Recursively attempts to find all resonance structures of an input molecule
    according to a modified version of the algorithm proposed by Gilson et al [1].

    Enumeration proceeds by:

    1) The molecule is turned into a ``networkx`` graph object.
    2) All hydrogen's and uncharged sp3 carbons are removed from the graph as these
       will not be involved in electron transfer.
    3) Disjoint sub-graphs are detected and separated out.
    4) Sub-graphs that don't contain at least 1 donor and 1 acceptor are discarded
    5. For each disjoint subgraph:
        a) The original v-charge algorithm is applied to yield the resonance structures
           of that subgraph.

    This will lead to ``M_i`` resonance structures for each of the ``N`` sub-graphs.

    If ``as_dicts=True`` then the resonance states in each sub-graph are returned. This
    avoids the need to combinatorially combining resonance information from each
    sub-graph. When ``as_dicts=False``, all ``M_0 x M_1 x ... x M_N`` forms are fully
    enumerated and return as molecule objects matching the input molecule type.

    Notes:
        This method will strip all stereochemistry and aromaticity information from the
        input molecule.

    Args:
        molecule: The input molecule.
        lowest_energy_only: Whether to only return the resonance forms with the lowest
            'energy' as defined in [1].
        max_path_length: The maximum number of bonds between a donor and acceptor to
            consider.
        as_dicts: Whether to return the resonance forms in a form that is more
            compatible with producing feature vectors. If false, all combinatorial
            resonance forms will be returned which may be significantly slow if the
            molecule is very heavily conjugated and has many donor / acceptor pairs.

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
        original_nx_graph = openff_molecule_to_networkx(molecule)
    elif isinstance(molecule, DGLMolecule):
        original_nx_graph = dgl_molecule_to_networkx(molecule)
    else:
        raise NotImplementedError

    # Attempt to split the input graph into smaller disjoint ones by removing nodes that
    # electron transfer could not possibly occur along. These will be substantially
    # faster to path find on and reduce the number of A/D combinations that we need to
    # consider.
    sub_graphs = _find_sub_graphs(original_nx_graph)

    # Find all the resonance forms for each of the sub-graphs.
    resonance_sub_graphs = [
        _enumerate_resonance_graphs(sub_graph, lowest_energy_only, max_path_length)
        for sub_graph in sub_graphs
    ]

    if not as_dicts:

        return _graphs_to_molecules(
            type(molecule), original_nx_graph, resonance_sub_graphs
        )

    return _graphs_to_dicts(resonance_sub_graphs)


def _graph_to_hash(nx_graph: networkx.Graph) -> bytes:
    """Attempts to hash a ``networkx`` graph by JSON serializing a dictionary containing
    the resonance atom types and all bond orders, and encoding the resulting string
    in a SHA1 hash.
    """

    atom_resonance_types = _find_donor_acceptors(nx_graph)

    return hashlib.sha1(
        json.dumps(
            {
                "a": atom_resonance_types,
                "b": {
                    i: nx_graph[index_a][index_b]["bond_order"]
                    for i, (index_a, index_b) in enumerate(nx_graph.edges)
                },
            },
            sort_keys=True,
        ).encode(),
        usedforsecurity=False,
    ).digest()


def _graphs_to_molecules(
    expected_type: Type[_T],
    original_nx_graph: networkx.Graph,
    resonance_sub_graphs: List[Dict[bytes, networkx.Graph]],
) -> List[_T]:

    found_resonance_forms = []

    for sub_graph_combinations in itertools.product(*resonance_sub_graphs):

        resonance_form = pickle.loads(pickle.dumps(original_nx_graph))

        for sub_graph_index, sub_graph_key in enumerate(sub_graph_combinations):

            sub_graph = resonance_sub_graphs[sub_graph_index][sub_graph_key]

            for atom_index in sub_graph.nodes:
                for attr in sub_graph.nodes[atom_index]:
                    resonance_form.nodes[atom_index][attr] = sub_graph.nodes[
                        atom_index
                    ][attr]

            for index_a, index_b in sub_graph.edges:
                for attr in sub_graph[index_a][index_b]:
                    resonance_form[index_a][index_b][attr] = sub_graph[index_a][
                        index_b
                    ][attr]

        found_resonance_forms.append(resonance_form)

    if issubclass(expected_type, Molecule):
        resonance_forms = [
            openff_molecule_from_networkx(resonance_graph)
            for resonance_graph in found_resonance_forms
        ]
    elif issubclass(expected_type, DGLMolecule):
        resonance_forms = [
            dgl_molecule_from_networkx(resonance_graph)
            for resonance_graph in found_resonance_forms
        ]
    else:
        raise NotImplementedError

    return resonance_forms


def _graphs_to_dicts(
    resonance_sub_graphs: List[Dict[bytes, networkx.Graph]],
) -> List[dict]:

    found_resonance_forms = []

    for resonance_graph_dicts in resonance_sub_graphs:

        resonance_graphs: List[networkx.Graph] = [*resonance_graph_dicts.values()]

        bond_orders = {
            (index_a, index_b): [
                resonance_graph[index_a][index_b]["bond_order"]
                for resonance_graph in resonance_graphs
            ]
            for index_a, index_b in resonance_graphs[0].edges
        }
        resonant_bonds = {
            bond_indices: values
            for bond_indices, values in bond_orders.items()
            if len({*values}) > 1
        }

        if len(resonant_bonds) == 0:
            continue

        donor_acceptor_indices = sorted(
            set.intersection(
                {*_find_donor_acceptors(resonance_graphs[0])},
                {i for bond_indices in resonant_bonds for i in bond_indices},
            )
        )
        resonant_atoms = {
            index: [
                {
                    "formal_charge": resonance_graph.nodes[index]["formal_charge"],
                    "type": _RESONANCE_TYPES[
                        ResonanceTypeKey(
                            resonance_graph.nodes[index]["element"],
                            resonance_graph.nodes[index]["formal_charge"],
                            resonance_graph.nodes[index]["bond_orders"],
                        )
                    ].type,
                }
                for resonance_graph in resonance_graphs
            ]
            for index in donor_acceptor_indices
        }

        found_resonance_forms.append({"atoms": resonant_atoms, "bonds": resonant_bonds})

    return found_resonance_forms


def _enumerate_resonance_graphs(
    nx_graph: networkx.Graph,
    lowest_energy_only: bool = True,
    max_path_length: Optional[int] = None,
) -> Dict[bytes, networkx.Graph]:
    """Attempts to find all resonance structures of an input molecule stored
    in a graph representation according to the v-charge algorithm proposed by Gilson et
    al [1].

    Args:
        nx_graph: The input molecular graph.
        lowest_energy_only: Whether to only return the resonance forms with the lowest
            'energy' as defined in [1].
        max_path_length: The maximum number of bonds between a donor and acceptor to
            consider

    References:
        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
        assignment of accurate partial atomic charges: an electronegativity
        equalization method that accounts for alternate resonance forms." Journal of
        chemical information and computer sciences 43.6 (2003): 1982-1997.

    """

    # create a cache to speed up finding all paths from D->A and vice-versa
    path_cache = PathCache(nx_graph, max_path_length)

    open_list = {_graph_to_hash(nx_graph): nx_graph}
    closed_list: Dict[bytes, networkx.Graph] = {}

    while len(open_list) > 0:

        found_graphs: Dict[bytes, networkx.Graph] = {}

        for current_key, current_graph in open_list.items():

            if current_key in closed_list:
                continue

            closed_list[current_key] = current_graph

            atom_resonance_types = _find_donor_acceptors(current_graph)

            acceptor_indices = {
                atom_index
                for atom_index, atom_type in atom_resonance_types.items()
                if atom_type == "A"
            }
            donor_indices = {
                atom_index
                for atom_index, atom_type in atom_resonance_types.items()
                if atom_type == "D"
            }

            for acceptor_index, donor_index in itertools.product(
                acceptor_indices, donor_indices
            ):

                transfer_paths = _find_transfer_paths(
                    current_graph, acceptor_index, donor_index, path_cache
                )

                for transfer_path in transfer_paths:

                    flipped_graph = _perform_electron_transfer(
                        current_graph, transfer_path
                    )
                    flipped_key = _graph_to_hash(flipped_graph)

                    found_graphs[flipped_key] = flipped_graph

        open_list = found_graphs

    if lowest_energy_only:
        closed_list = _select_lowest_energy_forms(closed_list)

    return closed_list


def _find_sub_graphs(nx_graph: networkx.Graph) -> List[networkx.Graph]:
    """Attempts to split a graph into sub-graphs that contain a minimal number of,
    but at least one, acceptor donor pairs.

    Args:
        nx_graph: The graph to split.

    Returns:
        The set of found sub-graphs.
    """

    nx_graph = pickle.loads(pickle.dumps(nx_graph))  # faster than deepcopy

    # Start by dropping all hydrogen atoms as these are implicitly captures by the bond
    # orders tuple.
    hydrogen_indices = [
        atom_index
        for atom_index in nx_graph
        if nx_graph.nodes[atom_index]["element"] == "H"
    ]
    nx_graph.remove_nodes_from(hydrogen_indices)

    # Next prune all CX4 carbon atoms - because they have no double bonds to begin with
    # there is no electron transfer that can occur to change that  otherwise they will
    # end up pentavalent, and so can never be part of a conjugated path
    carbon_indices = [
        atom_index
        for atom_index in nx_graph
        if nx_graph.nodes[atom_index]["element"] == "C"
        and nx_graph.nodes[atom_index]["bond_orders"] == (1, 1, 1, 1)
        and nx_graph.nodes[atom_index]["formal_charge"] == 0
    ]
    nx_graph.remove_nodes_from(carbon_indices)

    original_sub_graphs = [
        nx_graph.subgraph(components)
        for components in networkx.connected_components(nx_graph)
    ]
    pruned_sub_graphs = []

    # Discard any sub-graphs that don't have at least one donor and one acceptor.
    for sub_graph in original_sub_graphs:

        resonance_types = {*_find_donor_acceptors(sub_graph).values()}

        if resonance_types != {"A", "D"}:
            continue

        pruned_sub_graphs.append(sub_graph)

    return pruned_sub_graphs


def _find_donor_acceptors(nx_graph: networkx.Graph) -> Dict[int, Literal["A", "D"]]:
    """Attempts to find any potential acceptor / donor atoms in a molecular graph
    and returns the resonance type key (see the ``_RESONANCE_TYPES`` dictionary)
    associated with each.

    Args:
        nx_graph: The molecular graph.

    Returns:
        A dictionary of the found acceptor / donor atoms and their respective resonance
        type key ('A' or 'D').
    """

    atom_types: Dict[int, Literal["A", "D"]] = {}

    for atom_index in nx_graph:

        node_attributes = nx_graph.nodes[atom_index]

        node_features = ResonanceTypeKey(
            node_attributes["element"],
            node_attributes["formal_charge"],
            node_attributes["bond_orders"],
        )

        if node_features not in _RESONANCE_TYPES:
            continue

        atom_types[atom_index] = _RESONANCE_TYPES[node_features].type

    return atom_types


def _find_transfer_paths(
    nx_graph: networkx.Graph,
    acceptor_index: int,
    donor_index: int,
    path_cache: PathCache,
) -> List[Tuple[int, ...]]:
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

    all_paths = path_cache.all_odd_n_simple_paths(donor_index, acceptor_index)

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
    nx_graph: networkx.Graph, transfer_path: Sequence[int]
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

    flipped_graph = pickle.loads(pickle.dumps(nx_graph))

    for node_index in [donor_index, acceptor_index]:

        node = flipped_graph.nodes[node_index]

        conjugate_key = _RESONANCE_KEYS_BY_ID[
            _RESONANCE_TYPES[ResonanceTypeKey(**node)].conjugate_id
        ]

        node["element"] = conjugate_key.element
        node["formal_charge"] = conjugate_key.formal_charge
        node["bond_orders"] = conjugate_key.bond_orders

    pairwise_path = networkx.utils.pairwise(transfer_path)

    for bond_index, (atom_index_a, atom_index_b) in enumerate(pairwise_path):

        increment = 1 if (bond_index % 2 == 0) else -1
        flipped_graph[atom_index_a][atom_index_b]["bond_order"] += increment

    return flipped_graph


def _select_lowest_energy_forms(
    resonance_forms: Dict[bytes, networkx.Graph]
) -> Dict[bytes, networkx.Graph]:
    """Select the lowest 'energy' resonance forms from an input list."""

    if len(resonance_forms) == 0:
        return {}

    resonance_form_energies = {}

    donor_acceptor_indices = {
        *_find_donor_acceptors(next(iter(resonance_forms.values())))
    }

    for graph_id, resonance_form in resonance_forms.items():

        resonance_keys = [
            ResonanceTypeKey(**resonance_form.nodes[atom_index])
            for atom_index in donor_acceptor_indices
        ]

        total_energy = sum(_RESONANCE_TYPES[key].energy for key in resonance_keys)

        resonance_form_energies[graph_id] = total_energy

    lowest_energy = min(resonance_form_energies.values())

    lowest_energy_forms = {
        graph_id: resonance_forms[graph_id]
        for graph_id, resonance_energy in resonance_form_energies.items()
        if numpy.isclose(resonance_energy, lowest_energy)
    }

    return lowest_energy_forms
