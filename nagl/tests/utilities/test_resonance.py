import networkx
import pytest
from openff.toolkit.topology import Molecule

from nagl.molecules import DGLMolecule
from nagl.utilities.resonance import (
    _dgl_molecule_from_networkx,
    _dgl_molecule_to_networkx,
    _extract_type_features,
    _find_transfer_paths,
    _openff_molecule_from_networkx,
    _openff_molecule_to_networkx,
    _perform_electron_transfer,
    enumerate_resonance_forms,
)


@pytest.fixture()
def openff_carboxylate() -> Molecule:
    return Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])([H:5])")


@pytest.fixture()
def nx_carboxylate(openff_carboxylate) -> networkx.Graph:
    return _openff_molecule_to_networkx(openff_carboxylate)


@pytest.mark.parametrize(
    "to_function, input_object",
    [
        (
            _openff_molecule_to_networkx,
            Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])([H:5])"),
        ),
        (
            _dgl_molecule_to_networkx,
            DGLMolecule.from_openff(
                Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])([H:5])"),
                [],
                [],
                False,
            ),
        ),
    ],
)
def test_xxx_to_networkx(to_function, input_object):

    nx_graph = to_function(input_object)
    assert isinstance(nx_graph, networkx.Graph)

    assert nx_graph.number_of_nodes() == 5
    assert nx_graph.number_of_edges() == 4

    assert [nx_graph.nodes[i]["element"] for i in nx_graph.nodes] == [
        "C",
        "O",
        "O",
        "H",
        "H",
    ]
    assert [nx_graph.nodes[i]["formal_charge"] for i in nx_graph.nodes] == [
        0,
        -1,
        0,
        0,
        0,
    ]
    assert [nx_graph.nodes[i]["bond_orders"] for i in nx_graph.nodes] == [
        (1, 1, 1, 2),
        (1,),
        (2,),
        (1,),
        (1,),
    ]

    assert nx_graph[0][1]["bond_order"] == 1
    assert nx_graph[0][2]["bond_order"] == 2


@pytest.mark.parametrize(
    "from_function", [_openff_molecule_from_networkx, _dgl_molecule_from_networkx]
)
def test_xxx_from_networkx(from_function):

    expected_molecule = Molecule.from_mapped_smiles(
        "[C:1]([O-:2])(=[O:3])([H:4])([H:5])"
    )

    nx_graph = _openff_molecule_to_networkx(expected_molecule)
    actual_molecule = from_function(nx_graph)

    if isinstance(actual_molecule, Molecule):

        are_isomorphic, atom_map = Molecule.are_isomorphic(
            expected_molecule, actual_molecule, return_atom_map=True
        )

        assert are_isomorphic
        assert atom_map == {i: i for i in range(5)}


@pytest.mark.parametrize(
    "input_smiles, expected_smiles, lowest_energy_only",
    [
        (
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            [
                "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
                "[H]c1c(c([n+](c(c1[H])[N+](=[N+]([O-])[O-])[H])[O-])[H])[H]",
            ],
            False,
        ),
        (
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            [
                "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
            ],
            True,
        ),
    ],
)
def test_enumerate_resonance_forms(input_smiles, expected_smiles, lowest_energy_only):

    input_molecule: Molecule = Molecule.from_smiles(input_smiles)
    actual_molecules = enumerate_resonance_forms(input_molecule, lowest_energy_only)

    assert (
        sorted(molecule.to_smiles() for molecule in actual_molecules) == expected_smiles
    )


def test_extract_type_features(nx_carboxylate):
    assert _extract_type_features(nx_carboxylate) == {1: "D", 2: "A"}


def test_find_transfer_paths():

    molecule: Molecule = Molecule.from_smiles("[NH2+]=C1C=CNC=C1")

    [(acceptor_index,)] = molecule.chemical_environment_matches("[#7+1:1]")
    [(donor_index,)] = molecule.chemical_environment_matches("[#7+0:1]")

    expected_paths = molecule.chemical_environment_matches(
        "[#7+0:1]~[*:2]~[*:3]~[*:4]~[#7+1:5]"
    )

    transfer_paths = _find_transfer_paths(
        _openff_molecule_to_networkx(molecule), acceptor_index, donor_index
    )

    assert sorted(expected_paths) == sorted(map(tuple, transfer_paths))


def test_perform_electron_transfer(nx_carboxylate):

    final_graph = _perform_electron_transfer(nx_carboxylate, [1, 0, 2])

    assert final_graph.nodes[1]["formal_charge"] == 0
    assert final_graph.nodes[2]["formal_charge"] == -1
    assert nx_carboxylate.nodes[1]["formal_charge"] == -1
    assert nx_carboxylate.nodes[2]["formal_charge"] == 0

    assert final_graph[0][1]["bond_order"] == 2
    assert final_graph[0][2]["bond_order"] == 1
    assert nx_carboxylate[0][1]["bond_order"] == 1
    assert nx_carboxylate[0][2]["bond_order"] == 2
