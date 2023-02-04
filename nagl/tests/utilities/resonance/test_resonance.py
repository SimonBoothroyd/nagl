import pytest
from rdkit import Chem

from nagl.utilities.molecule import molecule_from_mapped_smiles, molecule_from_smiles
from nagl.utilities.resonance._resonance import (
    PathCache,
    _find_donor_acceptors,
    _find_sub_graphs,
    _find_transfer_paths,
    _graph_to_hash,
    _graphs_to_dicts,
    _perform_electron_transfer,
    _select_lowest_energy_forms,
    enumerate_resonance_forms,
    rdkit_molecule_from_networkx,
    rdkit_molecule_to_networkx,
)


@pytest.mark.parametrize(
    "input_smiles, "
    "n_expected, "
    "expected_smiles, "
    "lowest_energy_only, "
    "include_all_transfer_pathways",
    [
        (
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            6,
            [
                "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
                "[H]c1c(c([n+](c(c1[H])[N+](=[N+]([O-])[O-])[H])[O-])[H])[H]",
            ],
            False,
            False,
        ),
        (
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            9,
            [
                "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
                "[H]c1c(c([n+](c(c1[H])[N+](=[N+]([O-])[O-])[H])[O-])[H])[H]",
            ],
            False,
            True,
        ),
        (
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            5,
            [
                "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
            ],
            True,
            False,
        ),
        (
            "[O-][N+](=O)Nc1cccc[n+]1[O-]",
            7,
            [
                "[H]C1=C(C(=[N+]([H])N([O-])[O-])[N+](=O)C(=C1[H])[H])[H]",
                "[H]C1=C(C(=[N+]([H])[N+](=O)[O-])N(C(=C1[H])[H])[O-])[H]",
                "[H]c1c(c([n+](c(c1[H])N([H])[N+](=O)[O-])[O-])[H])[H]",
            ],
            True,
            True,
        ),
        ("C", 1, ["C"], False, False),
        ("C", 1, ["C"], True, False),
    ],
)
def test_enumerate_resonance_forms(
    input_smiles,
    expected_smiles,
    n_expected,
    lowest_energy_only,
    include_all_transfer_pathways,
):

    input_molecule = molecule_from_smiles(input_smiles)

    actual_molecules = enumerate_resonance_forms(
        input_molecule,
        lowest_energy_only,
        as_dicts=False,
        include_all_transfer_pathways=include_all_transfer_pathways,
    )
    assert len(actual_molecules) == n_expected

    expected_smiles = {
        Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smiles)))
        for smiles in expected_smiles
    }
    assert {
        Chem.MolToSmiles(molecule) for molecule in actual_molecules
    } == expected_smiles


def test_graphs_to_dicts():

    sub_graphs = [
        rdkit_molecule_to_networkx(molecule_from_mapped_smiles(smiles))
        for smiles in [
            "[N:1]([H:2])([H:3])[C:4](=[O:5])[H:6]",
            "[N+:1]([H:2])([H:3])=[C:4]([O-:5])[H:6]",
        ]
    ]
    resonance_sub_graphs_by_hash = [
        {_graph_to_hash(sub_graph, True): sub_graph for sub_graph in sub_graphs}
    ]
    dicts = _graphs_to_dicts(resonance_sub_graphs_by_hash)

    assert dicts == [
        {
            "atoms": {
                0: [
                    {"formal_charge": 0, "type": "D"},
                    {"formal_charge": 1, "type": "A"},
                ],
                4: [
                    {"formal_charge": 0, "type": "A"},
                    {"formal_charge": -1, "type": "D"},
                ],
            },
            "bonds": {(0, 3): [1, 2], (3, 4): [2, 1]},
        }
    ]


@pytest.mark.parametrize(
    "smiles, expected_groups",
    [
        (
            "[C:1](=[O:4])([O-:5])[C:2]([H:8])([H:9])[C:3](=[O:6])([O-:7])",
            [(0, 3, 4), (2, 5, 6)],
        ),
        ("[C:1](=[O:3])([O-:4])[C:2](=[O:5])([O-:6])", [(0, 1, 2, 3, 4, 5)]),
    ],
)
def test_find_sub_graphs(smiles, expected_groups):

    nx_graph = rdkit_molecule_to_networkx(molecule_from_mapped_smiles(smiles))
    sub_graphs = _find_sub_graphs(nx_graph)

    assert (
        sorted(tuple(sorted(sub_graph.nodes)) for sub_graph in sub_graphs)
        == expected_groups
    )


def test_find_donor_acceptors(nx_carboxylate):
    assert _find_donor_acceptors(nx_carboxylate) == {1: "D", 2: "A"}


def test_find_transfer_paths():

    molecule = molecule_from_smiles("[NH2+:1]=[C:2]1[C:3]=[C:4][N:5][C:6]=[C:7]1")

    (acceptor_index,) = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == 1
    ]
    (donor_index,) = [
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomMapNum() == 5
    ]

    map_idx_to_idx = {
        atom.GetAtomMapNum(): atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomMapNum() > 0
    }

    expected_paths = [
        tuple(map_idx_to_idx[map_idx] for map_idx in expected_path)
        for expected_path in [(5, 4, 3, 2, 1), (5, 6, 7, 2, 1)]
    ]

    nx_graph = rdkit_molecule_to_networkx(molecule)

    transfer_paths = _find_transfer_paths(
        nx_graph, acceptor_index, donor_index, PathCache(nx_graph)
    )

    assert sorted(expected_paths) == sorted(transfer_paths)


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


def test_select_lowest_energy_forms():

    input_molecules = [
        molecule_from_mapped_smiles("[N:1]([H:2])([H:3])[C:4](=[O:5])[H:6]"),
        molecule_from_mapped_smiles("[N+:1]([H:2])([H:3])=[C:4]([O-:5])[H:6]"),
    ]

    lowest_energy_forms = _select_lowest_energy_forms(
        {
            str(i).encode(): rdkit_molecule_to_networkx(molecule)
            for i, molecule in enumerate(input_molecules)
        }
    )
    assert len(lowest_energy_forms) == 1

    lowest_energy_form = rdkit_molecule_from_networkx(lowest_energy_forms[b"0"])
    assert Chem.MolToSmiles(Chem.AddHs(lowest_energy_form)) == Chem.MolToSmiles(
        input_molecules[0]
    )
