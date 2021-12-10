import networkx
import pytest
from openff.toolkit.topology import Molecule

from nagl.molecules import DGLMolecule
from nagl.resonance._conversion import (
    dgl_molecule_from_networkx,
    dgl_molecule_to_networkx,
    openff_molecule_from_networkx,
    openff_molecule_to_networkx,
)


@pytest.mark.parametrize(
    "to_function, input_object",
    [
        (
            openff_molecule_to_networkx,
            Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])"),
        ),
        (
            dgl_molecule_to_networkx,
            DGLMolecule.from_openff(
                Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])"),
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

    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3

    assert [nx_graph.nodes[i]["element"] for i in nx_graph.nodes] == [
        "C",
        "O",
        "O",
        "H",
    ]
    assert [nx_graph.nodes[i]["formal_charge"] for i in nx_graph.nodes] == [
        0,
        -1,
        0,
        0,
    ]
    assert [nx_graph.nodes[i]["bond_orders"] for i in nx_graph.nodes] == [
        (1, 1, 2),
        (1,),
        (2,),
        (1,),
    ]

    assert nx_graph[0][1]["bond_order"] == 1
    assert nx_graph[0][2]["bond_order"] == 2


@pytest.mark.parametrize(
    "from_function", [openff_molecule_from_networkx, dgl_molecule_from_networkx]
)
def test_xxx_from_networkx(from_function):

    expected_molecule = Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])")

    nx_graph = openff_molecule_to_networkx(expected_molecule)
    actual_molecule = from_function(nx_graph)

    if isinstance(actual_molecule, Molecule):

        are_isomorphic, atom_map = Molecule.are_isomorphic(
            expected_molecule, actual_molecule, return_atom_map=True
        )

        assert are_isomorphic
        assert atom_map == {i: i for i in range(4)}
