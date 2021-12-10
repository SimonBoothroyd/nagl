import networkx
import pytest
from openff.toolkit.topology import Molecule

from nagl.resonance._conversion import openff_molecule_to_networkx


@pytest.fixture()
def openff_carboxylate() -> Molecule:
    return Molecule.from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])")


@pytest.fixture()
def nx_carboxylate(openff_carboxylate) -> networkx.Graph:
    return openff_molecule_to_networkx(openff_carboxylate)
