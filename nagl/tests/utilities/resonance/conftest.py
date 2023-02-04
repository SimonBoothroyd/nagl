import networkx
import pytest
from rdkit import Chem

from nagl.utilities.molecule import molecule_from_mapped_smiles
from nagl.utilities.resonance._conversion import rdkit_molecule_to_networkx


@pytest.fixture()
def rdkit_carboxylate() -> Chem.Mol:
    return molecule_from_mapped_smiles("[C:1]([O-:2])(=[O:3])([H:4])")


@pytest.fixture()
def nx_carboxylate(rdkit_carboxylate) -> networkx.Graph:
    return rdkit_molecule_to_networkx(rdkit_carboxylate)
