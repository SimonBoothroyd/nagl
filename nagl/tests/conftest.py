import pathlib

import numpy
import pytest
from rdkit import Chem, Geometry

from nagl.features import AtomConnectivity, AtomicElement, AtomIsInRing, BondIsInRing
from nagl.molecules import DGLMolecule
from nagl.utilities.molecule import molecule_from_smiles


@pytest.fixture()
def rdkit_methane() -> Chem.Mol:
    molecule = molecule_from_smiles("C")
    conformer = Chem.Conformer(molecule.GetNumAtoms())

    coords = numpy.array(
        [
            [-0.0000658, -0.0000061, 0.0000215],
            [-0.0566733, 1.0873573, -0.0859463],
            [0.6194599, -0.3971111, -0.8071615],
            [-1.0042799, -0.4236047, -0.0695677],
            [0.4415590, -0.2666354, 0.9626540],
        ]
    )

    for i, coord in enumerate(coords):
        conformer.SetAtomPosition(i, Geometry.Point3D(*coord))

    molecule.AddConformer(conformer)
    return molecule


@pytest.fixture()
def dgl_methane(rdkit_methane) -> DGLMolecule:
    return DGLMolecule.from_rdkit(
        rdkit_methane,
        [AtomicElement(), AtomConnectivity(), AtomIsInRing()],
        [BondIsInRing()],
    )


@pytest.fixture
def tmp_cwd(tmp_path, monkeypatch) -> pathlib.Path:
    monkeypatch.chdir(tmp_path)
    yield tmp_path
