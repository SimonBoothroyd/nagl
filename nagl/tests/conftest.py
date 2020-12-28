import numpy
import pytest
from openforcefield.topology import Molecule
from simtk import unit

from nagl.dataset.dataset import molecule_to_graph
from nagl.dataset.features import AtomConnectivity, BondIsInRing


@pytest.fixture()
def methane() -> Molecule:

    molecule: Molecule = Molecule.from_smiles("C")
    molecule.add_conformer(
        numpy.array(
            [
                [-0.0000658, -0.0000061, 0.0000215],
                [-0.0566733, 1.0873573, -0.0859463],
                [0.6194599, -0.3971111, -0.8071615],
                [-1.0042799, -0.4236047, -0.0695677],
                [0.4415590, -0.2666354, 0.9626540],
            ]
        )
        * unit.angstrom
    )

    return molecule


@pytest.fixture()
def methane_graph(methane):
    return molecule_to_graph(methane, [AtomConnectivity()], [BondIsInRing()])
