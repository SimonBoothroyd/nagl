import numpy
import pytest
from openff.toolkit.topology import Molecule

from nagl.features import (
    AtomConnectivity,
    AtomFormalCharge,
    AtomicElement,
    AtomIsAromatic,
    AtomIsInRing,
    BondIsAromatic,
    BondIsInRing,
    BondOrder,
    WibergBondOrder,
    one_hot_encode,
)


def test_one_hot_encode():

    encoding = one_hot_encode("b", ["a", "b", "c"]).numpy()
    assert numpy.allclose(encoding, numpy.array([0, 1, 0]))


def test_atomic_element(openff_methane: Molecule):

    encoding = AtomicElement(["H", "C"])(openff_methane).numpy()

    assert encoding.shape == (5, 2)

    assert numpy.allclose(encoding[1:, 0], 1.0)
    assert numpy.allclose(encoding[1:, 1], 0.0)

    assert numpy.isclose(encoding[0, 0], 0.0)
    assert numpy.isclose(encoding[0, 1], 1.0)


def test_atom_connectivity(openff_methane: Molecule):

    encoding = AtomConnectivity()(openff_methane).numpy()

    assert encoding.shape == (5, 4)

    assert numpy.allclose(encoding[1:, 0], 1.0)
    assert numpy.isclose(encoding[0, 3], 1.0)


def test_atom_formal_charge():

    molecule = Molecule.from_smiles("[Cl-]")

    encoding = AtomFormalCharge([0, -1])(molecule).numpy()
    assert encoding.shape == (1, 2)

    assert numpy.isclose(encoding[0, 0], 0.0)
    assert numpy.isclose(encoding[0, 1], 1.0)


@pytest.mark.parametrize("feature_class", [AtomIsAromatic, BondIsAromatic])
def test_is_aromatic(feature_class):

    molecule = Molecule.from_smiles("c1ccccc1")

    encoding = feature_class()(molecule).numpy()
    assert encoding.shape == (12, 2)

    assert numpy.allclose(encoding[:6, 0], 0.0)
    assert numpy.allclose(encoding[:6, 1], 1.0)

    assert numpy.allclose(encoding[6:, 0], 1.0)
    assert numpy.allclose(encoding[6:, 1], 0.0)


@pytest.mark.parametrize("feature_class", [AtomIsInRing, BondIsInRing])
def test_is_in_ring(feature_class):

    molecule = Molecule.from_smiles("c1ccccc1")

    encoding = AtomIsInRing()(molecule).numpy()
    assert encoding.shape == (12, 2)

    assert numpy.allclose(encoding[:6, 0], 0.0)
    assert numpy.allclose(encoding[:6, 1], 1.0)

    assert numpy.allclose(encoding[6:, 0], 1.0)
    assert numpy.allclose(encoding[6:, 1], 0.0)


def test_bond_order():

    encoding = BondOrder()(Molecule.from_smiles("C=O")).numpy()
    assert encoding.shape == (3, 1)

    assert numpy.allclose(encoding, numpy.array([[2.0], [1.0], [1.0]]))


def test_wiberg_bond_order(openff_methane):

    for i, bond in enumerate(openff_methane.bonds):
        bond.fractional_bond_order = float(i)

    encoding = WibergBondOrder()(openff_methane).numpy()
    assert encoding.shape == (4, 1)

    assert numpy.allclose(encoding, numpy.arange(4).reshape(-1, 1))
