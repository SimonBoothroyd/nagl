import typing

import numpy
import pydantic
import pytest
import torch
from rdkit import Chem

from nagl.features import (
    AtomAverageFormalCharge,
    AtomConnectivity,
    AtomFeature,
    AtomFeatureType,
    AtomFormalCharge,
    AtomicElement,
    AtomIsAromatic,
    AtomIsInRing,
    BondFeature,
    BondFeatureType,
    BondIsAromatic,
    BondIsInRing,
    BondOrder,
    CustomAtomFeature,
    CustomBondFeature,
    one_hot_encode,
    register_atom_feature,
    register_bond_feature,
)
from nagl.utilities.molecule import molecule_from_mapped_smiles, molecule_from_smiles


@pytest.fixture()
def custom_atom_feature_registry(monkeypatch):

    import nagl.features

    feature_registry = {}
    monkeypatch.setattr(nagl.features, "_CUSTOM_ATOM_FEATURES", feature_registry)
    return feature_registry


@pytest.fixture()
def custom_bond_feature_registry(monkeypatch):

    import nagl.features

    feature_registry = {}
    monkeypatch.setattr(nagl.features, "_CUSTOM_BOND_FEATURES", feature_registry)
    return feature_registry


def test_one_hot_encode():

    encoding = one_hot_encode("b", ["a", "b", "c"]).numpy()
    assert numpy.allclose(encoding, numpy.array([0, 1, 0]))


def test_atomic_element(rdkit_methane: Chem.Mol):

    feature = AtomicElement(values=["H", "C"])
    assert len(feature) == 2

    encoding = feature(rdkit_methane).numpy()

    assert encoding.shape == (5, 2)

    assert numpy.allclose(encoding[1:, 0], 1.0)
    assert numpy.allclose(encoding[1:, 1], 0.0)

    assert numpy.isclose(encoding[0, 0], 0.0)
    assert numpy.isclose(encoding[0, 1], 1.0)


def test_atom_connectivity(rdkit_methane: Chem.Mol):

    feature = AtomConnectivity()
    assert len(feature) == 4

    encoding = feature(rdkit_methane).numpy()

    assert encoding.shape == (5, 4)

    assert numpy.allclose(encoding[1:, 0], 1.0)
    assert numpy.isclose(encoding[0, 3], 1.0)


def test_atom_formal_charge():

    molecule = molecule_from_smiles("[Cl-]")

    feature = AtomFormalCharge(values=[0, -1])
    assert len(feature) == 2

    encoding = feature(molecule).numpy()
    assert encoding.shape == (1, 2)

    assert numpy.isclose(encoding[0, 0], 0.0)
    assert numpy.isclose(encoding[0, 1], 1.0)


def test_atom_average_formal_charge():

    molecule = molecule_from_mapped_smiles("[H:1][C:2](=[O:3])[O-:4]")

    feature = AtomAverageFormalCharge()
    assert len(feature) == 1

    encoding = feature(molecule).numpy()
    assert encoding.shape == (4, 1)

    expected_encoding = numpy.array([[0], [0], [-0.5], [-0.5]])
    assert numpy.allclose(encoding, expected_encoding)


def test_custom_atom_feature(custom_atom_feature_registry):
    class MyAtomFeature(AtomFeature):

        type: typing.Literal["some-type"] = "some-type"

        def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
            return torch.unsqueeze(torch.arange(start=1, end=5), 1)

        def __len__(self):
            return 4

    register_atom_feature(MyAtomFeature)

    feature = CustomAtomFeature(type="some-type")
    assert len(feature) == 4

    output = feature(None)

    expected_output = torch.tensor([[1], [2], [3], [4]])
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output)


def test_discriminate_atom_feature():
    @pydantic.dataclasses.dataclass
    class MyConfig:
        atom_feature: AtomFeatureType

    model_a = MyConfig(atom_feature=AtomicElement())
    assert isinstance(model_a.atom_feature, AtomicElement)

    model_a = MyConfig(atom_feature=CustomAtomFeature(type="some-type"))
    assert isinstance(model_a.atom_feature, CustomAtomFeature)


@pytest.mark.parametrize("feature_class", [AtomIsAromatic, BondIsAromatic])
def test_is_aromatic(feature_class):

    molecule = molecule_from_smiles("c1ccccc1")

    feature = feature_class()
    assert len(feature) == 1

    encoding = feature(molecule).numpy()
    assert encoding.shape == (12, 1)

    assert numpy.allclose(encoding[:6], 1.0)
    assert numpy.allclose(encoding[6:], 0.0)


@pytest.mark.parametrize("feature_class", [AtomIsInRing, BondIsInRing])
def test_is_in_ring(feature_class):

    molecule = molecule_from_smiles("c1ccccc1")

    feature = feature_class()
    assert len(feature) == 1

    encoding = feature(molecule).numpy()
    assert encoding.shape == (12, 1)

    assert numpy.allclose(encoding[:6], 1.0)
    assert numpy.allclose(encoding[6:], 0.0)


def test_bond_order():

    feature = BondOrder(values=[2, 1])
    assert len(feature) == 2

    encoding = feature(molecule_from_smiles("C=O")).numpy()
    assert encoding.shape == (3, 2)

    assert numpy.allclose(encoding, numpy.array([[1, 0], [0, 1], [0, 1]]))


def test_custom_bond_feature(custom_bond_feature_registry):
    class MyBondFeature(BondFeature):

        type: typing.Literal["some-type"] = "some-type"

        def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
            return torch.unsqueeze(torch.arange(start=1, end=5), 1)

        def __len__(self):
            return 4

    register_bond_feature(MyBondFeature)

    feature = CustomBondFeature(type="some-type")
    assert len(feature) == 4

    output = feature(None)

    expected_output = torch.tensor([[1], [2], [3], [4]])
    assert output.shape == expected_output.shape
    assert torch.allclose(expected_output, output)


def test_discriminate_bond_feature():
    @pydantic.dataclasses.dataclass
    class MyConfig:
        bond_feature: BondFeatureType

    model_a = MyConfig(bond_feature=BondOrder())
    assert isinstance(model_a.bond_feature, BondOrder)

    model_a = MyConfig(bond_feature=CustomBondFeature(type="some-type"))
    assert isinstance(model_a.bond_feature, CustomBondFeature)


def test_register_atom_feature(custom_atom_feature_registry):
    class MyAtomFeature(AtomFeature):
        type: typing.Literal["some-type"] = "some-type"

    register_atom_feature(MyAtomFeature)
    assert custom_atom_feature_registry == {"some-type": MyAtomFeature}


@pytest.mark.parametrize(
    "feature_cls, expected_raises",
    [
        (str, pytest.raises(TypeError, match="feature should subclass `AtomFeature`")),
        (AtomFeature, pytest.raises(AttributeError, match="has no `type` attribute.")),
    ],
)
def test_register_atom_feature_raises(
    feature_cls, expected_raises, custom_atom_feature_registry
):

    with expected_raises:
        register_atom_feature(feature_cls)

    assert custom_atom_feature_registry == {}


def test_register_bond_feature(custom_bond_feature_registry):
    class MyBondFeature(BondFeature):
        type: typing.Literal["some-type"] = "some-type"

    register_bond_feature(MyBondFeature)
    assert custom_bond_feature_registry == {"some-type": MyBondFeature}


@pytest.mark.parametrize(
    "feature_cls, expected_raises",
    [
        (str, pytest.raises(TypeError, match="feature should subclass `BondFeature`")),
        (BondFeature, pytest.raises(AttributeError, match="has no `type` attribute.")),
    ],
)
def test_register_bond_feature_raises(
    feature_cls, expected_raises, custom_bond_feature_registry
):

    with expected_raises:
        register_bond_feature(feature_cls)

    assert custom_bond_feature_registry == {}
