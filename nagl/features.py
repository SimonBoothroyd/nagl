import abc
import dataclasses
import typing

import pydantic
import torch
from rdkit import Chem

from nagl.utilities.molecule import BOND_TYPE_TO_ORDER, normalize_molecule
from nagl.utilities.resonance import enumerate_resonance_forms

_DEFAULT_ELEMENTS = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]
_DEFAULT_CONNECTIVITIES = [1, 2, 3, 4]
_DEFAULT_CHARGES = [-3, -2, -1, 0, 1, 2, 3]

_DEFAULT_BOND_ORDERS = [1, 2, 3]

_CUSTOM_ATOM_FEATURES = {}
_CUSTOM_BOND_FEATURES = {}


def one_hot_encode(item, elements):

    return torch.tensor(
        [int(i == elements.index(item)) for i in range(len(elements))]
    ).reshape(1, -1)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class _Feature(abc.ABC):
    """The base class for features of molecules."""

    @abc.abstractmethod
    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        """A function which should generate the relevant feature tensor for the
        molecule.
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """The number of columns associated with this feature."""


def _get_custom_feature(
    wrapper: _Feature, features: typing.Dict[str, typing.Type[_Feature]]
) -> _Feature:
    """Return an instance of a feature defined outside of NAGL based on a
    ``CustomXXXFeature`` wrapper.

    Args:
        wrapper: The ``CustomXXXFeature`` wrapper.
        features: The feature registry that external features should be searched in.

    Returns:
        An instance of the external feature.
    """

    type_ = wrapper.type

    if type_ not in features:
        raise KeyError(f"No feature with type={type_} could be found.")

    feature_class = features[type_]

    features_kwargs = dataclasses.asdict(wrapper)
    features_kwargs.pop("type")

    return feature_class(**features_kwargs)


T = typing.TypeVar("T", bound=_Feature)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class _Featurizer(typing.Generic[T], abc.ABC):
    @classmethod
    def featurize(cls, molecule: Chem.Mol, features: typing.List[T]) -> torch.Tensor:
        """Featurizes a given molecule based on a given feature list."""
        return torch.hstack([feature(molecule) for feature in features])


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomFeature(_Feature, abc.ABC):
    """The base class for atomic features."""


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomicElement(AtomFeature):
    """One-hot encodes the atomic element of each atom in a molecule."""

    type: typing.Literal["element"] = pydantic.Field("element", const=True)

    values: typing.List[str] = pydantic.Field(
        _DEFAULT_ELEMENTS,
        description="The elements to include in the one-hot encoding in the order in "
        "which they should be encoded. If none are provided, the default set of "
        f"{_DEFAULT_ELEMENTS} will be used.",
    )

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        """A function which should generate the relevant feature tensor for the
        molecule.
        """

        return torch.vstack(
            [
                one_hot_encode(atom.GetSymbol(), self.values)
                for atom in molecule.GetAtoms()
            ]
        )

    def __len__(self):
        return len(self.values)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomConnectivity(AtomFeature):
    """One-hot encodes the connectivity (i.e. the number of bonds) of each atom in a
    molecule.
    """

    type: typing.Literal["connectivity"] = pydantic.Field("connectivity", const=True)

    values: typing.List[int] = pydantic.Field(
        _DEFAULT_CONNECTIVITIES,
        description="The connectivities (i.e. number of bonds to an atom) to include in "
        "the one-hot encoding in the order in which they should be encoded. If none "
        f"are provided, the default set of {_DEFAULT_CONNECTIVITIES} will be used.",
    )

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        return torch.vstack(
            [
                one_hot_encode(len(atom.GetBonds()), self.values)
                for atom in molecule.GetAtoms()
            ]
        )

    def __len__(self):
        return len(self.values)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomIsAromatic(AtomFeature):
    """Encodes whether each atom in a molecule is aromatic."""

    type: typing.Literal["is_aromatic"] = pydantic.Field("is_aromatic", const=True)

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        molecule = Chem.Mol(molecule)
        Chem.SetAromaticity(molecule, Chem.AROMATICITY_RDKIT)

        return torch.tensor(
            [int(atom.GetIsAromatic()) for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomIsInRing(AtomFeature):
    """Encodes whether each atom in a molecule is in a ring of any size."""

    type: typing.Literal["is_in_ring"] = pydantic.Field("is_in_ring", const=True)

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        return torch.tensor(
            [int(atom.IsInRing()) for atom in molecule.GetAtoms()]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomFormalCharge(AtomFeature):
    """One-hot encodes the formal charge on each atom in a molecule."""

    type: typing.Literal["formal_charge"] = pydantic.Field("formal_charge", const=True)

    values: typing.List[int] = pydantic.Field(
        _DEFAULT_CHARGES,
        description="The charges to include in the one-hot encoding in the order in "
        "which they should be encoded. If none are provided, the default set of "
        f"{_DEFAULT_CHARGES} will be used.",
    )

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        return torch.vstack(
            [
                one_hot_encode(
                    atom.GetFormalCharge(),
                    self.values,
                )
                for atom in molecule.GetAtoms()
            ]
        )

    def __len__(self):
        return len(self.values)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomAverageFormalCharge(AtomFeature):
    """Computes the average formal charge on each atom in a molecule across resonance
    structures."""

    type: typing.Literal["avg_formal_charge"] = pydantic.Field(
        "avg_formal_charge", const=True
    )

    lowest_energy_only: bool = pydantic.Field(
        True,
        description="Whether to only return the resonance forms with the lowest "
        "'energy'. See ``nagl.resonance.enumerate_resonance_forms`` for details.",
    )
    max_path_length: typing.Optional[int] = pydantic.Field(
        None,
        description="The maximum number of bonds between a donor and acceptor to "
        "consider.",
    )
    include_all_transfer_pathways: bool = pydantic.Field(
        False,
        description="Whether to include resonance forms that have the same formal "
        "charges but have different arrangements of bond orders.",
    )

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        molecule = normalize_molecule(molecule)

        resonance_forms = enumerate_resonance_forms(
            molecule,
            as_dicts=True,
            lowest_energy_only=self.lowest_energy_only,
            include_all_transfer_pathways=self.include_all_transfer_pathways,
            max_path_length=self.max_path_length,
        )

        formal_charges = [
            [
                atom["formal_charge"]
                for resonance_form in resonance_forms
                if i in resonance_form["atoms"]
                for atom in resonance_form["atoms"][i]
            ]
            for i in range(molecule.GetNumAtoms())
        ]

        feature_tensor = torch.tensor(
            [
                [
                    sum(formal_charges[i]) / len(formal_charges[i])
                    if len(formal_charges[i]) > 0
                    else 0.0
                ]
                for i in range(molecule.GetNumAtoms())
            ]
        )

        return feature_tensor

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.allow})
class CustomAtomFeature(AtomFeature):
    """A wrapper around a custom atom feature defined outside of NAGL."""

    type: str = pydantic.Field(..., description="The custom feature type.")

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        feature = _get_custom_feature(self, _CUSTOM_ATOM_FEATURES)
        return feature(molecule)

    def __len__(self):
        feature = _get_custom_feature(self, _CUSTOM_ATOM_FEATURES)
        return len(feature)


AtomFeatureType = typing.Union[
    AtomicElement,
    AtomConnectivity,
    AtomIsAromatic,
    AtomIsInRing,
    AtomFormalCharge,
    AtomAverageFormalCharge,
    CustomAtomFeature,
]


def register_atom_feature(feature_cls: typing.Type[AtomFeature]):
    """Register a class of atom feature for use with NAGL.

    This will make the feature available from the model configuration.
    """

    if not issubclass(feature_cls, AtomFeature):
        raise TypeError("feature should subclass `AtomFeature`")
    if not hasattr(feature_cls, "type"):
        raise AttributeError(f"{feature_cls.__name__} has no `type` attribute.")

    _CUSTOM_ATOM_FEATURES[feature_cls.type] = feature_cls


class AtomFeaturizer(_Featurizer[AtomFeature]):
    """A class for featurizing the atoms in a molecule."""


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class BondFeature(_Feature, abc.ABC):
    """The base class for bond features."""


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class BondIsAromatic(BondFeature):
    """Encodes whether each bond in a molecule is aromatic."""

    type: typing.Literal["is_aromatic"] = pydantic.Field("is_aromatic", const=True)

    @classmethod
    def __call__(cls, molecule: Chem.Mol) -> torch.Tensor:

        molecule = Chem.Mol(molecule)
        Chem.SetAromaticity(molecule, Chem.AROMATICITY_RDKIT)

        return torch.tensor(
            [int(bond.GetIsAromatic()) for bond in molecule.GetBonds()]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class BondIsInRing(BondFeature):
    """Encodes whether each bond in a molecule is in a ring of any size."""

    type: typing.Literal["is_in_ring"] = pydantic.Field("is_in_ring", const=True)

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        return torch.tensor(
            [int(bond.IsInRing()) for bond in molecule.GetBonds()]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class BondOrder(BondFeature):
    """One-hot encodes the formal bond order of all of the bonds in a molecule."""

    type: typing.Literal["bond_order"] = pydantic.Field("bond_order", const=True)

    values: typing.List[int] = pydantic.Field(
        _DEFAULT_BOND_ORDERS,
        description="The bond orders to include in the one-hot encoding in the order "
        "in which they should be encoded. If none are provided, the default set of "
        f"{_DEFAULT_BOND_ORDERS} will be used.",
    )

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:

        return torch.vstack(
            [
                one_hot_encode(BOND_TYPE_TO_ORDER[bond.GetBondType()], self.values)
                for bond in molecule.GetBonds()
            ]
        )

    def __len__(self):
        return len(self.values)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.allow})
class CustomBondFeature(BondFeature):
    """A wrapper around a custom bond feature defined outside of NAGL."""

    type: str = pydantic.Field(..., description="The custom feature type.")

    def __call__(self, molecule: Chem.Mol) -> torch.Tensor:
        feature = _get_custom_feature(self, _CUSTOM_BOND_FEATURES)
        return feature(molecule)

    def __len__(self):
        feature = _get_custom_feature(self, _CUSTOM_BOND_FEATURES)
        return len(feature)


BondFeatureType = typing.Union[
    BondIsAromatic, BondIsInRing, BondOrder, CustomBondFeature
]


def register_bond_feature(feature_cls: typing.Type[BondFeature]):
    """Register a class of bond feature for use with NAGL.

    This will make the feature available from the model configuration.
    """

    if not issubclass(feature_cls, BondFeature):
        raise TypeError("feature should subclass `BondFeature`")
    if not hasattr(feature_cls, "type"):
        raise AttributeError(f"{feature_cls.__name__} has no `type` attribute.")

    _CUSTOM_BOND_FEATURES[feature_cls.type] = feature_cls


class BondFeaturizer(_Featurizer[BondFeature]):
    """A class for featurizing the bonds in a molecule."""
