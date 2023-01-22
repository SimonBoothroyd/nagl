import abc
import typing

import pydantic
import torch

from nagl.utilities.resonance import enumerate_resonance_forms
from nagl.utilities.toolkits import normalize_molecule

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


_DEFAULT_ELEMENTS = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]
_DEFAULT_CONNECTIVITIES = [1, 2, 3, 4]
_DEFAULT_CHARGES = [-3, -2, -1, 0, 1, 2, 3]

_DEFAULT_BOND_ORDERS = [1, 2, 3]


def one_hot_encode(item, elements):

    return torch.tensor(
        [int(i == elements.index(item)) for i in range(len(elements))]
    ).reshape(1, -1)


class _Feature(abc.ABC):
    """The base class for features of molecules."""

    @abc.abstractmethod
    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        """A function which should generate the relevant feature tensor for the
        molecule.
        """


T = typing.TypeVar("T", bound=_Feature)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class _Featurizer(typing.Generic[T], abc.ABC):
    @classmethod
    def featurize(cls, molecule: "Molecule", features: typing.List[T]) -> torch.Tensor:
        """Featurizes a given molecule based on a given feature list."""
        return torch.hstack([feature(molecule) for feature in features])


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomFeature(_Feature, abc.ABC):
    """The base class for atomic features."""

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


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

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        """A function which should generate the relevant feature tensor for the
        molecule.
        """

        return torch.vstack(
            [one_hot_encode(atom.symbol, self.values) for atom in molecule.atoms]
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

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [one_hot_encode(len(atom.bonds), self.values) for atom in molecule.atoms]
        )

    def __len__(self):
        return len(self.values)


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomIsAromatic(AtomFeature):
    """Encodes whether each atom in a molecule is aromatic."""

    type: typing.Literal["is_aromatic"] = pydantic.Field("is_aromatic", const=True)

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.tensor([int(atom.is_aromatic) for atom in molecule.atoms]).reshape(
            -1, 1
        )

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class AtomIsInRing(AtomFeature):
    """Encodes whether each atom in a molecule is in a ring of any size."""

    type: typing.Literal["is_in_ring"] = pydantic.Field("is_in_ring", const=True)

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        ring_atoms = {
            index for index, in molecule.chemical_environment_matches("[*r:1]")
        }
        return torch.tensor(
            [int(i in ring_atoms) for i in range(molecule.n_atoms)]
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

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        from openff.units import unit

        return torch.vstack(
            [
                one_hot_encode(
                    atom.formal_charge.m_as(unit.elementary_charges),
                    self.values,
                )
                for atom in molecule.atoms
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

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

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
            for i in range(molecule.n_atoms)
        ]

        feature_tensor = torch.tensor(
            [
                [
                    sum(formal_charges[i]) / len(formal_charges[i])
                    if len(formal_charges[i]) > 0
                    else 0.0
                ]
                for i in range(molecule.n_atoms)
            ]
        )

        return feature_tensor

    def __len__(self):
        return 1


AtomFeatureType = typing.Union[
    AtomicElement,
    AtomConnectivity,
    AtomIsAromatic,
    AtomIsInRing,
    AtomFormalCharge,
    AtomAverageFormalCharge,
]


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
    def __call__(cls, molecule: "Molecule") -> torch.Tensor:

        return torch.tensor([int(bond.is_aromatic) for bond in molecule.bonds]).reshape(
            -1, 1
        )

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class BondIsInRing(BondFeature):
    """Encodes whether each bond in a molecule is in a ring of any size."""

    type: typing.Literal["is_in_ring"] = pydantic.Field("is_in_ring", const=True)

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        ring_bonds = {
            tuple(sorted(match))
            for match in molecule.chemical_environment_matches("[*:1]@[*:2]")
        }
        return torch.tensor(
            [
                int(tuple(sorted((bond.atom1_index, bond.atom2_index))) in ring_bonds)
                for bond in molecule.bonds
            ]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


@pydantic.dataclasses.dataclass(config={"extra": pydantic.Extra.forbid})
class WibergBondOrder(BondFeature):
    """Encodes the fractional Wiberg bond order of all of the bonds in a molecule."""

    type: typing.Literal["wbo"] = pydantic.Field("wbo", const=True)

    @classmethod
    def __call__(cls, molecule: "Molecule") -> torch.Tensor:
        return torch.tensor(
            [bond.fractional_bond_order for bond in molecule.bonds]
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

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [
                one_hot_encode(int(bond.bond_order), self.values)
                for bond in molecule.bonds
            ]
        )

    def __len__(self):
        return len(self.values)


class BondFeaturizer(_Featurizer[BondFeature]):
    """A class for featurizing the bonds in a molecule."""


BondFeatureType = typing.Union[BondIsAromatic, BondIsInRing, BondOrder]
