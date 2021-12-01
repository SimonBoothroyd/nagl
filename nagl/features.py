import abc
from typing import TYPE_CHECKING, Generic, List, Optional, TypeVar

import torch

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


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


T = TypeVar("T", bound=_Feature)


class _Featurizer(Generic[T], abc.ABC):
    @classmethod
    def featurize(cls, molecule: "Molecule", features: List[T]) -> torch.Tensor:
        """Featurizes a given molecule based on a given feature list."""
        return torch.hstack([feature(molecule) for feature in features])


class AtomFeature(_Feature, abc.ABC):
    """The base class for atomic features."""

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class AtomicElement(AtomFeature):
    """One-hot encodes the atomic element of each atom in a molecule."""

    _DEFAULT_ELEMENTS = ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"]

    def __init__(self, elements: Optional[List[str]] = None):
        """
        Parameters
        ----------
        elements
            The elements to include in the one-hot encoding in the order in which they
            should be encoded. If none are provided, the default set of
            ["H", "C", "N", "O", "F", "Cl", "Br", "S", "P"] will be used.
        """
        self.elements = elements

    def __call__(self, molecule: "Molecule") -> torch.Tensor:
        """A function which should generate the relevant feature tensor for the
        molecule.
        """

        return torch.vstack(
            [
                one_hot_encode(atom.element.symbol, self.elements)
                for atom in molecule.atoms
            ]
        )

    def __len__(self):
        return len(self.elements)


class AtomConnectivity(AtomFeature):
    """One-hot encodes the connectivity (i.e. the number of bonds) of each atom in a
    molecule.
    """

    _CONNECTIVITIES = [1, 2, 3, 4]

    def __init__(self, connectivities: Optional[List[int]] = None):
        """
        Parameters
        ----------
        connectivities
            The connectivities (i.e. number of bonds to an atom) to include in the
            one-hot encoding in the order in which they should be encoded. If none are
            provided, the default set of [1, 2, 3, 4] will be used.
        """
        self.connectivities = (
            connectivities if connectivities is not None else [*self._CONNECTIVITIES]
        )

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [
                one_hot_encode(len(atom.bonds), self.connectivities)
                for atom in molecule.atoms
            ]
        )

    def __len__(self):
        return len(self.connectivities)


class AtomIsAromatic(AtomFeature):
    """One-hot encodes whether each atom in a molecule is aromatic."""

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [one_hot_encode(int(atom.is_aromatic), [0, 1]) for atom in molecule.atoms]
        )

    def __len__(self):
        return 2


class AtomIsInRing(AtomFeature):
    """One-hot encodes whether each atom in a molecule is in a ring of any size."""

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [one_hot_encode(int(atom.is_in_ring), [0, 1]) for atom in molecule.atoms]
        )

    def __len__(self):
        return 2


class AtomFormalCharge(AtomFeature):
    """One-hot encodes the formal charge on each atom in a molecule."""

    _CHARGES = [-3, -2, -1, 0, 1, 2, 3]

    def __init__(self, charges: Optional[List[int]] = None):
        """
        Parameters
        ----------
        charges
            The charges to include in the one-hot encoding in the order in which they
            should be encoded. If none are provided, the default set of
            [-3, -2, -1, 0, 1, 2, 3] will be used.
        """
        self.charges = charges if charges is not None else [*self._CHARGES]

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        from simtk import unit

        return torch.vstack(
            [
                one_hot_encode(
                    atom.formal_charge.value_in_unit(unit.elementary_charges),
                    self.charges,
                )
                for atom in molecule.atoms
            ]
        )

    def __len__(self):
        return len(self.charges)


class AtomFeaturizer(_Featurizer[AtomFeature]):
    """A class for featurizing the atoms in a molecule."""


class BondFeature(_Feature, abc.ABC):
    """The base class for bond features."""


class BondIsAromatic(BondFeature):
    """One-hot encodes whether each bond in a molecule is aromatic."""

    @classmethod
    def __call__(cls, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [one_hot_encode(int(bond.is_aromatic), [0, 1]) for bond in molecule.bonds]
        )

    def __len__(self):
        return 2


class BondIsInRing(BondFeature):
    """One-hot encodes whether each bond in a molecule is in a ring of any size."""

    def __call__(self, molecule: "Molecule") -> torch.Tensor:

        return torch.vstack(
            [one_hot_encode(int(bond.is_in_ring), [0, 1]) for bond in molecule.bonds]
        )

    def __len__(self):
        return 2


class WibergBondOrder(BondFeature):
    """Encodes the fractional Wiberg bond order of all of the bonds in a molecule."""

    @classmethod
    def __call__(cls, molecule: "Molecule") -> torch.Tensor:
        return torch.tensor(
            [bond.fractional_bond_order for bond in molecule.bonds]
        ).reshape(-1, 1)

    def __len__(self):
        return 1


class BondOrder(BondFeature):
    """Encodes the formal bond order of all of the bonds in a molecule."""

    @classmethod
    def __call__(cls, molecule: "Molecule") -> torch.Tensor:
        return (
            torch.tensor([bond.bond_order for bond in molecule.bonds])
            .reshape(-1, 1)
            .float()
        )

    def __len__(self):
        return 1


class BondFeaturizer(_Featurizer[BondFeature]):
    """A class for featurizing the bonds in a molecule."""
