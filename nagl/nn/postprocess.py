import abc
import typing

import torch.nn

from nagl.config.model import PostprocessType
from nagl.molecules import DGLMolecule, DGLMoleculeBatch


class PostprocessLayer(torch.nn.Module, abc.ABC):
    """A layer to apply to the final readout of a neural network."""

    @abc.abstractmethod
    def forward(
        self,
        molecule: typing.Union[DGLMolecule, DGLMoleculeBatch],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the post-processed input vector."""


class PartialChargeLayer(PostprocessLayer):
    """A layer which will map an NN readout containing a set of atomic electronegativity
    and hardness parameters to a set of partial charges [1].

    The layer expects two features per atom (i.e an input of (n_atoms, 2)) - the
    electronegativity and the hardness.

    References:
        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
            assignment of accurate partial atomic charges: an electronegativity
            equalization method that accounts for alternate resonance forms." Journal of
            chemical information and computer sciences 43.6 (2003): 1982-1997.
    """

    @classmethod
    def atomic_parameters_to_charges(
        cls,
        electronegativity: torch.Tensor,
        hardness: torch.Tensor,
        total_charge: float,
    ) -> torch.Tensor:
        """Converts a set of atomic electronegativity and hardness parameters to a
        set of partial atomic charges subject to a total charge constraint.

        Args:
            electronegativity: The electronegativity of atoms in a given molecule.
            hardness: The hardness of atoms in a given molecule.
            total_charge: The total charge on the molecule.

        Returns:
            The atomic partial charges.
        """

        inverse_hardness = 1.0 / hardness

        charges = (
            -inverse_hardness * electronegativity
            + inverse_hardness
            * torch.div(
                torch.dot(inverse_hardness, electronegativity) + total_charge,
                torch.sum(inverse_hardness),
            )
        ).reshape(-1, 1)

        return charges

    def forward(
        self,
        molecule: typing.Union[DGLMolecule, DGLMoleculeBatch],
        inputs: torch.Tensor,
    ) -> torch.Tensor:

        charges = []
        counter = 0

        graph = molecule.graph

        n_atoms_per_molecule = (
            (molecule.n_atoms,)
            if isinstance(molecule, DGLMolecule)
            else molecule.n_atoms_per_molecule
        )
        n_representations_per_molecule = (
            (molecule.n_representations,)
            if isinstance(molecule, DGLMolecule)
            else molecule.n_representations_per_molecule
        )

        for n_atoms, n_representations in zip(
            n_atoms_per_molecule, n_representations_per_molecule
        ):

            atom_slices = [
                slice(counter + i * n_atoms, counter + (i + 1) * n_atoms)
                for i in range(n_representations)
            ]

            mol_charges = torch.stack(
                [
                    self.atomic_parameters_to_charges(
                        inputs[atom_slice, 0],
                        inputs[atom_slice, 1],
                        graph.ndata["formal_charge"][atom_slice].sum(),
                    )
                    for atom_slice in atom_slices
                ]
            ).mean(dim=0)

            charges.append(mol_charges)
            counter += n_atoms * n_representations

        return torch.vstack(charges)


def get_postprocess_layer(type_: PostprocessType) -> typing.Type[PostprocessLayer]:

    if type_.lower() == "charges":
        return PartialChargeLayer

    raise NotImplementedError(f"{type_} not a supported postprocess layer type")
