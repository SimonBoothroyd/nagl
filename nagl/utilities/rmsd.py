"""A module containing utilities for computing the RMSD between a molecule
in different conformers.
"""
from collections import defaultdict
from typing import TYPE_CHECKING, Tuple

import numpy
from openff.utilities import requires_package

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


def _compute_rmsd(
    conformer_a: numpy.ndarray, conformer_b: numpy.ndarray, align: bool = True
) -> Tuple[float, numpy.ndarray]:
    """A method which will compute the RMSD between two conformers

    Args:
        conformer_a: The first conformer with shape=(n_atoms, 3).
        conformer_b: The second conformer with shape=(n_atoms, 3).
        align: Whether to attempt to align the two conformers using the Kabsch algorithm

    Returns:
        The RMSD and the rotation matrix used to align the two conformers. The
        identity rotation matrix will always be returned if ``align`` is false.
    """

    conformer_a = conformer_a - conformer_a.mean(axis=0)
    conformer_b = conformer_b - conformer_b.mean(axis=0)

    if align:

        # From https://en.wikipedia.org/wiki/Kabsch_algorithm
        h = numpy.dot(conformer_a.T, conformer_b)

        u, s, v_transpose = numpy.linalg.svd(h)

        v_u_transpose = v_transpose.T @ u.T

        d = numpy.sign(numpy.linalg.det(v_u_transpose))

        rotation_matrix = v_transpose.T @ numpy.diag([1.0, 1.0, d]) @ u.T
        conformer_b = conformer_b @ rotation_matrix
    else:
        rotation_matrix = numpy.eye(3)

    delta = numpy.array(conformer_a) - numpy.array(conformer_b)
    rmsd = numpy.sqrt((delta * delta).sum() / len(conformer_b))

    return rmsd, rotation_matrix


def compute_rmsd(conformer_a: numpy.ndarray, conformer_b: numpy.ndarray) -> float:
    """A method which will compute the RMSD between two conformers after aligning them
    using the Kabsch algorithm.

    Args:
        conformer_a: The first conformer with shape=(n_atoms, 3).
        conformer_b: The second conformer with shape=(n_atoms, 3).

    Returns:
        The RMSD between the two conformers.
    """
    return _compute_rmsd(conformer_a, conformer_b, align=True)[0]


@requires_package("openff.toolkit")
def compute_best_rmsd(
    molecule: "Molecule", conformer_a: numpy.ndarray, conformer_b: numpy.ndarray
) -> float:
    """A method which attempts to compute the smallest RMSD between two conformers
    by considering all permutations of topologically symmetric atoms and aligning each
    permuted conformer using the Kabsch algorithm.

    Args:
        molecule: The molecule associated with the two conformers.
        conformer_a: The first conformer with shape=(n_atoms, 3).
        conformer_b: The second conformer with shape=(n_atoms, 3).

    Returns:
        The RMSD between the two conformers.
    """

    from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY

    # Find all of the different permutations of the molecule.
    indexed_smiles = molecule.to_smiles(isomeric=False, mapped=True)

    matches = set(
        GLOBAL_TOOLKIT_REGISTRY.call("find_smarts_matches", molecule, indexed_smiles)
    )
    assert len(matches) >= 1, "the SMILES should at minimum match itself"

    smallest_rmsd = None

    for match in matches:

        reordered_conformer_b = conformer_b[numpy.array(match)]
        rmsd = compute_rmsd(conformer_a, reordered_conformer_b)

        smallest_rmsd = rmsd if smallest_rmsd is None else min(smallest_rmsd, rmsd)

    return smallest_rmsd


@requires_package("openff.toolkit")
def are_conformers_identical(
    molecule: "Molecule",
    conformer_a: numpy.ndarray,
    conformer_b: numpy.ndarray,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-3,
) -> float:
    """A method which compares if two conformers are identical by computing the
    smallest RMSD between all permutations of topologically symmetric atoms.

    Args:
        molecule: The molecule associated with the two conformers.
        conformer_a: The first conformer with shape=(n_atoms, 3).
        conformer_b: The second conformer with shape=(n_atoms, 3).
        rtol: The relative tolerance to use when comparing RMSD values. See
            ``numpy.isclose`` for more information.
        atol: The absolute tolerance to use when comparing RMSD values. See
            ``numpy.isclose`` for more information.

    Returns:
        The RMSD between the two conformers.
    """

    from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY

    # Find all of the different permutations of the atoms in a molecule.
    indexed_smiles = molecule.to_smiles(isomeric=False, mapped=True)

    matches = set(
        GLOBAL_TOOLKIT_REGISTRY.call("find_smarts_matches", molecule, indexed_smiles)
    )
    assert len(matches) >= 1, "the SMILES should at minimum match itself"

    matches_by_heavy = defaultdict(list)

    for match in matches:

        heavy_match = tuple(i for i in match if molecule.atoms[i].atomic_number != 1)
        matches_by_heavy[heavy_match].append(match)

    conformer_a = conformer_a - conformer_a.mean(axis=0)
    conformer_b = conformer_b - conformer_b.mean(axis=0)

    heavy_conformer_a = conformer_a[
        [i for i in range(molecule.n_atoms) if molecule.atoms[i].atomic_number != 1], :
    ]

    for heavy_match, full_matches in matches_by_heavy.items():

        # See if the heavy atoms align first.
        heavy_rmsd, rotation_matrix = _compute_rmsd(
            heavy_conformer_a, conformer_b[heavy_match, :]
        )

        if not numpy.isclose(heavy_rmsd, 0.0, rtol=rtol, atol=atol):
            # If the heavy atoms don't align then including hydrogen, which may lead
            # to an explosion of different conformer permutations, won't change things.
            continue

        for match in full_matches:

            aligned_conformer = conformer_b @ rotation_matrix

            is_linear = True

            if len(heavy_match) >= 3:
                # Check for molecules where all heavy atoms are linear.
                # In these edge cases we will need to re-align the molecule
                # based on the hydrogen atoms.
                v1 = (
                    aligned_conformer[heavy_match[1], :]
                    - aligned_conformer[heavy_match[0], :]
                )
                v2 = (
                    aligned_conformer[heavy_match[2], :]
                    - aligned_conformer[heavy_match[0], :]
                )

                v1 = v1 / numpy.linalg.norm(v1)
                v2 = v2 / numpy.linalg.norm(v2)

                d = numpy.dot(v1, v2)

                is_linear = numpy.isclose(numpy.abs(d), 1.0)

            rmsd, _ = _compute_rmsd(
                conformer_a, aligned_conformer[match, :], align=is_linear
            )

            if numpy.isclose(rmsd, 0.0, rtol=rtol, atol=atol):
                return True

    return False
