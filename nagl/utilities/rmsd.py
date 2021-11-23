"""A module containing utilities for computing the RMSD between a molecule
in different conformers.
"""
import itertools
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy
from openff.utilities import requires_package

from nagl.utilities.toolkits import get_atom_symmetries

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


def align_conformers(
    conformer_a: numpy.ndarray,
    conformer_b: numpy.ndarray,
    subset_indices_a: Optional[List[int]] = None,
    subset_indices_b: Optional[List[int]] = None,
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """A method align two conformers using the Kabsch algorithm.

    Args:
        conformer_a: The first conformer with shape=(n_atoms, 3).
        conformer_b: The second conformer with shape=(n_atoms, 3).
        subset_indices_a: The (optional) subset of atom indices in the first conformer to
            attempt to align using.
        subset_indices_b: The (optional) subset of atom indices in the second conformer
            to attempt to align using.
    Returns:
        The aligned conformers.
    """

    assert (subset_indices_a is None and subset_indices_b is None) or (
        subset_indices_a is not None and subset_indices_b is not None
    ), "either both or neither of the basis indices arguments need to be provided"

    subset_indices_a = (
        subset_indices_a
        if subset_indices_a is not None
        else list(range(conformer_a.shape[0]))
    )
    subset_indices_b = (
        subset_indices_b
        if subset_indices_b is not None
        else list(range(conformer_b.shape[0]))
    )

    conformer_a = conformer_a - conformer_a[subset_indices_a, :].mean(axis=0)
    conformer_b = conformer_b - conformer_b[subset_indices_b, :].mean(axis=0)

    # From https://en.wikipedia.org/wiki/Kabsch_algorithm
    h = numpy.dot(conformer_a[subset_indices_a, :].T, conformer_b[subset_indices_b, :])

    u, s, v_transpose = numpy.linalg.svd(h)

    v_u_transpose = v_transpose.T @ u.T

    d = numpy.sign(numpy.linalg.det(v_u_transpose))

    rotation_matrix = v_transpose.T @ numpy.diag([1.0, 1.0, d]) @ u.T

    return conformer_a, conformer_b @ rotation_matrix


def compute_rmsd(
    conformer_a: numpy.ndarray,
    conformer_b: numpy.ndarray,
) -> float:
    """A method which will compute the RMSD between two conformers

    Args:
        conformer_a: The first conformer with shape=(n_atoms, 3).
        conformer_b: The second conformer with shape=(n_atoms, 3).

    Returns:
        The computed RMSD.
    """

    delta = numpy.array(conformer_a) - numpy.array(conformer_b)
    rmsd = numpy.sqrt((delta * delta).sum() / len(conformer_b))

    return rmsd


def _is_conformer_linear(conformer: numpy.ndarray) -> bool:
    """Checks if a conformer is linear"""

    if len(conformer) < 3:
        return True

    v1 = conformer[-1, :] - conformer[0, :]
    v1 = v1 / numpy.linalg.norm(v1)

    for i in range(1, len(conformer) - 1):

        v2 = conformer[i, :] - conformer[0, :]
        v2 = v2 / numpy.linalg.norm(v2)

        d = numpy.dot(v1, v2)

        if not numpy.isclose(numpy.abs(d), 1.0):
            return False

    return True


def _find_alignment_atoms(
    atom_symmetries: List[int], conformer: numpy.ndarray
) -> List[List[int]]:

    # Figure out which atoms are topologically symmetric.
    atoms_per_symmetry_group: Dict[int, List[int]] = defaultdict(list)

    for i, symmetry_group in enumerate(atom_symmetries):
        atoms_per_symmetry_group[symmetry_group].append(i)

    symmetry_groups_by_length = sorted(
        atoms_per_symmetry_group, key=lambda i: len(atoms_per_symmetry_group[i])
    )

    n_required_atoms = min(2 if _is_conformer_linear(conformer) else 3, len(conformer))

    # Attempt to select 3 (2 in the case of a linear conformer) atoms that are not
    # linear to align the molecule using.
    unselected_atoms = {
        group: [*indices] for group, indices in atoms_per_symmetry_group.items()
    }
    selected_atoms = []

    while (
        len(selected_atoms) != n_required_atoms and len(symmetry_groups_by_length) > 0
    ):

        proposed_atom = unselected_atoms[symmetry_groups_by_length[0]].pop(0)

        if len(unselected_atoms[symmetry_groups_by_length[0]]) == 0:
            symmetry_groups_by_length.pop(0)

        if len(selected_atoms) == 2 and _is_conformer_linear(
            conformer[[*selected_atoms, proposed_atom]]
        ):
            continue

        selected_atoms.append(proposed_atom)

    assert len(selected_atoms) == n_required_atoms, (
        f"a basis of {n_required_atoms} could not be formed "
        f"when comparing if two conformers are the same"
    )

    # Return all of the 'equivalent' atoms that can be aligned using.
    return [atoms_per_symmetry_group[atom_symmetries[i]] for i in selected_atoms]


@requires_package("openff.toolkit")
def are_conformers_identical(
    molecule: "Molecule",
    conformer_a: numpy.ndarray,
    conformer_b: numpy.ndarray,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-3,
) -> bool:
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

    conformer_a = conformer_a - conformer_a.mean(axis=0)
    conformer_b = conformer_b - conformer_b.mean(axis=0)

    is_linear_a = _is_conformer_linear(conformer_a)
    is_linear_b = _is_conformer_linear(conformer_b)

    if (is_linear_a and not is_linear_b) or (not is_linear_a and is_linear_b):
        return False

    # We need to try and find 3 atoms (2 in the case of linear molecules) to align the
    # two conformers using. If any of those 3 atoms have symmetrically equivalent atoms,
    # we need to try and align the conformers using these in place of the selected atoms
    # to try and account for automorphism.
    atom_symmetries = get_atom_symmetries(molecule)

    atoms_per_symmetry_group: Dict[int, List[int]] = defaultdict(list)

    for i, symmetry_group in enumerate(atom_symmetries):
        atoms_per_symmetry_group[symmetry_group].append(i)

    alignment_atom_indices = _find_alignment_atoms(atom_symmetries, conformer_a)

    subset_indices_a = next(
        iter(
            indices
            for indices in itertools.product(*alignment_atom_indices)
            if len(indices) == len({*indices})
        )
    )

    for subset_indices_b in itertools.product(*alignment_atom_indices):

        if len(subset_indices_b) != len({*subset_indices_b}):
            continue

        aligned_conformer_a, aligned_conformer_b = align_conformers(
            conformer_a, conformer_b, subset_indices_a, subset_indices_b
        )

        # Skip conformers whose aligned atoms don't match perfectly.
        aligned_rmsd = compute_rmsd(
            aligned_conformer_a[subset_indices_a, :],
            aligned_conformer_b[subset_indices_b, :],
        )

        if not numpy.isclose(aligned_rmsd, 0.0, rtol=rtol, atol=atol):
            continue

        distance_matrix = numpy.linalg.norm(
            aligned_conformer_a[:, None, :] - aligned_conformer_b[None, :, :], axis=-1
        )

        # The two conformers will be identical if there is a zero in every row. For
        # all physical conformers there should *only* be one zero per row.
        if not numpy.allclose(
            numpy.min(distance_matrix, axis=-1), 0.0, rtol=rtol, atol=atol
        ):
            continue

        minimum_distance_indices = numpy.argmin(distance_matrix, axis=-1)

        # Perform a last sanity check that the atoms that are on top of each other have
        # the same symmetry group.
        symmetry_groups_match = all(
            atom_symmetries[i] == atom_symmetries[index]
            for i, index in enumerate(minimum_distance_indices)
        )

        if symmetry_groups_match:
            return True

    return False
