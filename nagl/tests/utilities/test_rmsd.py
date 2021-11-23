import copy

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY
from simtk import unit

from nagl.utilities.rmsd import (
    _is_conformer_linear,
    align_conformers,
    are_conformers_identical,
    compute_rmsd,
)


def rdkit_compute_best_rmsd(
    molecule: Molecule, conformer_a: numpy.ndarray, conformer_b: numpy.ndarray
) -> float:

    from rdkit.Chem import AllChem
    from simtk import unit

    molecule_a = copy.deepcopy(molecule)
    molecule_a._conformers = [conformer_a * unit.angstrom]

    molecule_b = copy.deepcopy(molecule)
    molecule_b._conformers = [conformer_b * unit.angstrom]

    rdkit_molecule_a = molecule_a.to_rdkit()
    rdkit_molecule_b = molecule_b.to_rdkit()

    id_a = next(iter(conf.GetId() for conf in rdkit_molecule_a.GetConformers()))
    id_b = next(iter(conf.GetId() for conf in rdkit_molecule_b.GetConformers()))

    return AllChem.GetBestRMS(rdkit_molecule_a, rdkit_molecule_b, id_a, id_b)


def perturb_conformer(conformer: numpy.ndarray, scale: bool):

    theta = numpy.random.random()
    cos_theta, sin_theta = numpy.cos(theta), numpy.sin(theta)

    rotation = numpy.array(
        ((cos_theta, 0.0, -sin_theta), (0.0, 1.0, 0.0), (sin_theta, 0.0, cos_theta))
    )

    scale_factor = 1.0 if not scale else 2.0

    return (
        (conformer - numpy.mean(conformer, axis=0)) * scale_factor
    ) @ rotation + numpy.random.random()


def test_align_conformers():

    # molecule: Molecule = Molecule.from_mapped_smiles("[O:2]=[C:1]([H:3])[H:4]")

    conformer_a = numpy.array(
        [
            #          H4
            #         /
            # H3 - C1
            #      ||
            #      O2
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    conformer_b = numpy.array(
        [
            #          H3
            #         /
            # H4 - C1
            #      ||
            #      O2
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    # Rotate and shift the conformers to ensure we aren't just getting lucky.
    conformer_a = perturb_conformer(conformer_a, scale=False)
    conformer_b = perturb_conformer(conformer_b, scale=False)

    # Check passing the subset indices that will lead to poor alignment.
    conformer_a_aligned, conformer_b_aligned = align_conformers(
        conformer_a, conformer_b, subset_indices_a=[0, 1, 2], subset_indices_b=[0, 1, 2]
    )

    rmsd = compute_rmsd(conformer_a_aligned, conformer_b_aligned[[0, 1, 3, 2], :])
    assert not numpy.isclose(rmsd, 0.0)

    # Check passing the subset indices that will lead to perfect alignment.
    conformer_a_aligned, conformer_b_aligned = align_conformers(
        conformer_a, conformer_b, subset_indices_a=[0, 1, 2], subset_indices_b=[0, 1, 3]
    )

    rmsd = compute_rmsd(conformer_a_aligned, conformer_b_aligned[[0, 1, 3, 2], :])
    assert numpy.isclose(rmsd, 0.0)


@pytest.mark.parametrize("scale", [True, False])
def test_compute_rmsd(scale):

    molecule: Molecule = Molecule.from_smiles("C")
    molecule.generate_conformers(n_conformers=1)

    # Generate a random rotation matrix.
    conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)
    conformer_b = perturb_conformer(conformer_a, scale=scale)

    conformer_a, conformer_b = align_conformers(conformer_a, conformer_b)

    rmsd = compute_rmsd(conformer_a, conformer_b)
    rdkit_rmsd = rdkit_compute_best_rmsd(molecule, conformer_a, conformer_b)

    assert numpy.isclose(rmsd, 0.0) == (not scale)
    assert numpy.isclose(rmsd, rdkit_rmsd, atol=1.0e-4)


@pytest.mark.parametrize(
    "conformer, is_linear",
    [
        (numpy.array([[0.0, 0.0, 0.0]]), True),
        (numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), True),
        (numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]), True),
        (numpy.array([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-2.0, -2.0, -2.0]]), True),
        (numpy.array([[float(i)] * 3 for i in range(5)]), True),
        (numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]), False),
    ],
)
def test_is_conformer_linear(conformer, is_linear):
    assert _is_conformer_linear(conformer) == is_linear


@pytest.mark.parametrize(
    "smiles, conformer_a",
    [
        ("c1ccc(cc1)c2ccccc2", None),
        ("c1ccccc1", None),
        ("O=C(N)N", None),
        ("CCC", None),
        (
            "CCC",
            numpy.vstack(
                [
                    numpy.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                    numpy.random.random((8, 3)),
                ]
            ),
        ),
    ],
)
def test_are_conformers_identical(smiles, conformer_a):

    molecule: Molecule = Molecule.from_smiles(smiles)

    if conformer_a is None:

        molecule.generate_conformers(n_conformers=1)
        conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)

    # Create a permuted version of the conformer, permuting only topology symmetric
    # atoms.
    indexed_smiles = molecule.to_smiles(isomeric=False, mapped=True)

    matches = GLOBAL_TOOLKIT_REGISTRY.call(
        "find_smarts_matches", molecule, indexed_smiles
    )
    permuted_indices = next(
        iter(match for match in matches if match != tuple(range(len(match))))
    )

    conformer_b = perturb_conformer(conformer_a.copy(), False)[permuted_indices, :]

    assert are_conformers_identical(molecule, conformer_a, conformer_b)
    assert not are_conformers_identical(molecule, conformer_a, conformer_b * 2.0)


def test_are_conformers_not_identical():

    molecule: Molecule = Molecule.from_mapped_smiles(
        "[C:1]([H:4])([H:5])([H:6])[C:2]([H:7])([Cl:8])=[O:3]"
    )
    molecule.generate_conformers(n_conformers=1)

    conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)

    # Swap and perturb the hydrogen positions.
    hydrogen_coordinates = conformer_a[3, :]

    conformer_b = conformer_a.copy()
    conformer_b[3, :] = conformer_b[4, :]
    conformer_b[4, :] = hydrogen_coordinates + 0.1

    assert not are_conformers_identical(molecule, conformer_a, conformer_b)
