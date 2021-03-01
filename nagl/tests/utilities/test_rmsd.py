import copy

import numpy
import pytest
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import GLOBAL_TOOLKIT_REGISTRY
from simtk import unit

from nagl.utilities.rmsd import (
    are_conformers_identical,
    compute_best_rmsd,
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


@pytest.mark.parametrize("scale", [True, False])
def test_compute_rmsd(scale):

    molecule: Molecule = Molecule.from_smiles("C")
    molecule.generate_conformers(n_conformers=1)

    # Generate a random rotation matrix.
    conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)
    conformer_b = perturb_conformer(conformer_a, scale)

    rmsd = compute_rmsd(conformer_a, conformer_b)
    rdkit_rmsd = rdkit_compute_best_rmsd(molecule, conformer_a, conformer_b)

    assert numpy.isclose(rmsd, 0.0) == (not scale)
    assert numpy.isclose(rmsd, rdkit_rmsd, atol=1.0e-4)


@pytest.mark.parametrize("scale", [True, False])
def test_compute_best_rmsd(scale):

    molecule: Molecule = Molecule.from_smiles("CO")
    molecule.generate_conformers(n_conformers=1)

    # Generate a random rotation matrix.
    conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)
    conformer_b = perturb_conformer(conformer_a.copy(), scale)

    # Shuffle the second conformer
    value = conformer_b[3, :].copy()

    conformer_b[3, :] = conformer_b[2, :]
    conformer_b[2, :] = value

    rmsd = compute_best_rmsd(molecule, conformer_a, conformer_b)
    rdkit_rmsd = rdkit_compute_best_rmsd(molecule, conformer_a, conformer_b)

    assert numpy.isclose(rmsd, 0.0) == (not scale)
    assert numpy.isclose(rmsd, rdkit_rmsd, atol=1.0e-4)


@pytest.mark.parametrize("smiles", ["c1ccc(cc1)c2ccccc2", "CCC"])
def test_are_conformers_identical(smiles):

    molecule: Molecule = Molecule.from_smiles(smiles)
    molecule.generate_conformers(n_conformers=1)

    # Create a permuted version of the conformer, permuting only topology symmetric
    # atoms.
    indexed_smiles = molecule.to_smiles(isomeric=False, mapped=True)

    matches = GLOBAL_TOOLKIT_REGISTRY.call(
        "find_smarts_matches", molecule, indexed_smiles
    )
    permuted_indices = next(
        iter(match for match in matches if match != tuple(range(len(matches))))
    )

    conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)
    conformer_b = perturb_conformer(conformer_a.copy(), False)[permuted_indices, :]

    assert are_conformers_identical(molecule, conformer_a, conformer_b)
    assert not are_conformers_identical(molecule, conformer_a, conformer_b * 2.0)


def test_are_conformers_identical_linear():
    """Tests that ``are_conformers_identical`` performs correctly when all of the
    heavy atoms are positioned as a linear chain."""

    molecule: Molecule = Molecule.from_smiles("CCC")
    molecule._conformers = [
        numpy.vstack(
            [
                numpy.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                numpy.random.random((molecule.n_atoms - 3, 3)),
            ]
        )
        * unit.angstrom
    ]

    # Create a permuted version of the conformer, permuting only topology symmetric
    # atoms.
    indexed_smiles = molecule.to_smiles(isomeric=False, mapped=True)

    matches = GLOBAL_TOOLKIT_REGISTRY.call(
        "find_smarts_matches", molecule, indexed_smiles
    )
    permuted_indices = next(
        iter(match for match in matches if match != tuple(range(len(matches))))
    )

    conformer_a = molecule.conformers[0].value_in_unit(unit.angstrom)
    conformer_b = perturb_conformer(conformer_a.copy(), False)[permuted_indices, :]

    assert are_conformers_identical(molecule, conformer_a, conformer_b)
    assert not are_conformers_identical(molecule, conformer_a, conformer_b * 2.0)
