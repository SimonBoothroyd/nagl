import functools
import logging
import multiprocessing
import pathlib
import typing

import click
import rich.progress
from click_option_group import optgroup
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from nagl.utilities.molecule import (
    molecule_from_smiles,
    stream_from_file,
    stream_to_file,
)

_logger = logging.getLogger(__name__)


def apply_filter(
    molecule: Chem.Mol, retain_largest: bool
) -> typing.Tuple[Chem.Mol, bool]:

    try:

        split_smiles = Chem.MolToSmiles(molecule).split(".")
        n_sub_molecules = len(split_smiles)

        if retain_largest and n_sub_molecules > 1:

            largest_smiles = max(split_smiles, key=len)
            molecule = molecule_from_smiles(largest_smiles)

        # Retain H, C, N, O, F, P, S, Cl, Br, I
        allowed_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

        mass = sum(atom.GetMass() for atom in molecule.GetAtoms())

        return (
            molecule,
            (
                all(
                    atom.GetAtomicNum() in allowed_elements
                    for atom in molecule.GetAtoms()
                )
                and (250.0 < mass < 350.0)
                and (rdMolDescriptors.CalcNumRotatableBonds(molecule) <= 7)
            ),
        )

    except BaseException:
        _logger.exception("failed to apply filter")
        return molecule, False


@click.command(
    "filter",
    short_help="Filter undesirable chemistries and counter-ions.",
    help="Filters a set of molecules based on the criteria specified by:\n\n"
    "    [1] Bleiziffer, Patrick, Kay Schaller, and Sereina Riniker. 'Machine learning "
    "of partial charges derived from high-quality quantum-mechanical calculations.' "
    "JCIM 58.3 (2018): 579-590.\n\nIn particular molecules are only retained if they "
    "have a weight between 250 and 350 g/mol, have less than seven rotatable bonds and "
    "are composed of only H, C, N, O, F, P, S, Cl, Br, and I.\n\nThis script will also "
    "optionally remove any counter-ions by retaining only the largest molecule if "
    "multiple components are present.",
)
@click.option(
    "--input",
    "input_path",
    help="The path to the input molecules. This should either be an SDF or a GZipped "
    "SDF file.",
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The path to save the filtered molecules to. This should either be an SDF or "
    "a GZipped SDF file.",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
)
@click.option(
    "--strip-ions",
    "strip_ions",
    is_flag=True,
    help="If specified counter ions (and molecules) will be removed.",
    default=False,
    show_default=True,
)
@optgroup.group("Parallelization configuration")
@optgroup.option(
    "--n-processes",
    help="The number of processes to parallelize the filtering over.",
    type=int,
    default=1,
    show_default=True,
)
def filter_cli(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    n_processes: int,
    strip_ions: bool,
):

    print(" - Filtering molecules")

    with stream_to_file(output_path) as writer:

        with multiprocessing.Pool(processes=n_processes) as pool:

            for molecule, should_include in rich.progress.track(
                pool.imap(
                    functools.partial(apply_filter, retain_largest=strip_ions),
                    stream_from_file(input_path),
                ),
            ):

                if not should_include:
                    continue

                writer(molecule)
