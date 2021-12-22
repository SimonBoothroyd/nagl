import functools
import logging
from multiprocessing import Pool
from typing import TYPE_CHECKING, Tuple

import click
from click_option_group import optgroup
from openff.utilities import requires_package
from tqdm import tqdm

from nagl.utilities.toolkits import (
    capture_toolkit_warnings,
    stream_from_file,
    stream_to_file,
)

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

_logger = logging.getLogger(__name__)


def apply_filter(molecule: "Molecule", retain_largest: bool) -> Tuple["Molecule", bool]:

    with capture_toolkit_warnings():

        try:
            from openff.toolkit.topology import Molecule
            from simtk import unit as simtk_unit

            split_smiles = molecule.to_smiles().split(".")
            n_sub_molecules = len(split_smiles)

            if retain_largest and n_sub_molecules > 1:

                largest_smiles = max(split_smiles, key=len)
                molecule = Molecule.from_smiles(largest_smiles, allow_undefined_stereo=True)

            # Retain H, C, N, O, F, P, S, Cl, Br, I
            allowed_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

            mass = sum(
                atom.mass.value_in_unit(simtk_unit.gram / simtk_unit.mole)
                for atom in molecule.atoms
            )

            return (
                molecule,
                (
                    all(atom.atomic_number in allowed_elements for atom in molecule.atoms)
                    and (250.0 < mass < 350.0)
                    and (len(molecule.find_rotatable_bonds()) <= 7)
                ),
            )

        except BaseException:  # lgtm [py/catch-base-exception]
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
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The path to save the filtered molecules to. This should either be an SDF or "
    "a GZipped SDF file.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
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
@requires_package("openff.toolkit")
def filter_cli(
    input_path: str,
    output_path: str,
    n_processes: int,
    strip_ions: bool,
):

    print(" - Filtering molecules")

    with capture_toolkit_warnings():
        with stream_to_file(output_path) as writer:

            with Pool(processes=n_processes) as pool:

                for molecule, should_include in tqdm(
                    pool.imap(
                        functools.partial(apply_filter, retain_largest=strip_ions),
                        stream_from_file(input_path),
                    ),
                ):

                    if not should_include:
                        continue

                    writer(molecule)
