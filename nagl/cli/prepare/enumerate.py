import functools
from multiprocessing import Pool
from typing import TYPE_CHECKING, List

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


@requires_package("openff.toolkit")
def enumerate_tautomers(molecule: "Molecule", max_tautomers: int) -> List["Molecule"]:

    with capture_toolkit_warnings():

        from openff.toolkit.utils import (
            OpenEyeToolkitWrapper,
            RDKitToolkitWrapper,
            ToolkitRegistry,
        )

        toolkit_registry = ToolkitRegistry(
            toolkit_precedence=[RDKitToolkitWrapper, OpenEyeToolkitWrapper],
            exception_if_unavailable=False,
        )

        return [
            molecule,
            *molecule.enumerate_tautomers(
                max_states=max_tautomers, toolkit_registry=toolkit_registry
            ),
        ]


@click.command(
    "enumerate",
    short_help="Enumerate all reasonable tautomers of a molecule set.",
    help="Enumerates all reasonable tautomers (as determine by the OpenEye toolkit) "
    "of a specified set of molecules.",
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
    help="The path to save the enumerated molecules to. This should either be an SDF or "
    "a GZipped SDF file.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--max-tautomers",
    help="The maximum number of tautomers to generate per input molecule.",
    type=int,
    default=16,
    show_default=True,
)
@optgroup.group("Parallelization configuration")
@optgroup.option(
    "--n-processes",
    help="The number of processes to parallelize the enumeration over.",
    type=int,
    default=1,
    show_default=True,
)
@requires_package("openff.toolkit")
def enumerate_cli(
    input_path: str,
    output_path: str,
    max_tautomers: int,
    n_processes: int,
):

    print(" - Enumerating tautomers")

    unique_molecules = set()

    with capture_toolkit_warnings():
        with stream_to_file(output_path) as writer:

            with Pool(processes=n_processes) as pool:

                for molecules in tqdm(
                    pool.imap(
                        functools.partial(
                            enumerate_tautomers, max_tautomers=max_tautomers
                        ),
                        stream_from_file(input_path),
                    ),
                ):

                    for molecule in molecules:

                        smiles = molecule.to_smiles()

                        if smiles in unique_molecules:
                            continue

                        writer(molecule)
                        unique_molecules.add(smiles)
