import functools
from multiprocessing import Pool

import click
from click_option_group import optgroup
from tqdm import tqdm

from nagl.utilities.openeye import (
    capture_oe_warnings,
    enumerate_tautomers,
    requires_oe_package,
)


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
@click.option(
    "--pka-normalize",
    help="Whether to set the ionization state of each tautomer to the predominate state "
    "at pH ~7.4.",
    type=bool,
    default=True,
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
@requires_oe_package("oechem")
def enumerate_cli(
    input_path: str,
    output_path: str,
    max_tautomers: int,
    pka_normalize: bool,
    n_processes: int,
):

    from openeye import oechem

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(input_path)

    print(" - Enumerating tautomers")

    output_molecule_stream = oechem.oemolostream(output_path)

    with capture_oe_warnings():

        with Pool(processes=n_processes) as pool:

            for oe_molecules in tqdm(
                pool.imap(
                    functools.partial(
                        enumerate_tautomers,
                        max_tautomers=max_tautomers,
                        pka_normalize=pka_normalize,
                    ),
                    input_molecule_stream.GetOEMols(),
                ),
            ):

                for oe_molecule in oe_molecules:

                    oechem.OEWriteMolecule(
                        output_molecule_stream, oechem.OEMol(oe_molecule)
                    )
