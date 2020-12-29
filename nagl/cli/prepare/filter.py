import functools
from multiprocessing import Pool
from typing import TYPE_CHECKING, Tuple

import click
from click_option_group import optgroup
from tqdm import tqdm

from nagl.utilities.openeye import capture_oe_warnings, requires_oe_package

if TYPE_CHECKING:
    from openeye import oechem


@requires_oe_package("oechem")
@requires_oe_package("oemolprop")
def apply_filter(
    oe_molecule: "oechem.OEMol", retain_largest: bool
) -> Tuple["oechem.OEMol", bool]:

    from openeye import oechem, oemolprop

    with capture_oe_warnings():

        if retain_largest:
            oechem.OEDeleteEverythingExceptTheFirstLargestComponent(oe_molecule)

        # Retain H, C, N, O, F, P, S, Cl, Br, I
        allowed_elements = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

        return (
            oe_molecule,
            (
                all(
                    atom.GetAtomicNum() in allowed_elements
                    for atom in oe_molecule.GetAtoms()
                )
                and (250.0 < oechem.OECalculateMolecularWeight(oe_molecule) < 350.0)
                and (oemolprop.OEGetRotatableBondCount(oe_molecule) <= 7)
            ),
        )


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
@requires_oe_package("oechem")
@requires_oe_package("oemolprop")
def filter_cli(
    input_path: str,
    output_path: str,
    n_processes: int,
    strip_ions: bool,
):
    from openeye import oechem

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(input_path)

    print(" - Filtering molecules")

    output_molecule_stream = oechem.oemolostream(output_path)

    with capture_oe_warnings():

        with Pool(processes=n_processes) as pool:

            for oe_molecule, should_include in tqdm(
                pool.imap(
                    functools.partial(apply_filter, retain_largest=strip_ions),
                    input_molecule_stream.GetOEMols(),
                ),
            ):

                if not should_include:
                    continue

                oechem.OEWriteMolecule(output_molecule_stream, oe_molecule)
