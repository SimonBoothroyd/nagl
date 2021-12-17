import functools
from multiprocessing import Pool
from typing import Set

import click
from click_option_group import optgroup
from openff.utilities import requires_package
from tqdm import tqdm

from nagl.utilities.toolkits import (
    capture_toolkit_warnings,
    stream_from_file,
    stream_to_file,
)


@requires_package("openff.toolkit")
def _enumerate_tautomers(
    smiles: str,
    enumerate_tautomers: bool,
    max_tautomers: int,
    enumerate_protomers: bool,
    max_protomers: int,
) -> Set[str]:

    found_forms = {smiles}

    with capture_toolkit_warnings():

        from openff.toolkit.topology import Molecule
        from openff.toolkit.utils import (
            OpenEyeToolkitWrapper,
            RDKitToolkitWrapper,
            ToolkitRegistry,
        )

        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        if enumerate_tautomers:

            toolkit_registry = ToolkitRegistry(
                toolkit_precedence=[RDKitToolkitWrapper, OpenEyeToolkitWrapper],
                exception_if_unavailable=False,
            )

            found_forms.update(
                tautomer.to_smiles()
                for tautomer in molecule.enumerate_tautomers(
                    max_states=max_tautomers, toolkit_registry=toolkit_registry
                )
            )

        if enumerate_protomers:  # pragma: no cover

            from openeye import oechem, oequacpac

            oe_molecule: oechem.OEMol = molecule.to_openeye()

            for i, oe_protomer in enumerate(
                oequacpac.OEGetReasonableProtomers(oe_molecule)
            ):
                found_forms.add(oechem.OEMolToSmiles(oe_protomer))

                if i >= max_protomers:
                    break

    return found_forms


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
    "--tautomers/--no-tautomers",
    "enumerate_tautomers",
    help="Whether to enumerate possible tautomers or not.",
    default=False,
    show_default=True,
)
@click.option(
    "--max-tautomers",
    help="The maximum number of tautomers to generate per input molecule.",
    type=int,
    default=16,
    show_default=True,
)
@click.option(
    "--protomers/--no-protomers",
    "enumerate_protomers",
    help="Whether to enumerate the possible protontation states or not. "
    "(required oequacpac)",
    default=False,
    show_default=True,
)
@click.option(
    "--max-protomers",
    help="The maximum number of protontation states to generate per input molecule.",
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
    enumerate_tautomers: bool,
    max_tautomers: int,
    enumerate_protomers: bool,
    max_protomers: int,
    n_processes: int,
):

    print(
        f" - Enumerating"
        f"{' tautomers' if enumerate_tautomers else ''}"
        f"{'/' if enumerate_protomers and enumerate_tautomers else ''}"
        f"{' protomers' if enumerate_protomers else ''}"
    )

    unique_molecules = set()

    with capture_toolkit_warnings():
        with stream_to_file(output_path) as writer:

            with Pool(processes=n_processes) as pool:

                for smiles in tqdm(
                    pool.imap(
                        functools.partial(
                            _enumerate_tautomers,
                            enumerate_tautomers=enumerate_tautomers,
                            max_tautomers=max_tautomers,
                            enumerate_protomers=enumerate_protomers,
                            max_protomers=max_protomers,
                        ),
                        stream_from_file(input_path, as_smiles=True),
                    ),
                ):

                    for pattern in smiles:

                        from openff.toolkit.topology import Molecule

                        molecule: Molecule = Molecule.from_smiles(
                            pattern, allow_undefined_stereo=True
                        )

                        inchi_key = molecule.to_inchikey(fixed_hydrogens=True)

                        if inchi_key in unique_molecules:
                            continue

                        writer(molecule)
                        unique_molecules.add(inchi_key)
