import functools
import pathlib
import typing

import click
import rich.progress
from click_option_group import optgroup
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from nagl.utilities import get_map_func
from nagl.utilities.molecule import (
    molecule_from_smiles,
    stream_from_file,
    stream_to_file,
)


def _enumerate_tautomers(
    smiles: str,
    enumerate_tautomers: bool,
    max_tautomers: int,
) -> typing.Set[str]:

    # normalize the input SMILES
    smiles = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smiles)))
    found_forms = {smiles}

    molecule = molecule_from_smiles(smiles)

    if not enumerate_tautomers:
        return found_forms

    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(max_tautomers)

    tautomers = enumerator.Enumerate(Chem.RemoveHs(molecule))

    for tautomer in tautomers:
        found_forms.add(Chem.MolToSmiles(Chem.AddHs(tautomer)))

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
    type=click.Path(
        exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The path to save the enumerated molecules to. This should either be an SDF "
    "or a GZipped SDF file.",
    type=click.Path(
        exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
    ),
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
@optgroup.group("Parallelization configuration")
@optgroup.option(
    "--n-processes",
    help="The number of processes to parallelize the enumeration over.",
    type=int,
    default=0,
    show_default=True,
)
def enumerate_cli(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    enumerate_tautomers: bool,
    max_tautomers: int,
    n_processes: int,
):

    print(f" - Enumerating {' tautomers' if enumerate_tautomers else ''}")

    unique_molecules = set()

    enumerate_func = functools.partial(
        _enumerate_tautomers,
        enumerate_tautomers=enumerate_tautomers,
        max_tautomers=max_tautomers,
    )

    with get_map_func(n_processes) as map_func:

        enumerated_smiles = map_func(
            enumerate_func,
            stream_from_file(input_path, as_smiles=True),
        )

        with stream_to_file(output_path) as writer:

            for smiles in rich.progress.track(enumerated_smiles):
                for pattern in smiles:

                    molecule = molecule_from_smiles(pattern)

                    opts = Chem.SmilesWriteParams()
                    opts.allHsExplicit = True

                    pattern = Chem.MolToSmiles(molecule, opts)

                    if pattern in unique_molecules:
                        continue

                    writer(molecule)
                    unique_molecules.add(pattern)
