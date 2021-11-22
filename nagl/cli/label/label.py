import functools
import math
import traceback
from datetime import datetime
from typing import List, Optional, Tuple

import click
from click_option_group import optgroup
from openff.utilities import requires_package
from tqdm import tqdm

from nagl.storage.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
    WibergBondOrderSet,
)
from nagl.utilities.dask import setup_dask_local_cluster, setup_dask_lsf_cluster
from nagl.utilities.provenance import get_labelling_software_provenance
from nagl.utilities.smiles import smiles_to_molecule
from nagl.utilities.toolkits import capture_toolkit_warnings, stream_from_file

_OPENFF_CHARGE_METHODS = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}


@requires_package("openff.toolkit")
def _label_molecule(smiles: str, guess_stereochemistry: bool) -> MoleculeRecord:

    from simtk import unit

    molecule = smiles_to_molecule(smiles, guess_stereochemistry=guess_stereochemistry)

    # Generate a diverse set of ELF10 conformers
    molecule.generate_conformers(n_conformers=500, rms_cutoff=0.05 * unit.angstrom)
    molecule.apply_elf_conformer_selection()

    conformer_records = []

    for conformer in molecule.conformers:

        charge_sets = []

        # Compute partial charges.
        for charge_method in ["am1", "am1bcc"]:
            molecule.assign_partial_charges(
                _OPENFF_CHARGE_METHODS[charge_method], use_conformers=[conformer]
            )

            charge_sets.append(
                PartialChargeSet(
                    method=charge_method,
                    values=[
                        atom.partial_charge.value_in_unit(unit.elementary_charge)
                        for atom in molecule.atoms
                    ],
                )
            )

        # Compute WBOs.
        molecule.assign_fractional_bond_orders("am1-wiberg", use_conformers=[conformer])

        conformer_records.append(
            ConformerRecord(
                coordinates=conformer.value_in_unit(unit.angstrom),
                partial_charges=charge_sets,
                bond_orders=[
                    WibergBondOrderSet(
                        method="am1",
                        values=[
                            (
                                bond.atom1_index,
                                bond.atom2_index,
                                bond.fractional_bond_order,
                            )
                            for bond in molecule.bonds
                        ],
                    )
                ],
            )
        )

    return MoleculeRecord(
        smiles=molecule.to_smiles(isomeric=False, mapped=True),
        conformers=conformer_records,
    )


@requires_package("openff.toolkit")
def label_molecule_batch(
    smiles: List[str], guess_stereochemistry: bool
) -> List[Tuple[Optional[MoleculeRecord], Optional[str]]]:
    """Labels a batch of molecules using ``compute_am1_charge_and_wbo``.

    Returns
    -------
        A list of tuples. Each tuple will contain the processed molecule containing the
        AM1 charges and WBO if no exceptions were raised (``None`` otherwise) and the
        error string if an exception was raised (``None`` otherwise).
    """

    molecule_records = []

    with capture_toolkit_warnings():

        for pattern in tqdm(smiles):

            molecule_record = None
            error = None

            try:
                molecule_record = _label_molecule(pattern, guess_stereochemistry)
            except (BaseException, Exception) as e:

                formatted_traceback = traceback.format_exception(
                    etype=type(e), value=e, tb=e.__traceback__
                )
                error = f"Failed to process {pattern}: {formatted_traceback}"

            molecule_records.append((molecule_record, error))

    return molecule_records


@click.command(
    "label",
    short_help="Label molecules with AM1 charges and WBOs",
    help="Compute the AM1 partial charges and AM1 Wiberg bond orders (WBOs) for each "
    "molecule in a given set and store each labelled molecule as a pickled OpenFF "
    "``Molecule`` object.",
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
    help="The path to the SQLite database (.sqlite) to save the labelled molecules in.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
    required=True,
)
@click.option(
    "--guess-stereo",
    help="Whether to select a random stereoisomer for molecules with undefined "
    "stereochemistry.",
    type=bool,
    default=True,
    show_default=True,
)
@optgroup.group("Parallelization configuration")
@optgroup.option(
    "--n-workers",
    help="The number of workers to distribute the labelling across. Use -1 to request "
    "one worker per batch.",
    type=int,
    default=1,
    show_default=True,
)
@optgroup.option(
    "--worker-type",
    help="The type of worker to distribute the labelling across.",
    type=click.Choice(["lsf", "local"]),
    default="local",
    show_default=True,
)
@optgroup.option(
    "--batch-size",
    help="The number of molecules to processes at once on a particular worker.",
    type=int,
    default=500,
    show_default=True,
)
@optgroup.group("LSF configuration", help="Options to configure LSF workers.")
@optgroup.option(
    "--lsf-memory",
    help="The amount of memory (GB) to request per LSF queue worker.",
    type=int,
    default=3,
    show_default=True,
)
@optgroup.option(
    "--lsf-walltime",
    help="The maximum wall-clock time to request per LSF queue worker.",
    type=str,
    default="02:00",
    show_default=True,
)
@optgroup.option(
    "--lsf-queue",
    help="The LSF queue to submit workers to.",
    type=str,
    default="default",
    show_default=True,
)
@optgroup.option(
    "--lsf-env",
    help="The conda environment that LSF workers should run using.",
    type=str,
)
@requires_package("dask.distributed")
@requires_package("openff.toolkit")
def label_cli(
    input_path: str,
    output_path: str,
    guess_stereo: bool,
    worker_type: str,
    n_workers: int,
    batch_size: int,
    lsf_memory: int,
    lsf_walltime: str,
    lsf_queue: str,
    lsf_env: str,
):

    from dask import distributed

    print(" - Labeling molecules")

    with capture_toolkit_warnings():

        all_smiles = [
            smiles
            for smiles in tqdm(
                stream_from_file(input_path, as_smiles=True),
                desc="loading molecules",
                ncols=80,
            )
        ]

    unique_smiles = sorted({*all_smiles})

    if len(unique_smiles) != len(all_smiles):

        print(
            f"\n    [WARNING] {len(all_smiles) - len(unique_smiles)} duplicate "
            f"molecules were ignored"
        )

    n_batches = int(math.ceil(len(all_smiles) / batch_size))

    if n_workers < 0:
        n_workers = n_batches

    if n_workers > n_batches:
        print(
            f"\n    [WARNING] More workers were requested then there are batches to "
            f"compute. Only {n_batches} workers will be requested.\n"
        )

        n_workers = n_batches

    # Set-up dask to distribute the processing.
    if worker_type == "lsf":
        dask_cluster = setup_dask_lsf_cluster(
            n_workers, lsf_queue, lsf_memory, lsf_walltime, lsf_env
        )
    elif worker_type == "local":
        dask_cluster = setup_dask_local_cluster(n_workers)
    else:
        raise NotImplementedError()

    print(
        f"   * {len(unique_smiles)} molecules will labelled in {n_batches} batches "
        f"across {n_workers} workers\n"
    )

    dask_client = distributed.Client(dask_cluster)

    # Submit the tasks to be computed in chunked batches.
    def batch(iterable):
        n_iterables = len(iterable)

        for i in range(0, n_iterables, batch_size):
            yield iterable[i : min(i + batch_size, n_iterables)]

    futures = [
        dask_client.submit(
            functools.partial(label_molecule_batch, guess_stereochemistry=guess_stereo),
            batched_molecules,
        )
        for batched_molecules in batch(unique_smiles)
    ]

    # Create a database to store the labelled molecules in and store general
    # provenance information.
    storage = MoleculeStore(output_path)

    storage.set_provenance(
        general_provenance={
            "date": datetime.now().strftime("%d-%m-%Y"),
        },
        software_provenance=get_labelling_software_provenance(),
    )

    # Save out the molecules as they are ready.
    error_file_path = output_path.replace(".sqlite", "-errors.log")

    with open(error_file_path, "w") as file:

        for future in tqdm(
            distributed.as_completed(futures, raise_errors=False),
            total=n_batches,
            desc="labelling molecules",
            ncols=80,
        ):

            for molecule_record, error in future.result():

                try:
                    if molecule_record is not None and error is None:
                        storage.store(molecule_record)
                except (BaseException, Exception) as e:

                    formatted_traceback = traceback.format_exception(
                        etype=type(e), value=e, tb=e.__traceback__
                    )
                    error = f"Could not store record: {formatted_traceback}"

                if error is not None:

                    file.write("=".join(["="] * 40) + "\n")
                    file.write(error + "\n")

                    continue

            future.release()

    if worker_type == "lsf":
        dask_cluster.scale(n=0)


if __name__ == "__main__":
    label_cli()
