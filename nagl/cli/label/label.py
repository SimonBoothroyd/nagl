import math
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import click
from click_option_group import optgroup

from nagl.storage.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
    WibergBondOrderSet,
)
from nagl.utilities import requires_package
from nagl.utilities.dask import setup_dask_local_cluster, setup_dask_lsf_cluster
from nagl.utilities.smiles import smiles_to_molecule
from nagl.utilities.toolkits import capture_toolkit_warnings, stream_from_file

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


@requires_package("openff.toolkit")
def _label_molecule(molecule: "Molecule") -> MoleculeRecord:

    from simtk import unit

    OPENFF_CHARGE_METHODS = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}

    # Generate a diverse set of ELF10 conformers
    molecule.generate_conformers(n_conformers=500, rms_cutoff=0.05 * unit.angstrom)
    molecule.apply_elf_conformer_selection()

    conformer_records = []

    for conformer in molecule.conformers:

        charge_sets = []

        # Compute partial charges.
        for charge_method in ["am1", "am1bcc"]:
            molecule.assign_partial_charges(
                OPENFF_CHARGE_METHODS[charge_method], use_conformers=[conformer]
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
        molecule.assign_fractional_bond_orders(use_conformers=[conformer])

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
    molecules: List["Molecule"],
) -> List[Tuple[Optional[MoleculeRecord], Optional[str]]]:
    """Labels a batch of molecules using ``compute_am1_charge_and_wbo``.

    Returns
    -------
        A list of tuples. Each tuple will contain the processed molecule containing the
        AM1 charges and WBO if no exceptions were raised (``None`` otherwise) and the
        error string if an exception was raised (``None`` otherwise).
    """

    molecule_records = []

    for molecule in molecules:

        molecule_record = None
        error = None

        try:
            molecule_record = _label_molecule(molecule)
        except (BaseException, Exception) as e:
            error = (
                f"Failed to process {molecule.to_smiles(explicit_hydrogens=False)}: "
                f"{str(e)}"
            )

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
    help="The path to save the labelled and pickled molecules to.",
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

        molecules = [
            smiles_to_molecule(molecule.to_smiles(), guess_stereochemistry=guess_stereo)
            for molecule in stream_from_file(input_path)
        ]

    n_batches = int(math.ceil(len(molecules) / batch_size))

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

    dask_client = distributed.Client(dask_cluster)

    # Submit the tasks to be computed in chunked batches.
    def batch(iterable):
        n_iterables = len(iterable)

        for i in range(0, n_iterables, batch_size):
            yield iterable[i : min(i + batch_size, n_iterables)]

    futures = [
        dask_client.submit(label_molecule_batch, batched_molecules)
        for batched_molecules in batch(molecules)
    ]

    # Save out the molecules as they are ready.
    storage = MoleculeStore(output_path)

    for future in distributed.as_completed(futures, raise_errors=False):

        for molecule_record, error in future.result():

            try:
                storage.store(molecule_record)
            except (BaseException, Exception) as e:
                error = str(e)

            if error is not None:

                print("=".join(["="] * 40), file=sys.stderr)
                print(error + "\n", file=sys.stderr)

                continue

        future.release()

    if worker_type == "lsf":
        dask_cluster.scale(n=0)
