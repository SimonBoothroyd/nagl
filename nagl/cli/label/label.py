import math
import pickle
import sys
from typing import TYPE_CHECKING, List, Optional, Tuple

import click
from click_option_group import optgroup
from tqdm import tqdm

from nagl.labels.am1 import compute_am1_charge_and_wbo
from nagl.utilities.dask import setup_dask_local_cluster, setup_dask_lsf_cluster
from nagl.utilities.openeye import (
    capture_oe_warnings,
    guess_stereochemistry,
    requires_oe_package,
)
from nagl.utilities.utilities import requires_package

if TYPE_CHECKING:
    from openeye import oechem
    from openforcefield.topology import Molecule


@requires_package("openforcefield")
@requires_oe_package("oechem")
def label_molecule_batch(
    oe_molecules: List["oechem.OEMol"], wbo_method: str
) -> List[Tuple[Optional["Molecule"], Optional[str]]]:
    """Labels a batch of molecules using ``compute_am1_charge_and_wbo``.

    Returns
    -------
        A list of tuples. Each tuple will contain the processed molecule containing the
        AM1 charges and WBO if no exceptions were raised (``None`` otherwise) and the
        error string if an exception was raised (``None`` otherwise).
    """
    return [
        compute_am1_charge_and_wbo(oe_molecule, wbo_method=wbo_method)
        for oe_molecule in tqdm(oe_molecules)
    ]


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
    "--wbo-method",
    type=click.Choice(["single-conformer", "elf10-average"]),
    default="single-conformer",
    show_default=True,
    help="The method by which to compute the WBOs. WBOs may be computed using only a "
    "single conformer ('single-conformer'), or by computing the WBO for each ELF10 "
    "conformer and taking the average ('elf10-average')",
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
@requires_oe_package("oechem")
@requires_package("dask.distributed")
@requires_package("openforcefield")
def label_cli(
    input_path: str,
    output_path: str,
    wbo_method: str,
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
    from openeye import oechem

    input_stream = oechem.oemolistream(input_path)

    print(" - Labeling molecules")

    with capture_oe_warnings():

        oe_molecules = [
            oechem.OEMol(oe_molecule) for oe_molecule in input_stream.GetOEMols()
        ]

        input_stream.close()

        if guess_stereo:

            oe_molecules = [
                guess_stereochemistry(oe_molecule) for oe_molecule in oe_molecules
            ]

    n_batches = int(math.ceil(len(oe_molecules) / batch_size))

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
        dask_client.submit(
            label_molecule_batch, batched_molecules, wbo_method=wbo_method
        )
        for batched_molecules in batch(oe_molecules)
    ]

    # Save out the molecules as they are ready.
    with open(output_path, "wb") as file:

        for future in distributed.as_completed(futures, raise_errors=False):

            for molecule, error in future.result():

                if error is not None:

                    print("=".join(["="] * 40), file=sys.stderr)
                    print(error + "\n", file=sys.stderr)

                    continue

                pickle.dump(molecule, file)

            future.release()

    if worker_type == "lsf":
        dask_cluster.scale(n=0)
