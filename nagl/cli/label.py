import math
import pickle
import sys
from typing import List, Optional, Tuple

import click
from click_option_group import optgroup
from dask import distributed
from dask_jobqueue import LSFCluster
from distributed import LocalCluster, as_completed
from openeye import oechem
from openforcefield.topology import Molecule
from tqdm import tqdm

from nagl.labels.am1 import compute_am1_charge_and_wbo
from nagl.utilities.openeye import enumerate_tautomers, guess_stereochemistry


def label_molecule_batch(
    oe_molecules: List[oechem.OEMol],
) -> List[Tuple[Optional[Molecule], Optional[str]]]:
    """Labels a batch of molecules using ``compute_am1_charge_and_wbo``.

    Returns
    -------
        A list of tuples. Each tuple will contain the processed molecule containing the
        AM1 charges and WBO if no exceptions were raised (``None`` otherwise) and the
        error string if an exception was raised (``None`` otherwise).
    """
    return [
        compute_am1_charge_and_wbo(oe_molecule) for oe_molecule in tqdm(oe_molecules)
    ]


@click.command(
    "label", help="Compute AM1 charges and AM1 Wiberg bond orders for a molecule set."
)
@click.option(
    "--input",
    "input_path",
    help="The path to the input molecules. This should either be an SDF or a Gzipped "
    "file.",
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
    "--n-workers",
    help="The number of workers to distribute the labelling across. Use -1 to request "
    "one worker per batch.",
    type=int,
    required=True,
)
@click.option(
    "--worker-type",
    type=click.Choice(["lsf", "local"]),
    default="lsf",
    show_default=True,
    help="The type of worker to distribute the labelling across.",
)
@click.option(
    "--batch-size",
    help="The number of molecules to processes at once on a particular worker.",
    type=int,
    default=500,
    show_default=True,
)
@optgroup.group("LSF configuration", help="Options configure LSF workers.")
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
def label(
    input_path: str,
    output_path: str,
    worker_type: str,
    n_workers: int,
    batch_size: int,
    lsf_memory: int,
    lsf_walltime: str,
    lsf_queue: str,
    lsf_env: str,
):

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(input_path)

    print(" - Enumerating tautomers")

    output_stream = oechem.oeosstream()

    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()
    enumerated_molecules = [
        oechem.OEMol(oe_tautomer)
        for oe_molecule in input_molecule_stream.GetOEMols()
        for oe_tautomer in enumerate_tautomers(guess_stereochemistry(oe_molecule))
    ]

    input_molecule_stream.close()
    oechem.OEThrow.SetOutputStream(oechem.oeerr)

    print(" - Stripping salts")

    for oe_molecule in enumerated_molecules:
        oechem.OEDeleteEverythingExceptTheFirstLargestComponent(oe_molecule)

    print(" - Labeling molecules")

    n_batches = int(math.ceil(len(enumerated_molecules) / batch_size))

    if n_workers < 0:
        n_workers = n_batches

    if n_workers > n_batches:
        print(
            f"\n    [WARNING] More workers were requested then there are batches to "
            f"compute. Only {n_batches} workers will be requested."
        )

        n_workers = n_batches

    # Set-up dask to distribute the processing.
    if worker_type == "lsf":
        dask_cluster = LSFCluster(
            queue=lsf_queue,
            cores=1,
            memory=f"{lsf_memory * 1e9}B",
            walltime=lsf_walltime,
            local_directory="dask-worker-space",
            log_directory="dask-worker-logs",
            env_extra=[f"conda activate {lsf_env}"],
        )
        dask_cluster.scale(n=n_workers)

    elif worker_type == "local":
        dask_cluster = LocalCluster(n_workers=n_workers)
    else:
        raise NotImplementedError()

    dask_client = distributed.Client(dask_cluster)

    # Submit the tasks to be computed in chunked batches.
    def batch(iterable, batch_size):
        n_iterables = len(iterable)

        for i in range(0, n_iterables, batch_size):
            yield iterable[i : min(i + batch_size, n_iterables)]

    futures = [
        dask_client.submit(label_molecule_batch, batched_molecules)
        for batched_molecules in batch(enumerated_molecules, batch_size=batch_size)
    ]

    # Save out the molecules as they are ready.
    with open(output_path, "wb") as file:

        for future in as_completed(futures, raise_errors=False):

            for molecule, error in future.result():

                if error is not None:

                    print("=".join(["="] * 40), file=sys.stderr)
                    print(error + "\n", file=sys.stderr)

                    continue

                pickle.dump(molecule, file)

            future.release()

    if worker_type == "lsf":
        dask_cluster.scale(n=0)
