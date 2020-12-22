import pickle
import sys
from typing import List, Optional, Tuple

import click
from dask import distributed
from dask_jobqueue import LSFCluster
from distributed import LocalCluster, as_completed
from openeye import oechem
from openforcefield.topology import Molecule

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
    return [compute_am1_charge_and_wbo(oe_molecule) for oe_molecule in oe_molecules]


@click.command(
    "label", help="Compute AM1 charges and AM1 Wiberg bond orders for a molecule set."
)
@click.option(
    "--input",
    "input_path",
    help="The path to the input molecules. This should either be an SDF or a Gzipped "
    "file.",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
)
@click.option(
    "--output",
    "output_path",
    default="labelled-molecules.pkl",
    help="The path to save the labelled and pickled molecules to.",
    type=click.Path(exists=False, file_okay=True, dir_okay=False),
)
@click.option(
    "--worker-type",
    type=click.Choice(["lsf", "local"]),
    default="lsf",
    show_default=True,
    help="The type of worker to distribute the labelling across.",
)
@click.option(
    "--n-workers",
    help="The number of workers to distribute the labelling across.",
    type=int,
)
@click.option(
    "--worker-memory",
    help="The amount of memory (GB) to request per LSF queue worker.",
    default=3,
    show_default=True,
    type=int,
)
@click.option(
    "--worker-time",
    help="The maximum wall-clock time to request per LSF queue worker.",
    default="02:00",
    show_default=True,
    type=str,
)
@click.option(
    "--queue",
    help="The LSF queue to submit workers to.",
    type=str,
    default="default",
    show_default=True,
)
@click.option(
    "--conda-env",
    help="The conda environment that LSF workers should run using.",
    type=str,
)
def label(
    input_path: str,
    output_path: str,
    worker_type: str,
    worker_memory: int,
    worker_time: str,
    n_workers: int,
    queue: str,
    conda_env: str,
):

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(input_path)

    print("Enumerating tautomers")

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

    print("Stripping salts")

    for oe_molecule in enumerated_molecules:
        oechem.OEDeleteEverythingExceptTheFirstLargestComponent(oe_molecule)

    print("Labeling molecules")

    # Set-up dask to distribute the processing.
    if worker_type == "lsf":
        dask_cluster = LSFCluster(
            queue=queue,
            cores=1,
            memory=f"{worker_memory * 1e9}B",
            walltime=worker_time,
            local_directory="dask-worker-space",
            log_directory="dask-worker-logs",
            env_extra=[f"conda activate {conda_env}"],
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
        for batched_molecules in batch(enumerated_molecules, batch_size=500)
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
