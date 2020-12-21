import pickle
import sys

import click
from dask import distributed
from dask_jobqueue import LSFCluster
from distributed import LocalCluster, as_completed
from openeye import oechem

from nagl.labels.am1 import compute_am1_charge_and_wbo
from nagl.utilities.openeye import enumerate_tautomers, guess_stereochemistry


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
    n_workers: int,
    queue: str,
    conda_env: str,
):

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(input_path)

    enumerated_molecules = (
        oechem.OEMol(oe_tautomer)
        for oe_molecule in input_molecule_stream.GetOEMols()
        for oe_tautomer in enumerate_tautomers(guess_stereochemistry(oe_molecule))
    )

    # Set-up dask to distribute the processing.
    if worker_type == "lsf":
        dask_cluster = LSFCluster(
            queue=queue,
            cores=1,
            memory=f"{worker_memory * 1e9}B",
            walltime="02:00",
            local_directory="dask-worker-space",
            log_directory="dask-worker-logs",
            env_extra=[f"conda activate {conda_env}"],
        )
        dask_cluster.adapt(minimum=1, maximum=n_workers)

    elif worker_type == "local":
        dask_cluster = LocalCluster(n_workers=n_workers)
    else:
        raise NotImplementedError()

    dask_client = distributed.Client(dask_cluster)

    futures = (
        dask_client.submit(compute_am1_charge_and_wbo, x) for x in enumerated_molecules
    )

    molecules = []

    for i, future in enumerate(as_completed(futures)):

        molecule, error = future.result()

        if i % 1000 == 0:
            print(f"Finished labelling molecule {i + 1}.", file=sys.stdout)

        if error is not None:

            print("=".join(["="] * 40), file=sys.stderr)
            print(error + "\n", file=sys.stderr)

            continue

        molecules.append(molecule)

    with open(output_path, "wb") as file:
        pickle.dump(molecules, file)

    input_molecule_stream.close()
