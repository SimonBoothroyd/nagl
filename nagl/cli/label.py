import pickle

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
def label(input_path: str, output_path: str, worker_type: str, n_workers: int):

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
            queue="default",
            cores=1,
            memory="2000000000B",
            walltime="02:00",
            local_directory="dask-worker-space",
        )
        dask_cluster.adapt(
            minimum=1,
            maximum=n_workers,
            interval="10000ms",
            target_duration="0.00000000001s",
        )

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
            print(f"Finished labelling molecule {i + 1}.")

        if error is not None:

            print("=".join(["="] * 40))
            print(error + "\n")

            continue

        molecules.append(molecule)

    with open(output_path, "wb") as file:
        pickle.dump(molecules, file)

    input_molecule_stream.close()
