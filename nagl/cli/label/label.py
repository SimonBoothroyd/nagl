import functools
import logging
import math
import traceback
from datetime import datetime

import click
from click_option_group import optgroup
from openff.utilities import requires_package
from tqdm import tqdm

from nagl.labelling import label_molecules
from nagl.storage import MoleculeStore
from nagl.utilities.dask import setup_dask_local_cluster, setup_dask_lsf_cluster
from nagl.utilities.provenance import get_labelling_software_provenance
from nagl.utilities.toolkits import capture_toolkit_warnings, stream_from_file

_logger = logging.getLogger(__name__)


@click.command(
    "label",
    short_help="Label molecules with AM1 charges and WBOs",
    help="Compute the AM1 partial charges and AM1 Wiberg bond orders (WBOs) for each "
    "molecule in a given set and store each labelled molecule in a SQLite record store.",
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
@optgroup.group("Charging configuration")
@optgroup.option(
    "--conf-rms",
    "rms_cutoff",
    help="The RMS cutoff [Å] to use when generating the conformers used for charge "
    "generation.",
    type=float,
    default=0.5,
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
    rms_cutoff: float,
    worker_type: str,
    n_workers: int,
    batch_size: int,
    lsf_memory: int,
    lsf_walltime: str,
    lsf_queue: str,
    lsf_env: str,
):

    from dask import distributed

    root_logger: logging.Logger = logging.getLogger("nagl")
    root_logger.setLevel(logging.INFO)

    root_handler = logging.StreamHandler()
    root_handler.setFormatter(logging.Formatter("%(message)s"))

    _logger.info("Labeling molecules")

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

        _logger.warning(
            f"{len(all_smiles) - len(unique_smiles)} duplicate molecules were ignored"
        )

    n_batches = int(math.ceil(len(all_smiles) / batch_size))

    if n_workers < 0:
        n_workers = n_batches

    if n_workers > n_batches:

        _logger.warning(
            f"More workers were requested then there are batches to compute. Only "
            f"{n_batches} workers will be requested."
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

    _logger.info(
        f"{len(unique_smiles)} molecules will labelled in {n_batches} batches across "
        f"{n_workers} workers\n"
    )

    dask_client = distributed.Client(dask_cluster)

    # Submit the tasks to be computed in chunked batches.
    def batch(iterable):
        n_iterables = len(iterable)

        for i in range(0, n_iterables, batch_size):
            yield iterable[i : min(i + batch_size, n_iterables)]

    futures = [
        dask_client.submit(
            functools.partial(
                label_molecules,
                guess_stereochemistry=guess_stereo,
                partial_charge_methods=["am1", "am1bcc"],
                bond_order_methods=["am1"],
                rms_cutoff=rms_cutoff,
            ),
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

            for molecule_record, error in tqdm(
                future.result(),
                desc="storing batch",
                ncols=80,
            ):

                try:

                    with capture_toolkit_warnings():

                        if molecule_record is not None and error is None:
                            storage.store(molecule_record)

                except BaseException as e:

                    formatted_traceback = traceback.format_exception(
                        etype=type(e), value=e, tb=e.__traceback__
                    )
                    error = f"Could not store record: {formatted_traceback}"

                if error is not None:

                    file.write("=".join(["="] * 40) + "\n")
                    file.write(error + "\n")
                    file.flush()

                    continue

            future.release()

    if worker_type == "lsf":
        dask_cluster.scale(n=0)


if __name__ == "__main__":
    label_cli()
