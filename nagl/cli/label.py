import functools
import json
import logging
import multiprocessing
import pathlib

import click
import rich.progress
from click_option_group import optgroup
from openff.utilities import requires_package

import nagl
from nagl.labelling import compute_batch_charges
from nagl.utilities.provenance import get_labelling_software_provenance
from nagl.utilities.toolkits import capture_toolkit_warnings, stream_from_file

_logger = logging.getLogger("nagl.label")

_INPUT_PATH = click.Path(
    exists=True, file_okay=True, dir_okay=False, path_type=pathlib.Path
)
_OUTPUT_PATH = click.Path(
    exists=False, file_okay=True, dir_okay=False, path_type=pathlib.Path
)


@click.command(
    "label",
    short_help="Label molecules with partial charges",
    help="Compute partial charges for each molecule in a given set and store "
    "each them in a parquet data file.",
)
@click.option(
    "--input",
    "input_path",
    help="The path to the input molecules. This should either be an SDF or a GZipped "
    "SDF file.",
    type=_INPUT_PATH,
    required=True,
)
@click.option(
    "--output",
    "output_path",
    help="The parquet file to store the labels in.",
    type=_OUTPUT_PATH,
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
    "--n-confs",
    "n_conformers",
    help="The number of conformers to generate before pruning using ELF10.",
    type=int,
    default=500,
    show_default=True,
)
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
@requires_package("openff.toolkit")
@requires_package("pyarrow")
def label_cli(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    guess_stereo: bool,
    n_conformers: int,
    rms_cutoff: float,
    n_workers: int,
):
    import pyarrow

    logging.basicConfig(level=logging.INFO)
    _logger.info(f"Labeling molecules using nagl version {nagl.__version__}")

    with capture_toolkit_warnings():

        all_smiles = [
            smiles
            for smiles in rich.progress.track(
                stream_from_file(input_path, as_smiles=True),
                description="loading molecules",
            )
        ]

    unique_smiles = sorted({*all_smiles})

    if len(unique_smiles) != len(all_smiles):

        _logger.warning(
            f"{len(all_smiles) - len(unique_smiles)} duplicate molecules were ignored"
        )

    label_func = functools.partial(
        compute_batch_charges,
        methods=["am1", "am1bcc"],
        n_conformers=n_conformers,
        rms_cutoff=rms_cutoff,
        guess_stereo=guess_stereo,
    )

    batch_smiles = [[smiles] for smiles in unique_smiles]

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.imap(
            label_func,
            rich.progress.track(batch_smiles, description="labelling molecules"),
        )

    rows = []

    for (smiles,), (result, error) in zip(batch_smiles, results):

        if error is not None:
            _logger.warning(f"failed to label {smiles} - {error}")
            continue

        rows.append((result["smiles"], result["am1"], result["am1bcc"]))

    smiles, charges_am1, charges_am1bcc = zip(*rows)

    table = pyarrow.table(
        [smiles, charges_am1, charges_am1bcc],
        ["smiles", "charges-am1", "charges-am1bcc"],
        metadata={"package-versions": json.dumps(get_labelling_software_provenance())},
    )

    pyarrow.parquet.write_table(table, output_path)


if __name__ == "__main__":
    label_cli()
