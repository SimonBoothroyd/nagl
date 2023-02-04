import functools
import json
import pathlib

import pyarrow.parquet
import rich
import rich.progress

from nagl.labelling import compute_charges_func, label_molecules
from nagl.utilities.provenance import default_software_provenance


def main():
    console = rich.get_console()

    input_data = {
        "train": ["C", "CCCC", "CCCCCC"],
        "val": ["CC", "CCC"],
        "test": ["CCCCCCC"],
    }

    output_dir = pathlib.Path("000-label-data")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Define any metadata / provenance we wish to store with the labels.
    metadata = {
        "label_func": "nagl.labelling.compute_charges_func",
        "software": json.dumps(default_software_provenance()),
    }

    for stage, smiles_set in input_data.items():
        progress_bar = functools.partial(
            rich.progress.track, description=f"labelling {stage}"
        )

        labels, errors = label_molecules(
            smiles_set,
            compute_charges_func(methods=["am1", "am1bcc"], n_conformers=500),
            metadata=metadata,
            guess_stereo=True,
            progress_iterator=progress_bar,
        )

        for error in errors:
            console.print(f"[WARNING] {error}")
            continue

        pyarrow.parquet.write_table(labels, output_dir / f"{stage}.parquet")


if __name__ == "__main__":
    main()
