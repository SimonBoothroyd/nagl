import logging
import pathlib

import pyarrow.parquet
import pytest

from nagl.cli.label import label_cli
from nagl.utilities.molecule import molecule_from_smiles, stream_to_file


@pytest.fixture
def mock_molecules(tmp_path) -> pathlib.Path:
    molecule_path = tmp_path / "molecules.sdf"

    with stream_to_file(molecule_path) as writer:
        writer(molecule_from_smiles("C"))
        writer(molecule_from_smiles("[Rn]"))

    return molecule_path


def test_label_cli(tmp_cwd, mock_molecules, runner, caplog):
    expected_output_path = tmp_cwd / "labelled.parquet"

    arguments = [
        "--input",
        mock_molecules,
        "--output",
        expected_output_path,
        "--guess-stereo",
        "False",
    ]

    with caplog.at_level(logging.INFO):
        result = runner.invoke(label_cli, arguments)

    if result.exit_code != 0:
        print(result.stdout, flush=True)
        raise result.exception

    assert "Failed to process [Rn]" in caplog.text

    assert expected_output_path.is_file()

    labels = pyarrow.parquet.read_table("labelled.parquet")

    assert labels.column_names == ["smiles", "charges-am1", "charges-am1bcc"]
    assert len(labels) == 1
    assert labels["smiles"][0].as_py() == "[H:1][C:2]([H:3])([H:4])[H:5]"

    assert b"package-versions" in labels.schema.metadata
