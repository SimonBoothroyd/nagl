import pytest
import torch

from nagl.molecules import DGLMolecule
from nagl.reporting._reporting import (
    _draw_molecule,
    _draw_molecule_with_atom_labels,
    _generate_molecule_jinja_dicts,
    _generate_per_atom_jinja_dicts,
    create_atom_label_report,
    create_molecule_label_report,
)
from nagl.utilities.molecule import molecule_from_smiles


def test_draw_molecule_with_atom_labels():
    molecule = molecule_from_smiles("[Cl-]")

    svg = _draw_molecule_with_atom_labels(
        molecule, torch.tensor([9.87]), torch.tensor([1.23])
    )
    assert "svg" in svg


def test_generate_per_atom_jinja_dicts():
    entries = [
        (molecule_from_smiles("[H]Cl"), torch.zeros(2), torch.zeros(2)),
        (DGLMolecule.from_smiles("[H]Br", [], []), torch.zeros(2), torch.zeros(2)),
    ]

    jinja_dicts = _generate_per_atom_jinja_dicts(entries, ["rmse"], True, 1.0)
    assert len(jinja_dicts) == 2

    assert all("img" in item for item in jinja_dicts)
    assert all("metrics" in item for item in jinja_dicts)
    assert all("RMSE" in item["metrics"] for item in jinja_dicts)


def test_create_atom_label_report(tmp_cwd):
    entries = [
        (molecule_from_smiles("[H]Cl"), torch.zeros(2), torch.ones(2)),
        (molecule_from_smiles("[H]Br"), torch.zeros(2), torch.zeros(2)),
        (molecule_from_smiles("[H]O[H]"), torch.zeros(3), torch.zeros(3)),
    ]

    report_path = tmp_cwd / "report.html"

    create_atom_label_report(entries, ["rmse"], "rmse", report_path, 2, 1)
    assert report_path.is_file()

    report_contents = report_path.read_text()
    assert "Top 2 Structures" in report_contents
    assert "Bottom 1 Structures" in report_contents


@pytest.mark.parametrize(
    "input_type",
    [pytest.param("dgl_methane", id="DGL"), pytest.param("rdkit_methane", id="RDKit")],
)
def test_draw_molecule(input_type, request):
    """
    Make sure the molecule can be drawn regardless if it is rdkit mol or DGLmolecule
    """
    molecule = request.getfixturevalue(input_type)
    image = _draw_molecule(molecule=molecule)
    assert "svg" in image


def test_generate_molecule_jinja_dict(rdkit_methane, dgl_methane):
    """
    Test making the jinja dicts for the html template for molecular level labels
    """
    entries = [
        (dgl_methane, torch.Tensor([1]).squeeze()),
        (rdkit_methane, torch.Tensor([0.5]).squeeze()),
    ]
    dicts = _generate_molecule_jinja_dicts(
        entries_and_metrics=entries, metric_label="rmse"
    )
    assert len(dicts) == 2
    assert all("img" in item for item in dicts)
    assert all("metrics" in item for item in dicts)
    assert all("RMSE" in item["metrics"] for item in dicts)


def test_create_molecular_label_report(rdkit_methane, dgl_methane, tmp_cwd):
    """
    Generate a molecular level label report this should be single images of molecules ranked by metric.
    """
    entries = [
        (dgl_methane, torch.Tensor([1]).squeeze()),
        (rdkit_methane, torch.Tensor([0.5]).squeeze()),
    ]
    report_path = tmp_cwd.joinpath("dipole.html")
    create_molecule_label_report(
        entries_and_metrics=entries,
        metric_label="rmse",
        output_path=report_path,
        top_n_entries=2,
        bottom_n_entries=1,
    )
    assert report_path.is_file()

    report_contents = report_path.read_text()
    assert "Top 2 Structures" in report_contents
    assert "Bottom 1 Structures" in report_contents
