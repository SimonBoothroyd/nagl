import torch
from openff.toolkit import Molecule

from nagl.molecules import DGLMolecule
from nagl.reporting._reporting import (
    _draw_molecule_with_atom_labels,
    _generate_per_atom_jinja_dicts,
    create_atom_label_report,
)


def test_draw_molecule_with_atom_labels():

    molecule = Molecule.from_smiles("[Cl-]")

    svg = _draw_molecule_with_atom_labels(
        molecule, torch.tensor([9.87]), torch.tensor([1.23])
    )
    assert "svg" in svg


def test_generate_per_atom_jinja_dicts():

    entries = [
        (Molecule.from_smiles("[H]Cl"), torch.zeros(2), torch.zeros(2)),
        (DGLMolecule.from_smiles("[H]Br", [], []), torch.zeros(2), torch.zeros(2)),
    ]

    jinja_dicts = _generate_per_atom_jinja_dicts(entries, ["rmse"], True, 1.0)
    assert len(jinja_dicts) == 2

    assert all("img" in item for item in jinja_dicts)
    assert all("metrics" in item for item in jinja_dicts)
    assert all("RMSE" in item["metrics"] for item in jinja_dicts)


def test_create_atom_label_report(tmp_cwd):

    entries = [
        (Molecule.from_smiles("[H]Cl"), torch.zeros(2), torch.ones(2)),
        (Molecule.from_smiles("[H]Br"), torch.zeros(2), torch.zeros(2)),
        (Molecule.from_smiles("[H]O[H]"), torch.zeros(3), torch.zeros(3)),
    ]

    report_path = tmp_cwd / "report.html"

    create_atom_label_report(entries, ["rmse"], "rmse", report_path, 2, 1)
    assert report_path.is_file()

    report_contents = report_path.read_text()
    assert "Top 2 Structures" in report_contents
    assert "Bottom 1 Structures" in report_contents
