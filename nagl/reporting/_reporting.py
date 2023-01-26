import base64
import pathlib
import typing

import jinja2
import torch
from rdkit import Chem
from rdkit.Chem import Draw

import nagl.training.metrics
from nagl.config.data import MetricType
from nagl.molecules import DGLMolecule

if typing.TYPE_CHECKING:
    from openff.toolkit import Molecule

_TEMPLATE_DIR = pathlib.Path(__file__).parent


def _draw_molecule_with_atom_labels(
    molecule: "Molecule", pred: torch.Tensor, ref: torch.Tensor
) -> str:
    """Renders two molecules with per atom labels as an SVG image - one showing
    predicted labels and another showing reference ones.
    """

    pred_molecule: Chem = molecule.to_rdkit()

    for atom, label in zip(pred_molecule.GetAtoms(), pred.detach().numpy()):
        atom.SetProp("atomNote", str(f"{label:.3f}"))

    Draw.PrepareMolForDrawing(pred_molecule)

    ref_molecule: Chem = molecule.to_rdkit()

    for atom, label in zip(ref_molecule.GetAtoms(), ref.detach().numpy()):
        atom.SetProp("atomNote", str(f"{label:.3f}"))

    Draw.PrepareMolForDrawing(ref_molecule)

    draw_options = Draw.MolDrawOptions()
    draw_options.legendFontSize = 25

    image = Draw.MolsToGridImage(
        [pred_molecule, ref_molecule],
        legends=["prediction", "reference"],
        molsPerRow=2,
        subImgSize=(400, 400),
        useSVG=True,
        drawOptions=draw_options,
    )
    return image


def _generate_per_atom_jinja_dicts(
    entries: typing.List[typing.Tuple["Molecule", torch.Tensor, torch.Tensor]],
    metrics: typing.List[MetricType],
):

    metrics_funcs = {
        metric: nagl.training.metrics.get_metric(metric) for metric in metrics
    }

    return_value = []

    for molecule, per_atom_pred, per_atom_ref in entries:

        if isinstance(molecule, DGLMolecule):
            molecule = molecule.to_openff()

        image = _draw_molecule_with_atom_labels(molecule, per_atom_pred, per_atom_ref)

        image_encoded = base64.b64encode(image.encode()).decode()
        image_src = f"data:image/svg+xml;base64,{image_encoded}"

        entry_metrics = {
            metric.upper(): f"{metrics_func(per_atom_pred, per_atom_ref):.4f}"
            for metric, metrics_func in metrics_funcs.items()
        }
        return_value.append({"img": image_src, "metrics": entry_metrics})

    return return_value


def create_atom_label_report(
    entries: typing.List[typing.Tuple["Molecule", torch.Tensor, torch.Tensor]],
    metrics: typing.List[MetricType],
    rank_by: MetricType,
    output_path: pathlib.Path,
    top_n_entries: int = 100,
    bottom_n_entries: int = 100,
):
    """Creates a simple HTML report that shows the values of predicted and reference
    labels for the top N and bottom M entries in the specified list.

    Args:
        entries: The list of molecules to consider for the report. Each entry should be
            a tuple of the form ``(molecule, per_atom_pred, per_atom_ref)``. Here
            ``per_atom_pred`` should be a tensor with ``shape=(n_atoms)`` and contain
            predictions by a model, and ``per_atom_ref`` the same but containg the
            reference labels.
        metrics: The metrics to compute for each entry.
        rank_by: The metric to rank the entries by.
        output_path: The path to save the report to.
        top_n_entries: The number of highest ranking entries to show according to
            ``rank_by``.
        bottom_n_entries: The number of lowest ranking entries to show according to
            ``rank_by``.
    """

    entries_and_ranks = []

    rank_by_func = nagl.training.metrics.get_metric(rank_by)

    for entry in entries:

        molecule, per_atom_pred, per_atom_ref = entry

        metric = rank_by_func(per_atom_pred, per_atom_ref)
        entries_and_ranks.append((entry, metric))

    entries_and_ranks = sorted(entries_and_ranks, key=lambda x: x[1], reverse=True)

    top_n_structures = _generate_per_atom_jinja_dicts(
        [x for x, _ in entries_and_ranks[:top_n_entries]], metrics
    )
    bottom_n_structures = _generate_per_atom_jinja_dicts(
        [x for x, _ in entries_and_ranks[-bottom_n_entries:]], metrics
    )

    template_loader = jinja2.FileSystemLoader(searchpath=_TEMPLATE_DIR)
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template("template.html")
    rendered = template.render(
        top_n_structures=top_n_structures, bottom_n_structures=bottom_n_structures
    )

    output_path.write_text(rendered)
