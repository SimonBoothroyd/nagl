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

_TEMPLATE_DIR = pathlib.Path(__file__).parent


def _draw_molecule_with_atom_labels(
    molecule: Chem.Mol,
    pred: torch.Tensor,
    ref: torch.Tensor,
    highlight_outliers: bool = False,
    outlier_threshold: float = 1.0,
) -> str:
    """Renders two molecules with per atom labels as an SVG image - one showing
    predicted labels and another showing reference ones.
    """

    highlight_atoms = None

    if highlight_outliers:
        delta_sq = torch.abs(pred.squeeze() - ref.squeeze())

        delta_mean = delta_sq.mean()
        delta_std = delta_sq.std()

        should_highlight = (delta_sq - delta_mean) > (delta_std * outlier_threshold)

        highlight_atoms = [i for i, outlier in enumerate(should_highlight) if outlier]

    pred_molecule = Chem.Mol(molecule)

    for atom, label in zip(pred_molecule.GetAtoms(), pred.detach().numpy()):
        atom.SetProp("atomNote", str(f"{float(label):.3f}"))

    Draw.PrepareMolForDrawing(pred_molecule)

    ref_molecule = Chem.Mol(molecule)

    for atom, label in zip(ref_molecule.GetAtoms(), ref.detach().numpy()):
        atom.SetProp("atomNote", str(f"{float(label):.3f}"))

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
        highlightAtomLists=[highlight_atoms, highlight_atoms],
    )
    return image


def _generate_per_atom_jinja_dicts(
    entries: typing.List[typing.Tuple[Chem.Mol, torch.Tensor, torch.Tensor]],
    metrics: typing.List[MetricType],
    highlight_outliers: bool,
    outlier_threshold: float,
):
    metrics_funcs = {
        metric: nagl.training.metrics.get_metric(metric) for metric in metrics
    }

    return_value = []

    for molecule, per_atom_pred, per_atom_ref in entries:
        if isinstance(molecule, DGLMolecule):
            molecule = molecule.to_rdkit()

        entry_metrics = {
            metric.upper(): f"{metrics_func(per_atom_pred, per_atom_ref):.4f}"
            for metric, metrics_func in metrics_funcs.items()
        }

        image = _draw_molecule_with_atom_labels(
            molecule, per_atom_pred, per_atom_ref, highlight_outliers, outlier_threshold
        )

        image_encoded = base64.b64encode(image.encode()).decode()
        image_src = f"data:image/svg+xml;base64,{image_encoded}"

        return_value.append({"img": image_src, "metrics": entry_metrics})

    return return_value


def create_atom_label_report(
    entries: typing.List[typing.Tuple[Chem.Mol, torch.Tensor, torch.Tensor]],
    metrics: typing.List[MetricType],
    rank_by: MetricType,
    output_path: pathlib.Path,
    top_n_entries: int = 100,
    bottom_n_entries: int = 100,
    highlight_outliers: bool = True,
    outlier_threshold: float = 1.0,
):
    """Creates a simple HTML report that shows the values of predicted and reference
    labels for the top N and bottom M entries in the specified list.

    Args:
        entries: The list of molecules to consider for the report. Each entry should be
            a tuple of the form ``(molecule, per_atom_pred, per_atom_ref)``. Here
            ``per_atom_pred`` should be a tensor with ``shape=(n_atoms)`` and contain
            predictions by a model, and ``per_atom_ref`` the same but containing the
            reference labels.
        metrics: The metrics to compute for each entry.
        rank_by: The metric to rank the entries by.
        output_path: The path to save the report to.
        top_n_entries: The number of highest ranking entries to show according to
            ``rank_by``.
        bottom_n_entries: The number of lowest ranking entries to show according to
            ``rank_by``.
        highlight_outliers: Whether to highlight atoms whose predicted and reference
            labels differ by more than ``outlier_threshold * std(|pred - ref|)``
        outlier_threshold: The threshold for detecting outliers.
    """

    entries_and_ranks = []

    rank_by_func = nagl.training.metrics.get_metric(rank_by)

    for entry in entries:
        _, per_atom_pred, per_atom_ref = entry

        metric = rank_by_func(per_atom_pred, per_atom_ref)
        entries_and_ranks.append((entry, metric))

    entries_and_ranks = sorted(entries_and_ranks, key=lambda x: x[1], reverse=True)

    top_n_structures = _generate_per_atom_jinja_dicts(
        [x for x, _ in entries_and_ranks[:top_n_entries]],
        metrics,
        highlight_outliers,
        outlier_threshold,
    )
    bottom_n_structures = _generate_per_atom_jinja_dicts(
        [x for x, _ in entries_and_ranks[-bottom_n_entries:]],
        metrics,
        highlight_outliers,
        outlier_threshold,
    )

    template_loader = jinja2.FileSystemLoader(searchpath=_TEMPLATE_DIR)
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template("template.html")
    rendered = template.render(
        top_n_structures=top_n_structures, bottom_n_structures=bottom_n_structures
    )

    output_path.write_text(rendered)


def _draw_molecule(
    molecule: typing.Union[Chem.Mol, DGLMolecule],
) -> str:
    """
    Renders a molecule as an SVG image.

    Args:
        molecule: The rdkit or DGLMolecule which should be rendered.

    Returns:
        The SVG text
    """

    if isinstance(molecule, DGLMolecule):
        molecule = molecule.to_rdkit()

    draw_mol = Chem.Mol(molecule)

    drawer = Draw.rdMolDraw2D.MolDraw2DSVG(400, 400)
    Draw.PrepareAndDrawMolecule(drawer, draw_mol)

    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _generate_molecule_jinja_dicts(
    entries_and_metrics: typing.List[
        typing.Tuple[typing.Union[Chem.Mol, DGLMolecule], torch.Tensor]
    ],
    metric_label: str,
) -> typing.List[typing.Dict]:
    """
    Create dictionaries for each of the molecules containing the SVG images and metrics to be displayed in
    the artifact.

    Args:
        entries_and_metrics: A list of tuples of the form ``(molecule, metric value)`` containing the molecule
            and its associated metric calculated using the metric set in `metric_label`
        metric_label: The name of the metric used to calculate the value for example `rmse`.

    Returns:
        A list of dictionaries of data for the jinja template.
    """
    all_dicts = []
    for molecule, metric in entries_and_metrics:
        image = _draw_molecule(molecule=molecule)
        image_encoded = base64.b64encode(image.encode()).decode()
        image_src = f"data:image/svg+xml;base64,{image_encoded}"
        data = {"img": image_src, "metrics": {metric_label.upper(): f"{metric:.4f}"}}
        all_dicts.append(data)

    return all_dicts


def create_molecule_label_report(
    entries_and_metrics: typing.List[
        typing.Tuple[typing.Union[Chem.Mol, DGLMolecule], torch.Tensor]
    ],
    metric_label: str,
    output_path: pathlib.Path,
    top_n_entries: int = 100,
    bottom_n_entries: int = 100,
):
    """
    Creates a simple HTML report that shows metrics for molecule level labels like Dipole
    for the top N and bottom M entries in the specified list.

    Args:
        entries_and_metrics: The list of molecules and their corresponding metrics to consider for the report.
            Each entry should be a tuple of the form ``(molecule, calculated_metric)``.
        metric_label: The name of the metric used to calculate the value for example `rmse`.
        output_path: The path to save the report to.
        top_n_entries: The number of the highest ranking entries to show according to
            ``rank_by``.
        bottom_n_entries: The number of the lowest ranking entries to show according to
            ``rank_by``.
    """

    entries_and_metrics = sorted(entries_and_metrics, key=lambda x: x[1], reverse=True)
    top_n_structures = _generate_molecule_jinja_dicts(
        entries_and_metrics=entries_and_metrics[:top_n_entries],
        metric_label=metric_label,
    )
    bottom_n_structures = _generate_molecule_jinja_dicts(
        entries_and_metrics=entries_and_metrics[-bottom_n_entries:],
        metric_label=metric_label,
    )

    template_loader = jinja2.FileSystemLoader(searchpath=_TEMPLATE_DIR)
    template_env = jinja2.Environment(loader=template_loader)

    template = template_env.get_template("template.html")
    rendered = template.render(
        top_n_structures=top_n_structures, bottom_n_structures=bottom_n_structures
    )

    output_path.write_text(rendered)
