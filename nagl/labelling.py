import functools
import traceback
import typing
from collections import defaultdict

import numpy
import pyarrow
from rdkit import Chem

from nagl.utilities import get_map_func
from nagl.utilities.molecule import molecule_from_smiles

_OPENFF_CHARGE_METHODS = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}


ChargeMethod = typing.Literal["am1", "am1bcc"]

Labels = typing.Dict[str, numpy.ndarray]
LabelFunction = typing.Callable[[Chem.Mol], Labels]

ProgressIterator = typing.Callable[
    [typing.Iterable[typing.Union[str, Chem.Mol]]],
    typing.Iterable[typing.Union[str, Chem.Mol]],
]


def compute_charges(
    molecule: Chem.Mol,
    methods: typing.Optional[typing.Union[ChargeMethod, typing.List[ChargeMethod]]],
    n_conformers: int = 500,
    rms_cutoff: float = 0.05,
) -> Labels:
    """Computes sets of partial charges  for an input molecule.

    Notes:
        Conformers will be pruned using the ELF10 method provided by the OpenFF toolkit

    Args:
        molecule: The molecule to label.
        methods: The method(s) to compute the partial charges using. By
            default, all available methods will be used.
        n_conformers: The *maximum* number of conformers to compute partial charge and
            bond orders using.
        rms_cutoff: The RMS cutoff [Å] to use when generating the conformers.

    Returns:
        The labelled molecule stored in a record object
    """

    from openff.toolkit.topology import Molecule
    from openff.units import unit

    molecule = Molecule.from_rdkit(molecule)

    methods = [methods] if isinstance(methods, str) else methods
    methods = methods if methods is not None else [*_OPENFF_CHARGE_METHODS]

    molecule.generate_conformers(
        n_conformers=n_conformers, rms_cutoff=rms_cutoff * unit.angstrom
    )
    molecule.apply_elf_conformer_selection()

    return_value = {"smiles": molecule.to_smiles(mapped=True)}

    for method in methods:
        molecule.assign_partial_charges(
            _OPENFF_CHARGE_METHODS[method], use_conformers=molecule.conformers
        )
        charges = [
            atom.partial_charge.m_as(unit.elementary_charge) for atom in molecule.atoms
        ]
        return_value[f"charges-{method}"] = numpy.array(charges).reshape(-1, 1)

    return return_value


def compute_charges_func(
    methods: typing.Optional[typing.Union[ChargeMethod, typing.List[ChargeMethod]]],
    n_conformers: int = 500,
    rms_cutoff: float = 0.05,
) -> LabelFunction:
    """Returns a function for computing a set of partial charges for an input molecule.
    See ``nagl.labelling.compute_charges`` for details.

    Args:
        methods: The method(s) to compute the partial charges using. By
            default, all available methods will be used.
        n_conformers: The *maximum* number of conformers to compute partial charge and
            bond orders using.
        rms_cutoff: The RMS cutoff [Å] to use when generating the conformers.

    Returns:
        A wrapped version of ``nagl.labelling.compute_charges`` that only requires
        a molecule as input.
    """

    return functools.partial(
        compute_charges,
        methods=methods,
        n_conformers=n_conformers,
        rms_cutoff=rms_cutoff,
    )


def _label_molecule(
    molecule: typing.Union[str, Chem.Mol],
    label_func: LabelFunction,
    guess_stereo: bool = True,
) -> typing.Tuple[typing.Optional[Labels], typing.Optional[str]]:
    try:
        molecule = (
            molecule
            if not isinstance(molecule, str)
            else molecule_from_smiles(molecule, guess_stereo)
        )
        return label_func(molecule), None

    except BaseException as e:
        smiles = None if molecule is None else Chem.MolToSmiles(molecule)

        formatted_traceback = traceback.format_exception(type(e), e, e.__traceback__)
        return None, f"Failed to process {str(smiles)}: {formatted_traceback}"


def label_molecules(
    molecules: typing.List[typing.Union[str, Chem.Mol]],
    label_func: LabelFunction,
    metadata: typing.Optional[typing.Dict[str, str]] = None,
    guess_stereo: bool = True,
    progress_iterator: typing.Optional[ProgressIterator] = None,
    n_processes: int = 0,
) -> typing.Tuple[typing.Optional[pyarrow.Table], typing.List[str]]:
    """Computes labels for a batch of molecules using ``label_func``.

    Args:
        molecules: A list of the molecule (or SMILES representation of the molecules) to
            label.
        label_func: A function that should take a molecule or SMILES pattern as input
            and return a dictionary of labels.

            The labels **must** contain a ``'smiles'`` key that corresponds to the
            **mapped** SMILES representation of the molecule that was labelled.

            Each additional key should map to a 2D numpy array with ``shape=(n, m)``
            where ``n`` is the number of labelled elements (e.g. atoms, bonds, angles).
        metadata: Metadata to store in the table, such as provenance information.
        guess_stereo: Whether to guess the stereochemistry of SMILES patterns that
            do not contain full information.
        progress_iterator: An iterator (e.g. a ``tqdm`` or rich ``ProgressBar``) to
            loop over all molecules using. This is useful if you wish to display a
            progress bar for example.
        n_processes: The number of processes to parallelize the labelling over.
    Returns:
        A pyarrow table containing the labels, and a list of errors for molecules
        that could not be labelled.
    """

    molecules = molecules if progress_iterator is None else progress_iterator(molecules)

    results = defaultdict(list)
    errors = []

    label_molecule_func = functools.partial(
        _label_molecule, label_func=label_func, guess_stereo=guess_stereo
    )

    with get_map_func(n_processes) as map_func:
        labels_and_errors = map_func(label_molecule_func, molecules)

    for labels, error in labels_and_errors:
        if error is not None:
            errors.append(error)
            continue

        smiles = labels.pop("smiles")
        results["smiles"].append(smiles)

        for label, values in labels.items():
            assert values.ndim == 2, "labels must be arrays with 2 dimensions"
            results[label].append(values.tolist())

    column_names = ["smiles"] + sorted(
        label for label in results if label.lower() != "smiles"
    )
    columns = [] if len(results) == 0 else [results[label] for label in column_names]

    table = pyarrow.table(columns, column_names, metadata=metadata)
    return table, errors
