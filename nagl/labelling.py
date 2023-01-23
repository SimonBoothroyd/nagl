import copy
import traceback
import typing

import numpy
from openff.utilities import requires_package

from nagl.utilities.smiles import smiles_to_molecule
from nagl.utilities.toolkits import capture_toolkit_warnings

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


_OPENFF_CHARGE_METHODS = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}
_OPENFF_WBO_METHODS = {"am1": "am1-wiberg"}

ChargeMethod = typing.Literal["am1", "am1bcc"]
WBOMethod = typing.Literal["am1"]


DataLabels = typing.Dict[str, numpy.ndarray]


@requires_package("openff.toolkit")
def compute_charges(
    molecule: typing.Union[str, "Molecule"],
    methods: typing.Optional[typing.Union[ChargeMethod, typing.List[ChargeMethod]]],
    n_conformers: int = 500,
    rms_cutoff: float = 0.05,
    guess_stereo: bool = True,
) -> DataLabels:
    """Computes sets of partial charges  for an input molecule.

    Notes:
        Conformers will be pruned using the ELF10 method provided by the OpenFF toolkit

    Args:
        molecule: The molecule (or SMILES representation of the molecule) to label.
        methods: The method(s) to compute the partial charges using. By
            default, all available methods will be used.
        n_conformers: The *maximum* number of conformers to compute partial charge and
            bond orders using.
        rms_cutoff: The RMS cutoff [Å] to use when generating the conformers.
        guess_stereo: Whether to guess the stereochemistry of the SMILES
            representation of the molecule if provided and if the stereochemistry of
            some atoms / bonds is not fully defined.

    Returns:
        The labelled molecule stored in a record object
    """

    from openff.units import unit

    if isinstance(molecule, str):
        molecule = smiles_to_molecule(molecule, guess_stereo)
    else:
        molecule = copy.deepcopy(molecule)

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
        return_value[method] = numpy.array(charges)

    return return_value


@requires_package("openff.toolkit")
def compute_batch_charges(
    molecules: typing.List[typing.Union[str, "Molecule"]],
    methods: typing.Optional[typing.Union[ChargeMethod, typing.List[ChargeMethod]]],
    n_conformers: int = 500,
    rms_cutoff: float = 0.05,
    guess_stereo: bool = True,
    progress_iterator: typing.Optional[typing.Any] = None,
) -> typing.List[typing.Tuple[typing.Optional[DataLabels], typing.Optional[str]]]:
    """Computes charges for a batch of molecules using ``compute_charges``.

    Args:
        molecules: A list of the molecule (or SMILES representation of the molecules) to
            label.
        methods: The method(s) to compute the partial charges using. By
            default, all available methods will be used.
        n_conformers: The *maximum* number of conformers to compute partial charge and
            bond orders using.
        rms_cutoff: The RMS cutoff [Å] to use when generating the conformers.
        guess_stereo: Whether to guess the stereochemistry of the SMILES
            representation of the molecule if provided and if the stereochemistry of
            some atoms / bonds is not fully defined.
        progress_iterator: An iterator (each a ``tqdm``) to loop over all molecules
           using. This is useful if you wish to display a progress bar for example.
    Returns:
        A list of tuples of the form ``(labelled_record, error_message)``.
    """

    molecules = molecules if progress_iterator is None else progress_iterator(molecules)
    results = []

    with capture_toolkit_warnings():

        for molecule in molecules:

            result = None
            error = None

            try:
                result = compute_charges(
                    molecule, methods, n_conformers, rms_cutoff, guess_stereo
                )
            except BaseException as e:

                formatted_traceback = traceback.format_exception(
                    type(e), e, e.__traceback__
                )
                error = f"Failed to process {str(molecule)}: {formatted_traceback}"

            results.append((result, error))

    return results
