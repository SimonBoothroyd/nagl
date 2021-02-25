import functools
import logging
import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Type, TypeVar

from typing_extensions import Literal

from nagl.utilities.utilities import MissingOptionalDependency, requires_package

if TYPE_CHECKING:
    from openeye import oechem

logger = logging.getLogger(__name__)

T = TypeVar("T")


class MoleculeFromSmilesError(RuntimeError):
    """An exception raised when attempting to create a molecule from a
    SMILES pattern."""

    def __init__(self, *args, smiles: str, **kwargs):
        """

        Parameters
        ----------
        smiles
            The SMILES pattern which could not be parsed.
        """

        super(MoleculeFromSmilesError, self).__init__(*args, **kwargs)
        self.smiles = smiles


def requires_oe_package(
    package_name: Literal["oechem", "oeomega", "oequacpac", "oemolprop"]
):
    @requires_package(f"openeye.{package_name}")
    def inner_decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):

            from openeye import oechem, oemolprop, oeomega, oequacpac

            if (
                (package_name == "oechem" and not oechem.OEChemIsLicensed())
                or (package_name == "oechem" and not oeomega.OEOmegaIsLicensed())
                or (package_name == "oequacpac" and not oequacpac.OEQuacPacIsLicensed())
                or (package_name == "oemolprop" and not oemolprop.OEMolPropIsLicensed())
            ):

                raise MissingOptionalDependency(f"openeye.{package_name}", True)

            return function(*args, **kwargs)

        return wrapper

    return inner_decorator


@requires_oe_package("oechem")
def call_openeye(
    oe_callable: Callable[[T], bool],
    *args: T,
    exception_type: Type[BaseException] = RuntimeError,
    exception_kwargs: Dict[str, Any] = None,
):
    """Wraps a call to an OpenEye function, either capturing the output in an
    exception if the function does not complete successfully, or redirecting it
    to the logger.

    Parameters
    ----------
    oe_callable
        The OpenEye function to call.
    args
        The arguments to pass to the OpenEye function.
    exception_type:
        The type of exception to raise when the function does not
        successfully complete.
    exception_kwargs
        The keyword arguments to pass to the exception.
    """

    from openeye import oechem

    if exception_kwargs is None:
        exception_kwargs = {}

    output_stream = oechem.oeosstream()

    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    status = oe_callable(*args)

    oechem.OEThrow.SetOutputStream(oechem.oeerr)

    output_string = output_stream.str().decode("UTF-8")

    output_string = output_string.replace("Warning: ", "")
    output_string = re.sub("^: +", "", output_string, flags=re.MULTILINE)
    output_string = re.sub("\n$", "", output_string)

    if not status:

        # noinspection PyArgumentList
        raise exception_type("\n" + output_string, **exception_kwargs)

    elif len(output_string) > 0:
        logger.debug(output_string)


@requires_oe_package("oechem")
@requires_oe_package("oeomega")
def guess_stereochemistry(oe_molecule: "oechem.OEMol") -> "oechem.OEMol":
    """Generates and returns a random stereoisomer of the input molecule if the
    input has undefined stereochemistry, otherwise a copy of the original input is
    returned.
    """
    from openeye import oechem, oeomega

    oe_molecule = oechem.OEMol(oe_molecule)

    # Guess the stereochemistry if it is not already perceived
    unspecified_stereochemistry = any(
        entity.IsChiral() and not entity.HasStereoSpecified()
        for entity in [*oe_molecule.GetAtoms(), *oe_molecule.GetBonds()]
    )

    if unspecified_stereochemistry:
        stereoisomer = next(iter(oeomega.OEFlipper(oe_molecule.GetActive(), 12, True)))
        oe_molecule = oechem.OEMol(stereoisomer)

    return oe_molecule


@requires_oe_package("oechem")
def smiles_to_molecule(
    smiles: str, choose_stereochemistry: bool = False
) -> "oechem.OEMol":
    """Attempts to parse a smiles pattern into a molecule object.

    Parameters
    ----------
    smiles
        The smiles pattern to parse.
    choose_stereochemistry
        If true, the stereochemistry of molecules which is not
        defined in the SMILES pattern will be guessed using the
        OpenEye ``OEFlipper`` utility.

    Returns
    -------
    The parsed molecule.
    """

    from openeye import oechem

    oe_molecule = oechem.OEMol()

    call_openeye(
        oechem.OESmilesToMol,
        oe_molecule,
        smiles,
        exception_type=MoleculeFromSmilesError,
        exception_kwargs={"smiles": smiles},
    )
    call_openeye(
        oechem.OEAddExplicitHydrogens,
        oe_molecule,
        exception_type=MoleculeFromSmilesError,
        exception_kwargs={"smiles": smiles},
    )
    call_openeye(
        oechem.OEPerceiveChiral,
        oe_molecule,
        exception_type=MoleculeFromSmilesError,
        exception_kwargs={"smiles": smiles},
    )

    if choose_stereochemistry:
        oe_molecule = guess_stereochemistry(oe_molecule)

    return oe_molecule


@requires_oe_package("oechem")
@requires_oe_package("oequacpac")
def enumerate_tautomers(
    oe_molecule: "oechem.OEMol",
    max_tautomers: int = 16,
    pka_normalize: bool = True,
) -> List["oechem.OEMol"]:
    """Enumerates the pKa normalized tautomers (up to a specified maximum) of an input
    molecule.
    """
    from openeye import oechem, oequacpac

    oe_molecule = oechem.OEMol(oe_molecule)
    original_smiles = oechem.OEMolToSmiles(oe_molecule)

    tautomer_options = oequacpac.OETautomerOptions()
    tautomer_options.SetMaxTautomersToReturn(max_tautomers)

    tautomers = [
        oechem.OEMol(oe_tautomer)
        for oe_tautomer in oequacpac.OEGetReasonableTautomers(
            oe_molecule, tautomer_options, pka_normalize
        )
    ]

    if original_smiles not in {
        oechem.OEMolToSmiles(tautomer) for tautomer in tautomers
    }:
        tautomers.append(oe_molecule)

    return tautomers


@contextmanager
@requires_oe_package("oechem")
def capture_oe_warnings() -> "oechem.oeosstream":
    from openeye import oechem

    output_stream = oechem.oeosstream()

    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    yield output_stream

    oechem.OEThrow.SetOutputStream(oechem.oeerr)


@requires_oe_package("oechem")
def map_indexed_smiles(smiles_a: str, smiles_b: str) -> Dict[int, int]:
    """Creates a map between the indices of atoms in one indexed SMILES pattern and
    the indices of atoms in another indexed SMILES pattern.

    Args:
        smiles_a: The first indexed SMILES pattern.
        smiles_b: The second indexed SMILES pattern.

    Returns
        A dictionary where each key is the index of an atom in ``smiles_a`` and the
        corresponding value the index of the corresponding atom in ``smiles_b``.

    Examples:

        >>> map_indexed_smiles("[Cl:1][H:2]", "[Cl:2][H:1]")
        {0: 1, 1: 0}
    """

    from openeye import oechem

    oe_original_molecule = oechem.OEGraphMol()
    call_openeye(oechem.OESmilesToMol, oe_original_molecule, smiles_a)

    oe_expected_molecule = oechem.OESubSearch(smiles_b)
    oechem.OEPrepareSearch(oe_original_molecule, oe_expected_molecule)

    match = next(iter(oe_expected_molecule.Match(oe_original_molecule)))

    index_map = {
        atom.target.GetMapIdx() - 1: atom.pattern.GetMapIdx() - 1
        for atom in match.GetAtoms()
    }

    return index_map
