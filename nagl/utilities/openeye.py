import functools
from contextlib import contextmanager
from typing import TYPE_CHECKING, List

from typing_extensions import Literal

from nagl.utilities.utilities import MissingOptionalDependency, requires_package

if TYPE_CHECKING:
    from openeye import oechem


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
