from typing import Generator

from openeye import oechem, oeomega, oequacpac


def guess_stereochemistry(oe_molecule: oechem.OEMol) -> oechem.OEMol:
    """Generates and returns a random stereoisomer of the input molecule if the
    input has undefined stereochemistry, otherwise a copy of the original input is
    returned.
    """

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


def enumerate_tautomers(
    oe_molecule: oechem.OEMol,
) -> Generator[oechem.OEMol, None, None]:
    """Enumerates the pKa normalized tautomers (up to a maximum of 16) of an input
    molecule.
    """
    oe_molecule = oechem.OEMol(oe_molecule)

    tautomer_options = oequacpac.OETautomerOptions()
    tautomer_options.SetMaxTautomersGenerated(16)
    tautomer_options.SetMaxTautomersToReturn(16)
    tautomer_options.SetCarbonHybridization(True)
    tautomer_options.SetMaxZoneSize(50)
    tautomer_options.SetApplyWarts(True)

    return oequacpac.OEGetReasonableTautomers(oe_molecule, tautomer_options, True)
