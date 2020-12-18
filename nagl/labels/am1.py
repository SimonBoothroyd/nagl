"""Utilities for generating AM1 based labels (namely partial charges and WBO) for sets
of molecules."""
from typing import Optional, Tuple

from openeye import oechem
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openforcefield.topology import Molecule


def compute_am1_charge_and_wbo(
    oe_molecule: oechem.OEMol,
) -> Tuple[Optional[Molecule], Optional[str]]:
    """Computes the AM1ELF10 partial charges and AM1 WBO for the input molecule and
    returns the values stored in an OpenMM molecule object.

    Returns
    -------
        A tuple of the processed molecule containing the AM1 charges and WBO if no
        exceptions were raised (``None`` otherwise) and the error string in an exception
        was raised (``None`` otherwise).
    """

    smiles = None
    error = None

    try:

        smiles = oechem.OEMolToSmiles(oe_molecule)

        # Generate a set of conformers and charges for the molecule.
        conformers = ConformerGenerator.generate(
            oe_molecule,
            ConformerSettings(
                method="omega-elf10", sampling_mode="dense", max_conformers=5
            ),
        )
        charges = ChargeGenerator.generate(
            oe_molecule,
            conformers,
            ChargeSettings(theory="am1", symmetrize=True, optimize=True),
        )

        # Add the charges and conformers to the OE object.
        for oe_atom in oe_molecule.GetAtoms():
            oe_atom.SetPartialCharge(charges[oe_atom.GetIdx()].item())

        oe_molecule.DeleteConfs()

        for conformer in conformers:
            oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

        oechem.OE3DToInternalStereo(oe_molecule)

        # Map to an OpenFF molecule object.
        molecule = Molecule.from_openeye(oe_molecule)

        # Compute the WBOs
        molecule.assign_fractional_bond_orders(
            "am1-wiberg", use_conformers=molecule.conformers
        )

    except (BaseException, Exception) as e:
        molecule = None
        error = f"Failed to process {smiles}: {str(e)}"

    return molecule, error
