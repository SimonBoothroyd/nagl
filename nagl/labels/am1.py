"""Utilities for generating AM1 based labels (namely partial charges and WBO) for sets
of molecules."""
import re
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy

from nagl.utilities.openeye import capture_oe_warnings, requires_oe_package
from nagl.utilities.utilities import requires_package

if TYPE_CHECKING:
    from openeye import oechem
    from openforcefield.topology import Molecule


@requires_oe_package("oechem")
@requires_oe_package("oequacpac")
def compute_wbo(
    oe_molecule: "oechem.OEMol", conformer: numpy.ndarray
) -> Dict[Tuple[int, int], float]:

    from openeye import oechem, oequacpac

    # Set the conformer on the molecule.
    oe_molecule = oechem.OEMol(oe_molecule)
    oe_molecule.DeleteConfs()
    oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

    oechem.OE3DToInternalStereo(oe_molecule)

    am1 = oequacpac.OEAM1()
    am1.GetOptions().SetSemiMethod(oequacpac.OEMethodType_AM1)

    am1_results = oequacpac.OEAM1Results()

    with capture_oe_warnings() as oe_warning_stream:
        status = am1.CalcAM1(am1_results, oe_molecule)

    if not status:
        output_string = oe_warning_stream.str().decode("UTF-8").replace("Warning: ", "")
        output_string = re.sub("^: +", "", output_string, flags=re.MULTILINE)
        output_string = re.sub("\n$", "", output_string)

        raise ValueError(f"Unable to compute the WBO: {output_string}")

    bond_orders = {}

    for bond in oe_molecule.GetBonds():

        bond_indices = (
            min(bond.GetBgnIdx(), bond.GetEndIdx()),
            max(bond.GetBgnIdx(), bond.GetEndIdx()),
        )
        bond_orders[bond_indices] = am1_results.GetBondOrder(*bond_indices)

    return bond_orders


@requires_package("openff.recharge")
def compute_am1_charge_and_wbo(
    oe_molecule: "oechem.OEMol", wbo_method: str
) -> Tuple[Optional["Molecule"], Optional[str]]:
    """Computes the AM1ELF10 partial charges and AM1 WBO for the input molecule and
    returns the values stored in an OpenMM molecule object.

    Args:
        oe_molecule: The molecule to compute charges and WBOs for.
        wbo_method: The method by which to compute the WBOs. WBOs may be computed using
            only a single conformer ("single-conformer"), or by computing the WBO for
            each ELF10 conformer and taking the average ("elf10-average").

    Returns
    -------
        A tuple of the processed molecule containing the AM1 charges and WBO if no
        exceptions were raised (``None`` otherwise) and the error string in an exception
        was raised (``None`` otherwise).
    """

    from openeye import oechem
    from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
    from openff.recharge.conformers import ConformerGenerator, ConformerSettings
    from openforcefield.topology import Molecule

    smiles = None
    error = None

    try:

        smiles = oechem.OEMolToSmiles(oe_molecule)

        # Generate a set of conformers and charges for the molecule.
        conformers = ConformerGenerator.generate(
            oe_molecule,
            ConformerSettings(
                method="omega-elf10", sampling_mode="dense", max_conformers=None
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

        # Map to an OpenFF molecule object.
        molecule = Molecule.from_openeye(oe_molecule)

        # Compute the WBOs
        wbo_conformers = (
            conformers if wbo_method == "elf10-average" else [conformers[0]]
        )
        wbo_per_conformer = [
            compute_wbo(oe_molecule, conformer) for conformer in wbo_conformers
        ]

        for bond in molecule.bonds:

            bond_order = (
                sum(
                    conformer_wbo[
                        (
                            min(bond.atom1_index, bond.atom2_index),
                            max(bond.atom1_index, bond.atom2_index),
                        )
                    ]
                    for conformer_wbo in wbo_per_conformer
                )
                / len(wbo_conformers)
            )

            bond.fractional_bond_order = bond_order

    except (BaseException, Exception) as e:
        molecule = None
        error = f"Failed to process {smiles}: {str(e)}"

    return molecule, error
