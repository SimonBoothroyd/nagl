import copy
import traceback
from typing import TYPE_CHECKING, List, Optional, Tuple, Union, get_args

from openff.utilities import requires_package
from tqdm import tqdm

from nagl.storage import (
    ChargeMethod,
    ConformerRecord,
    MoleculeRecord,
    PartialChargeSet,
    WBOMethod,
    WibergBondOrderSet,
)
from nagl.utilities.smiles import smiles_to_molecule
from nagl.utilities.toolkits import capture_toolkit_warnings

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

_OPENFF_CHARGE_METHODS = {"am1": "am1-mulliken", "am1bcc": "am1bcc"}
_OPENFF_WBO_METHODS = {"am1": "am1-wiberg"}


@requires_package("openff.toolkit")
def label_molecule(
    molecule: Union[str, "Molecule"],
    guess_stereochemistry: bool,
    partial_charge_methods: List[ChargeMethod],
    bond_order_methods: List[WBOMethod],
    n_conformers: int = 500,
    rms_cutoff: float = 0.05,
) -> MoleculeRecord:
    """Computes sets of partial charges and bond orders for an input molecule.

    Notes:
        Conformers will be pruned using the ELF10 method provided by the OpenFF toolkit

    Args:
        molecule: The molecule (or SMILES representation of the molecule) to label.
        guess_stereochemistry: Whether to guess the stereochemistry of the SMILES
            representation of the molecule if provided and if the stereochemistry of
            some atoms / bonds is not fully defined.
        partial_charge_methods: The methods to compute the partial charges using.
        bond_order_methods: The methods to compute the bond orders using.
        n_conformers: The *maximum* number of conformers to compute partial charge and
            bond orders using.
        rms_cutoff: The RMS cutoff [Å] to use when generating the conformers.

    Returns:
        The labelled molecule stored in a record object
    """

    from openff.units import unit

    if isinstance(molecule, str):

        molecule = smiles_to_molecule(
            molecule, guess_stereochemistry=guess_stereochemistry
        )

    else:
        molecule = copy.deepcopy(molecule)

    # Generate a diverse set of ELF10 conformers
    molecule.generate_conformers(
        n_conformers=n_conformers, rms_cutoff=rms_cutoff * unit.angstrom
    )
    molecule.apply_elf_conformer_selection()

    conformer_records = []

    for conformer in molecule.conformers:

        charge_sets = []

        for charge_method in partial_charge_methods:

            molecule.assign_partial_charges(
                _OPENFF_CHARGE_METHODS[charge_method], use_conformers=[conformer]
            )

            charge_sets.append(
                PartialChargeSet(
                    method=charge_method,
                    values=[
                        atom.partial_charge.m_as(unit.elementary_charge)
                        for atom in molecule.atoms
                    ],
                )
            )

        bond_order_sets = []

        for bond_order_method in bond_order_methods:

            molecule.assign_fractional_bond_orders(
                _OPENFF_WBO_METHODS[bond_order_method], use_conformers=[conformer]
            )

            bond_order_sets.append(
                WibergBondOrderSet(
                    method=bond_order_method,
                    values=[
                        (
                            bond.atom1_index,
                            bond.atom2_index,
                            bond.fractional_bond_order,
                        )
                        for bond in molecule.bonds
                    ],
                )
            )

        conformer_records.append(
            ConformerRecord(
                coordinates=conformer.m_as(unit.angstrom),
                partial_charges=charge_sets,
                bond_orders=bond_order_sets,
            )
        )

    return MoleculeRecord(
        smiles=molecule.to_smiles(isomeric=True, mapped=True),
        conformers=conformer_records,
    )


@requires_package("openff.toolkit")
def label_molecules(
    molecules: List[Union[str, "Molecule"]],
    guess_stereochemistry: bool,
    partial_charge_methods: Optional[List[ChargeMethod]],
    bond_order_methods: Optional[List[WBOMethod]],
    n_conformers: int = 500,
    rms_cutoff: float = 0.05,
) -> List[Tuple[Optional[MoleculeRecord], Optional[str]]]:
    """Labels a batch of molecules using ``label_molecule``.

    Args:
        molecules: A list of the molecule (or SMILES representation of the molecules) to
            label.
        guess_stereochemistry: Whether to guess the stereochemistry of the SMILES
            representation of the molecule if provided and if the stereochemistry of
            some atoms / bonds is not fully defined.
        partial_charge_methods: The methods to compute the partial charges using. By
            default, all available methods will be used.
        bond_order_methods: The methods to compute the bond orders using. By
            default, all available methods will be used.
        n_conformers: The *maximum* number of conformers to compute partial charge and
            bond orders using.
        rms_cutoff: The RMS cutoff [Å] to use when generating the conformers.

    Returns:
        A list of tuples of the form ``(labelled_record, error_message)``.
    """

    partial_charge_methods = (
        partial_charge_methods
        if partial_charge_methods is not None
        else get_args(ChargeMethod)
    )
    bond_order_methods = (
        bond_order_methods if bond_order_methods is not None else get_args(WBOMethod)
    )

    molecule_records = []

    with capture_toolkit_warnings():

        for molecule in tqdm(molecules, ncols=80, desc="labelling batch"):

            molecule_record = None
            error = None

            try:
                molecule_record = label_molecule(
                    molecule,
                    guess_stereochemistry,
                    partial_charge_methods,
                    bond_order_methods,
                    n_conformers,
                    rms_cutoff,
                )
            except (BaseException, Exception) as e:

                formatted_traceback = traceback.format_exception(
                    type(e), e, e.__traceback__
                )
                error = f"Failed to process {str(molecule)}: {formatted_traceback}"

            molecule_records.append((molecule_record, error))

    return molecule_records
