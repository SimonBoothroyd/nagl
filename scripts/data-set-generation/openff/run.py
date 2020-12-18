import pickle
from multiprocessing.pool import Pool
from typing import Generator, Optional, Tuple

from openeye import oechem, oequacpac
from openff.recharge.charges.charges import ChargeGenerator, ChargeSettings
from openff.recharge.conformers import ConformerGenerator, ConformerSettings
from openff.recharge.utilities.openeye import smiles_to_molecule
from openforcefield.topology import Molecule
from tqdm import tqdm


def enumerate_tautomers(oe_molecule: oechem.OEMol) -> Generator[oechem.OEMol]:

    tautomer_options = oequacpac.OETautomerOptions()
    tautomer_options.SetMaxTautomersGenerated(4096)
    tautomer_options.SetMaxTautomersToReturn(16)
    tautomer_options.SetCarbonHybridization(True)
    tautomer_options.SetMaxZoneSize(50)
    tautomer_options.SetApplyWarts(True)

    return oequacpac.OEGetReasonableTautomers(oe_molecule, tautomer_options, True)


def process_molecule(smiles: str) -> Tuple[Optional[Molecule], Optional[str]]:

    error = None

    try:
        oe_molecule = smiles_to_molecule(smiles, guess_stereochemistry=True)

        # Generate a set of 5 ELF conformers and charges for the molecule.
        conformers = ConformerGenerator.generate(oe_molecule, ConformerSettings())
        charges = ChargeGenerator.generate(oe_molecule, conformers, ChargeSettings())

        # Add the charges and conformers to the OE object.
        for oe_atom in oe_molecule.GetAtoms():
            oe_atom.SetPartialCharge(charges[oe_atom.GetIdx()].item())

        oe_molecule.DeleteConfs()

        for conformer in conformers:
            oe_molecule.NewConf(oechem.OEFloatArray(conformer.flatten()))

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


def main():

    n_processes = 4

    # Load in the train and test sets.
    for label in ["train"]:

        with open(f"{label}-smiles-scaffold.txt") as file:
            smiles_patterns = file.read().split("\n")

        with Pool(processes=n_processes) as pool:
            processed_molecules = list(
                tqdm(
                    pool.imap(process_molecule, smiles_patterns),
                    total=len(smiles_patterns),
                )
            )

        # Retain only the molecules which could be processed and print errors
        # to a log file.
        molecules = []

        with open("../errors.log", "w") as file:

            for molecule, error in processed_molecules:

                if error is None:

                    molecules.append(molecule)
                    continue

                file.write("".join(["="] * 80) + "\n")
                file.write(error + "\n")
                file.write("".join(["="] * 80) + "\n")

        with open(f"{label}-set.pkl", "wb") as file:
            pickle.dump(molecules, file)


if __name__ == "__main__":
    main()
