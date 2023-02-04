"""Utilities for manipulating RDKit molecule objects."""
import contextlib
import pathlib
import typing

from rdkit import Chem
from rdkit.Chem import EnumerateStereoisomers, rdChemReactions

from nagl.utilities.normalization import NORMALIZATION_SMARTS

BOND_TYPE_TO_ORDER = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
}
BOND_ORDER_TO_TYPE = {order: type_ for type_, order in BOND_TYPE_TO_ORDER.items()}


def stream_from_file(
    file_path: pathlib.Path, as_smiles=False
) -> typing.Union[str, Chem.Mol]:
    """Yields all molecules in an SDF file.

    Args:
        file_path: The path to the SDF file to read.
        as_smiles: Whether to return the molecules as their SMILES representation
            or as a molecule object.
    """

    for molecule in Chem.SupplierFromFilename(
        str(file_path), removeHs=False, sanitize=True, strictParsing=True
    ):
        if molecule is None:
            continue

        molecule = Chem.AddHs(molecule)

        yield Chem.MolToSmiles(molecule) if as_smiles else molecule


@contextlib.contextmanager
def stream_to_file(file_path: pathlib.Path) -> typing.Callable[[Chem.Mol], None]:
    """A context manager that yields a function that will write a molecule to the
    specified SDF file path.

    Args:
        file_path: The path to the SDF file to write to.
    """

    output_molecule_stream = Chem.SDWriter(str(file_path))

    def _save_molecule(molecule: Chem.Mol):
        output_molecule_stream.write(molecule)
        output_molecule_stream.flush()

    yield _save_molecule

    output_molecule_stream.close()


def normalize_molecule(
    molecule: Chem.Mol,
    reaction_smarts: typing.Optional[typing.List[str]] = None,
    max_iterations=10000,
) -> Chem.Mol:
    """Applies a set of reaction SMARTS in sequence to an input molecule in order to
    attempt to 'normalize' its structure.

    This involves, for example, converting ``-N(=O)=O`` groups to ``-N(=O)[O-]`` and
    ``-[S+2]([O-])([O-])-`` to ``-S(=O)=O-``.

    Args:
        molecule: The molecule to normalize.
        reaction_smarts: The list of SMARTS that define the normalizations to perform.
            If no list is provided, the default set defined by
            ``nagl.utilities.normalization.NORMALIZATION_SMARTS`` will be used.
        max_iterations: The maximum number of attempts to apply a SMARTS pattern if
            the molecule continues to change after application.
    """

    reaction_smarts = (
        reaction_smarts
        if reaction_smarts is not None
        else [entry["smarts"] for entry in NORMALIZATION_SMARTS]
    )

    molecule = Chem.Mol(molecule)

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    original_smiles = Chem.MolToSmiles(molecule)

    old_smiles = original_smiles

    for pattern in reaction_smarts:
        reaction = rdChemReactions.ReactionFromSmarts(pattern)
        n_iterations = 0

        while True:
            products = reaction.RunReactants((molecule,), maxProducts=1)

            if len(products) == 0:
                break

            ((molecule,),) = products

            for atom in molecule.GetAtoms():
                atom.SetAtomMapNum(atom.GetIntProp("react_atom_idx") + 1)

            new_smiles = Chem.MolToSmiles(Chem.AddHs(molecule))

            has_changed = old_smiles != new_smiles
            old_smiles = new_smiles

            if not has_changed:
                break

            n_iterations += 1
            assert (
                n_iterations <= max_iterations
            ), f"could not normalize {original_smiles}"

    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)

    return molecule


def molecule_from_smiles(smiles: str, guess_stereo: bool = True) -> Chem.Mol:
    """Attempts to parse a smiles pattern into a molecule object.

    Args:
        smiles: The smiles pattern to parse.
        guess_stereo: If true, any missing stereochemistry will be guessed.

    Returns:
        The parsed molecule.
    """

    opts = Chem.SmilesParserParams()
    opts.removeHs = False

    molecule = Chem.MolFromSmiles(smiles, opts)

    if molecule is None:
        raise ValueError(f"could not parse {smiles}")

    molecule = Chem.AddHs(molecule)

    if guess_stereo:
        Chem.AssignStereochemistry(
            molecule, cleanIt=True, force=True, flagPossibleStereoCenters=True
        )
        Chem.FindPotentialStereoBonds(molecule)

        stereo_opts = EnumerateStereoisomers.StereoEnumerationOptions(
            maxIsomers=1, onlyUnassigned=True
        )

        isomers = list(
            EnumerateStereoisomers.EnumerateStereoisomers(molecule, options=stereo_opts)
        )
        molecule = molecule if len(isomers) == 0 else isomers[0]

    return molecule


def molecule_from_mapped_smiles(smiles: str) -> Chem.Mol:
    """Loads a molecule from a mapped SMILES string, using atom map indices as the
    atom indices.

    Args:
        smiles: The mapped SMILES pattern.

    Returns:
        The molecule object with the correct atom ordering.
    """

    opts = Chem.SmilesParserParams()
    opts.removeHs = False

    mol = Chem.AddHs(Chem.MolFromSmiles(smiles, opts))

    idx_map = {atom.GetAtomMapNum() - 1: atom.GetIdx() for atom in mol.GetAtoms()}

    if any(idx < 0 for idx in idx_map):
        raise ValueError("all atoms must have a map index")

    new_order = [idx_map[i] for i in range(mol.GetNumAtoms())]

    mol = Chem.RenumberAtoms(mol, new_order)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    return mol


def molecule_to_mapped_smiles(molecule: Chem.Mol) -> str:
    """Creates a fully mapped SMILES pattern from a molecule object.

    Args:
        molecule: The molecule to map to a SMILES pattern.

    Returns:
        The fully mapped SMILES representation of the molecule.
    """

    molecule = Chem.Mol(molecule)
    atom: Chem.Atom

    for i, atom in enumerate(molecule.GetAtoms()):
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    return Chem.MolToSmiles(molecule, allHsExplicit=True)
