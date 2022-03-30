"""A module for handling cheminformatics toolkit calls directly which are not yet
available in the OpenFF toolkit.
"""
import json
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, List, Literal, overload

from openff.utilities import requires_package
from openff.utilities.utilities import MissingOptionalDependency

from nagl import data

if TYPE_CHECKING:

    from openff.toolkit.topology import Molecule


@overload
def _oe_stream_from_file(
    file_path: str, as_smiles: Literal[True] = True
) -> Generator[str, None, None]:  # pragma: no cover
    ...


@overload
def _oe_stream_from_file(
    file_path: str, as_smiles: Literal[False] = False
) -> Generator["Molecule", None, None]:  # pragma: no cover
    ...


@requires_package("openeye.oechem")
@requires_package("openff.toolkit")
def _oe_stream_from_file(file_path: str, as_smiles=False):  # pragma: no cover

    from openeye import oechem
    from openff.toolkit.topology import Molecule

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(file_path)

    for oe_molecule in input_molecule_stream.GetOEMols():

        yield (
            oechem.OEMolToSmiles(oe_molecule)
            if as_smiles
            else Molecule.from_openeye(oe_molecule, allow_undefined_stereo=True)
        )


@overload
def _rdkit_stream_from_file(
    file_path: str, as_smiles: Literal[True] = True
) -> Generator[str, None, None]:  # pragma: no cover
    ...


@overload
def _rdkit_stream_from_file(
    file_path: str, as_smiles: Literal[False] = False
) -> Generator["Molecule", None, None]:  # pragma: no cover
    ...


@requires_package("rdkit")
@requires_package("openff.toolkit")
def _rdkit_stream_from_file(file_path: str, as_smiles=False):

    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    for rd_molecule in Chem.SupplierFromFilename(
        file_path, removeHs=False, sanitize=True, strictParsing=True
    ):

        if rd_molecule is None:
            continue

        Chem.AddHs(rd_molecule)

        yield Chem.MolToSmiles(rd_molecule) if as_smiles else Molecule.from_rdkit(
            rd_molecule, allow_undefined_stereo=True
        )


@overload
def stream_from_file(
    file_path: str, as_smiles: Literal[True] = True
) -> Generator[str, None, None]:  # pragma: no cover
    ...


@overload
def stream_from_file(
    file_path: str, as_smiles: Literal[False] = False
) -> Generator["Molecule", None, None]:  # pragma: no cover
    ...


def stream_from_file(file_path: str, as_smiles: bool = False):

    try:
        for molecule in _oe_stream_from_file(file_path, as_smiles):
            yield molecule
    except MissingOptionalDependency:
        for molecule in _rdkit_stream_from_file(file_path, as_smiles):
            yield molecule


@requires_package("openeye.oechem")
@requires_package("openff.toolkit")
@contextmanager
def _oe_stream_to_file(file_path: str):  # pragma: no cover

    from openeye import oechem
    from openff.toolkit.topology import Molecule

    output_molecule_stream = oechem.oemolostream(file_path)

    def _save_molecule(molecule: Molecule):
        oechem.OEWriteMolecule(output_molecule_stream, molecule.to_openeye())

    yield _save_molecule

    output_molecule_stream.close()


@requires_package("rdkit")
@contextmanager
def _rdkit_stream_to_file(file_path: str):

    from rdkit import Chem

    output_molecule_stream = Chem.SDWriter(file_path)

    def _save_molecule(molecule):
        output_molecule_stream.write(molecule.to_rdkit())
        output_molecule_stream.flush()

    yield _save_molecule

    output_molecule_stream.close()


@contextmanager
def stream_to_file(file_path: str):

    try:
        with _oe_stream_to_file(file_path) as writer:
            yield writer
    except MissingOptionalDependency:
        with _rdkit_stream_to_file(file_path) as writer:
            yield writer


@contextmanager
@requires_package("openeye.oechem")
def _oe_capture_warnings():  # pragma: no cover

    from openeye import oechem

    output_stream = oechem.oeosstream()

    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    yield

    oechem.OEThrow.SetOutputStream(oechem.oeerr)


@contextmanager
def capture_toolkit_warnings():  # pragma: no cover
    """A convenience method to capture and discard any warning produced by external
    cheminformatics toolkits excluding the OpenFF toolkit. This should be used with
    extreme caution and is only really intended for use when processing tens of
    thousands of molecules at once."""

    import logging
    import warnings

    warnings.filterwarnings("ignore")

    openff_logger_level = logging.getLogger("openff.toolkit").getEffectiveLevel()
    logging.getLogger("openff.toolkit").setLevel(logging.ERROR)

    try:
        with _oe_capture_warnings():
            yield
    except MissingOptionalDependency:
        yield

    logging.getLogger("openff.toolkit").setLevel(openff_logger_level)


@requires_package("openeye.oechem")
def _oe_smiles_to_inchi_key(smiles: str) -> str:  # pragma: no cover

    from openeye import oechem

    oe_molecule = oechem.OEMol()
    oechem.OESmilesToMol(oe_molecule, smiles)

    opts = oechem.OEInChIOptions()
    opts.SetFixedHLayer(True)

    return oechem.OEMolToInChIKey(oe_molecule)


@requires_package("rdkit")
def _rdkit_smiles_to_inchi_key(smiles: str) -> str:

    from rdkit import Chem

    rd_molecule = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(rd_molecule, options="-FixedH")


def smiles_to_inchi_key(smiles: str) -> str:

    try:  # pragma: no cover
        return _oe_smiles_to_inchi_key(smiles)
    except MissingOptionalDependency:
        return _rdkit_smiles_to_inchi_key(smiles)


def _oe_get_atom_symmetries(molecule: "Molecule") -> List[int]:  # pragma: no cover

    from openeye import oechem

    oe_molecule = molecule.to_openeye()
    oechem.OEPerceiveSymmetry(oe_molecule)

    symmetry_classes_by_index = {
        a.GetIdx(): a.GetSymmetryClass() for a in oe_molecule.GetAtoms()
    }
    return [symmetry_classes_by_index[i] for i in range(molecule.n_atoms)]


def _rd_get_atom_symmetries(molecule: "Molecule") -> List[int]:

    from rdkit import Chem

    rd_mol = molecule.to_rdkit()
    return list(Chem.CanonicalRankAtoms(rd_mol, breakTies=False))


def get_atom_symmetries(molecule: "Molecule") -> List[int]:

    from openff.toolkit.utils import ToolkitUnavailableException

    try:  # pragma: no cover
        return _oe_get_atom_symmetries(molecule)
    except (ImportError, ModuleNotFoundError, ToolkitUnavailableException):
        return _rd_get_atom_symmetries(molecule)


def _oe_normalize_molecule(
    molecule: "Molecule", reaction_smarts: List[str]
) -> "Molecule":  # pragma: no cover

    from openeye import oechem
    from openff.toolkit.topology import Molecule

    oe_molecule: oechem.OEMol = molecule.to_openeye()

    for pattern in reaction_smarts:

        reaction = oechem.OEUniMolecularRxn(pattern)
        reaction(oe_molecule)

    return Molecule.from_openeye(oe_molecule, allow_undefined_stereo=True)


def _rd_normalize_molecule(
    molecule: "Molecule", reaction_smarts: List[str], max_iterations=10000
) -> "Molecule":

    from openff.toolkit.topology import Molecule
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions

    rd_molecule: Chem.Mol = molecule.to_rdkit()

    original_smiles = Chem.MolToSmiles(rd_molecule)
    old_smiles = original_smiles

    for pattern in reaction_smarts:

        reaction = rdChemReactions.ReactionFromSmarts(pattern)
        n_iterations = 0

        while True:

            products = reaction.RunReactants((rd_molecule,), maxProducts=1)

            if len(products) == 0:
                break

            ((rd_molecule,),) = products
            new_smiles = Chem.MolToSmiles(rd_molecule)

            has_changed = old_smiles != new_smiles
            old_smiles = new_smiles

            if not has_changed:
                break

            n_iterations += 1
            assert (
                n_iterations <= max_iterations
            ), f"could not normalize {original_smiles}"

    return Molecule.from_rdkit(rd_molecule, allow_undefined_stereo=True)


def normalize_molecule(molecule: "Molecule", check_output: bool = True) -> "Molecule":
    """Applies a set of reaction SMARTS in sequence to an input molecule in order to
    attempt to 'normalize' its structure.

    This involves, for example, converting ``-N(=O)=O`` groups to ``-N(=O)[O-]`` and
    ``-[S+2]([O-])([O-])-`` to ``-S(=O)=O-``. See ``nagl/data/normalizations.json`` for
    a full list of transforms.

    Args:
        molecule: The molecule to normalize.
        check_output: Whether to make sure the normalized molecule is isomorphic with
            the input molecule, ignoring aromaticity, bond order, formal charge, and
            stereochemistry.
    """

    from openff.toolkit.topology import Molecule
    from openff.toolkit.utils import ToolkitUnavailableException

    reaction_smarts_path = data.get_file_path("normalizations.json")

    with open(reaction_smarts_path) as file:
        reaction_smarts = [entry["smarts"] for entry in json.load(file)]

    try:  # pragma: no cover
        # normal_molecule = _oe_normalize_molecule(molecule, reaction_smarts)
        raise NotImplementedError()
    except (
        ImportError,
        ModuleNotFoundError,
        ToolkitUnavailableException,
        NotImplementedError,
    ):
        normal_molecule = _rd_normalize_molecule(molecule, reaction_smarts)

    assert (
        not check_output
        or Molecule.are_isomorphic(
            molecule,
            normal_molecule,
            aromatic_matching=False,
            formal_charge_matching=False,
            bond_order_matching=False,
            atom_stereochemistry_matching=False,
            bond_stereochemistry_matching=False,
        )[0]
    ), "normalization changed the molecule - this should not happen"

    return normal_molecule
