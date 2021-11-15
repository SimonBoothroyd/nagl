import logging
from typing import TYPE_CHECKING, Dict, TypeVar

from openff.utilities import requires_package

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

logger = logging.getLogger(__name__)

T = TypeVar("T")


@requires_package("openff.toolkit")
def smiles_to_molecule(smiles: str, guess_stereochemistry: bool = False) -> "Molecule":
    """Attempts to parse a smiles pattern into a molecule object.

    Parameters
    ----------
    smiles
        The smiles pattern to parse.
    guess_stereochemistry
        If true, the stereochemistry of molecules which is not defined in the SMILES
        pattern will be guessed by enumerating possible stereoisomers and selecting
        the first one in the list.

    Returns
    -------
    The parsed molecule.
    """
    from openff.toolkit.topology import Molecule
    from openff.toolkit.utils import UndefinedStereochemistryError

    try:
        molecule = Molecule.from_smiles(smiles)
    except UndefinedStereochemistryError:

        if not guess_stereochemistry:
            raise

        molecule: Molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        stereoisomers = molecule.enumerate_stereoisomers(
            undefined_only=True, max_isomers=1
        )

        if len(stereoisomers) > 0:
            # We would ideally raise an exception here if the number of stereoisomers
            # is zero, however due to the way that the OFF toolkit perceives pyramidal
            # nitrogen stereocenters these would show up as undefined stereochemistry
            # but have no enumerated stereoisomers.
            molecule = stereoisomers[0]

    return molecule


@requires_package("openff.toolkit")
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

    from openff.toolkit.topology import Molecule

    original_molecule: Molecule = Molecule.from_mapped_smiles(smiles_a)
    expected_molecule: Molecule = Molecule.from_mapped_smiles(smiles_b)

    _, index_map = Molecule.are_isomorphic(
        original_molecule, expected_molecule, return_atom_map=True
    )

    return index_map
