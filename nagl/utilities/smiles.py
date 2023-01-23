import logging
import typing

from openff.utilities import requires_package

if typing.TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

logger = logging.getLogger(__name__)

T = typing.TypeVar("T")


@requires_package("openff.toolkit")
def smiles_to_molecule(smiles: str, guess_stereo: bool = False) -> "Molecule":
    """Attempts to parse a smiles pattern into a molecule object.

    Parameters
    ----------
    smiles
        The smiles pattern to parse.
    guess_stereo
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

        if not guess_stereo:
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
