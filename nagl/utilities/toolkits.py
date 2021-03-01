"""A module for handling cheminformatics toolkit calls directly which are not yet
available in the OpenFF toolkit.
"""
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

from nagl.utilities import MissingOptionalDependency, requires_package

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


@requires_package("openeye.oechem")
@requires_package("openff.toolkit")
def _oe_stream_from_file(
    file_path: str,
) -> Generator["Molecule", None, None]:  # pragma: no cover

    from openeye import oechem
    from openff.toolkit.topology import Molecule

    input_molecule_stream = oechem.oemolistream()
    input_molecule_stream.open(file_path)

    for oe_molecule in input_molecule_stream.GetOEMols():
        yield Molecule.from_openeye(oe_molecule, allow_undefined_stereo=True)


@requires_package("rdkit")
@requires_package("openff.toolkit")
def _rdkit_stream_from_file(file_path: str) -> Generator["Molecule", None, None]:

    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    for rd_molecule in Chem.SupplierFromFilename(
        file_path, removeHs=False, sanitize=False, strictParsing=True
    ):

        if rd_molecule is None:
            continue

        Chem.AddHs(rd_molecule)

        yield Molecule.from_rdkit(rd_molecule, allow_undefined_stereo=True)


def stream_from_file(file_path: str) -> Generator["Molecule", None, None]:

    try:
        for molecule in _oe_stream_from_file(file_path):
            yield molecule
    except MissingOptionalDependency:
        for molecule in _rdkit_stream_from_file(file_path):
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

    def _save_molecule(molecule: Molecule):
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
        yield _oe_capture_warnings()
    except MissingOptionalDependency:
        yield

    logging.getLogger("openff.toolkit").setLevel(openff_logger_level)
