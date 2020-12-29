import numpy
from openforcefield.topology import Molecule
from simtk import unit

from nagl.labels.am1 import compute_am1_charge_and_wbo, compute_wbo
from nagl.utilities.openeye import requires_oe_package


@requires_oe_package("oechem")
def test_compute_wbo(methane: Molecule):

    oe_molecule = methane.to_openeye()
    conformer = methane.conformers[0].value_in_unit(unit.angstrom)

    wbo_per_bond = compute_wbo(oe_molecule, conformer)

    assert len(wbo_per_bond) == 4

    assert all(bond in wbo_per_bond for bond in [(0, i) for i in range(1, 5)])
    assert all(numpy.isclose(x, 0.9878536626) for x in wbo_per_bond.values())


@requires_oe_package("oechem")
def test_compute_am1_charge_and_wbo(methane: Molecule):

    labelled_molecule, error = compute_am1_charge_and_wbo(
        methane.to_openeye(), "single-conformer"
    )

    assert labelled_molecule is not None
    assert error is None

    assert not numpy.allclose(
        labelled_molecule.partial_charges.value_in_unit(unit.elementary_charge), 0.0
    )
    assert all(
        numpy.isclose(bond.fractional_bond_order, 0.9878536626)
        for bond in labelled_molecule.bonds
    )
