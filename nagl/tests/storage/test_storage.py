import re

import numpy
import pytest
from pydantic import ValidationError

from nagl.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
    WibergBondOrderSet,
)
from nagl.storage.db import DB_VERSION, DBConformerRecord, DBInformation
from nagl.storage.exceptions import IncompatibleDBVersion
from nagl.tests import does_not_raise


@pytest.fixture()
def tmp_molecule_store(tmp_path) -> MoleculeStore:

    store = MoleculeStore(f"{tmp_path}.sqlite")

    expected_records = [
        MoleculeRecord(
            smiles="[Ar:1]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[0.0, 0.0, 0.0]]),
                    partial_charges=[PartialChargeSet(method="am1", values=[0.5])],
                    bond_orders=[],
                )
            ],
        ),
        MoleculeRecord(
            smiles="[He:1]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[0.0, 0.0, 0.0]]),
                    partial_charges=[PartialChargeSet(method="am1bcc", values=[-0.5])],
                    bond_orders=[],
                )
            ],
        ),
        MoleculeRecord(
            smiles="[Cl:1][Cl:2]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    partial_charges=[
                        PartialChargeSet(method="am1", values=[0.5, -0.5]),
                        PartialChargeSet(method="am1bcc", values=[0.75, -0.75]),
                    ],
                    bond_orders=[
                        WibergBondOrderSet(method="am1", values=[(0, 1, 1.2)])
                    ],
                )
            ],
        ),
    ]

    store.store(*expected_records)

    return store


class TestPartialChargeSet:
    def test_tuple_coercion(self):
        """Ensure that list types are coerced to immutable tuples"""
        charge_set = PartialChargeSet(method="am1", values=[0.1])
        assert isinstance(charge_set.values, tuple)


class TestWibergBondOrderSet:
    def test_tuple_coercion(self):
        """Ensure that list types are coerced to immutable tuples"""
        wbo_set = WibergBondOrderSet(
            method="am1",
            values=[
                (0, 1, 0.1),
            ],
        )
        assert isinstance(wbo_set.values, tuple)


class TestConformerRecord:
    def test_partial_charges_by_method(self):

        record = ConformerRecord(
            coordinates=numpy.ones((4, 3)),
            partial_charges=[
                PartialChargeSet(method="am1", values=[0.1, 0.2, 0.3, 0.4]),
                PartialChargeSet(method="am1bcc", values=[1.0, 2.0, 3.0, 4.0]),
            ],
        )

        assert record.partial_charges_by_method == {
            "am1": (0.1, 0.2, 0.3, 0.4),
            "am1bcc": (1.0, 2.0, 3.0, 4.0),
        }

    def test_bond_orders_by_method(self):

        record = ConformerRecord(
            coordinates=numpy.ones((2, 3)),
            bond_orders=[WibergBondOrderSet(method="am1", values=[(0, 1, 0.1)])],
        )

        assert record.bond_orders_by_method == {"am1": ((0, 1, 0.1),)}

    @pytest.mark.parametrize(
        "value, expected_raises",
        [
            (numpy.arange(6), does_not_raise()),
            (numpy.arange(6).reshape((-1, 3)), does_not_raise()),
            (
                numpy.arange(5),
                pytest.raises(ValidationError, match="coordinates must be re-shapable"),
            ),
            (
                numpy.arange(4).reshape((-1, 2)),
                pytest.raises(ValidationError, match="coordinates must be re-shapable"),
            ),
        ],
    )
    def test_validate_coordinates(self, value, expected_raises):

        with expected_raises:
            record = ConformerRecord(coordinates=value)

            assert isinstance(record.coordinates, numpy.ndarray)
            assert record.coordinates.flags.writeable is False

    @pytest.mark.parametrize(
        "value, expected_raises",
        [
            (tuple(), does_not_raise()),
            ((PartialChargeSet(method="am1", values=(0.1, 0.2)),), does_not_raise()),
            (
                (
                    PartialChargeSet(method="am1", values=(0.1, 0.2)),
                    PartialChargeSet(method="am1", values=(0.1, 0.2)),
                ),
                pytest.raises(
                    ValidationError,
                    match="multiple charge sets computed using the same method",
                ),
            ),
            (
                (PartialChargeSet(method="am1", values=(0.1,)),),
                pytest.raises(
                    ValidationError, match="the number of partial charges must match"
                ),
            ),
        ],
    )
    def test_validate_partial_charges(self, value, expected_raises):

        with expected_raises:

            record = ConformerRecord(
                coordinates=numpy.ones((2, 3)), partial_charges=value
            )

            assert isinstance(record.partial_charges, tuple)
            assert len(record.partial_charges) == len(value)

    @pytest.mark.parametrize(
        "value, expected_raises",
        [
            (tuple(), does_not_raise()),
            (
                (WibergBondOrderSet(method="am1", values=((0, 1, 0.1),)),),
                does_not_raise(),
            ),
            (
                (
                    WibergBondOrderSet(method="am1", values=((0, 1, 0.1),)),
                    WibergBondOrderSet(method="am1", values=((0, 1, 0.1),)),
                ),
                pytest.raises(
                    ValidationError,
                    match="multiple bond order sets computed using the same method",
                ),
            ),
        ],
    )
    def test_validate_bond_orders(self, value, expected_raises):
        with expected_raises:
            record = ConformerRecord(coordinates=numpy.ones((2, 3)), bond_orders=value)

            assert isinstance(record.bond_orders, tuple)
            assert len(record.bond_orders) == len(value)


class TestMoleculeRecord:
    def test_average_partial_charges(self):

        record = MoleculeRecord(
            smiles="[C:1]([H:2])([H:3])([H:4])",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.ones((4, 3)),
                    partial_charges=[
                        PartialChargeSet(method="am1", values=[0.1, 0.2, 0.3, 0.4]),
                    ],
                ),
                ConformerRecord(
                    coordinates=numpy.zeros((4, 3)),
                    partial_charges=[
                        PartialChargeSet(method="am1", values=[0.3, 0.4, 0.5, 0.6]),
                    ],
                ),
            ],
        )

        average_charges = record.average_partial_charges("am1")

        assert isinstance(average_charges, tuple)
        assert len(average_charges) == 4

        assert numpy.allclose(average_charges, (0.2, 0.3, 0.4, 0.5))

    def test_reorder(self):

        original_coordinates = numpy.arange(6).reshape((2, 3))

        original_record = MoleculeRecord(
            smiles="[Cl:2][H:1]",
            conformers=[
                ConformerRecord(
                    coordinates=original_coordinates,
                    partial_charges=[PartialChargeSet(method="am1", values=[0.5, 1.5])],
                    bond_orders=[
                        WibergBondOrderSet(method="am1", values=[(0, 1, 0.2)])
                    ],
                )
            ],
        )

        reordered_record = original_record.reorder("[Cl:1][H:2]")
        assert reordered_record.smiles == "[Cl:1][H:2]"

        reordered_conformer = reordered_record.conformers[0]

        assert numpy.allclose(
            reordered_conformer.coordinates, numpy.flipud(original_coordinates)
        )

        assert numpy.allclose(reordered_conformer.partial_charges[0].values, [1.5, 0.5])
        assert numpy.allclose(reordered_conformer.bond_orders[0].values, [(1, 0, 0.2)])


class TestMoleculeStore:
    def test_db_version_property(self, tmp_path):
        """Tests that a version is correctly added to a new store."""

        store = MoleculeStore(f"{tmp_path}.sqlite")

        with store._get_session() as db:
            db_info = db.query(DBInformation).first()

            assert db_info is not None
            assert db_info.version == DB_VERSION

        assert store.db_version == DB_VERSION

    def test_provenance_property(self, tmp_path):
        """Tests that a stores provenance can be set / retrieved."""

        store = MoleculeStore(f"{tmp_path}.sqlite")

        assert store.general_provenance == {}
        assert store.software_provenance == {}

        general_provenance = {"author": "Author 1"}
        software_provenance = {"psi4": "0.1.0"}

        store.set_provenance(general_provenance, software_provenance)

        assert store.general_provenance == general_provenance
        assert store.software_provenance == software_provenance

    def test_smiles_property(self, tmp_molecule_store):
        assert {*tmp_molecule_store.smiles} == {"[Ar:1]", "[He:1]", "[Cl:1][Cl:2]"}

    def test_charge_methods_property(self, tmp_molecule_store):
        assert {*tmp_molecule_store.charge_methods} == {"am1", "am1bcc"}

    def test_wbo_methods_property(self, tmp_molecule_store):
        assert {*tmp_molecule_store.wbo_methods} == {"am1"}

    def test_db_invalid_version(self, tmp_path):
        """Tests that the correct exception is raised when loading a store
        with an unsupported version."""

        store = MoleculeStore(f"{tmp_path}.sqlite")

        with store._get_session() as db:
            db_info = db.query(DBInformation).first()
            db_info.version = DB_VERSION - 1

        with pytest.raises(IncompatibleDBVersion) as error_info:
            MoleculeStore(f"{tmp_path}.sqlite")

        assert error_info.value.found_version == DB_VERSION - 1
        assert error_info.value.expected_version == DB_VERSION

    def test_match_conformers(self):

        matches = MoleculeStore._match_conformers(
            "[Cl:1][H:2]",
            db_conformers=[
                DBConformerRecord(
                    coordinates=numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
                ),
                DBConformerRecord(
                    coordinates=numpy.array([[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
                ),
            ],
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[0.0, -2.0, 0.0], [0.0, 2.0, 0.0]]),
                    partial_charges=[],
                    bond_orders=[],
                ),
                ConformerRecord(
                    coordinates=numpy.array([[0.0, -2.0, 0.0], [0.0, 3.0, 0.0]]),
                    partial_charges=[],
                    bond_orders=[],
                ),
                ConformerRecord(
                    coordinates=numpy.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 0.0]]),
                    partial_charges=[],
                    bond_orders=[],
                ),
            ],
        )

        assert matches == {0: 1, 2: 0}

    def test_store_partial_charge_data(self, tmp_path):

        store = MoleculeStore(f"{tmp_path}.sqlite")

        store.store(
            MoleculeRecord(
                smiles="[Cl:1][H:2]",
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.arange(6).reshape((2, 3)),
                        partial_charges=[
                            PartialChargeSet(method="am1", values=[0.50, 1.50])
                        ],
                    )
                ],
            )
        )
        assert len(store) == 1

        store.store(
            MoleculeRecord(
                smiles="[Cl:2][H:1]",
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.flipud(numpy.arange(6).reshape((2, 3))),
                        partial_charges=[
                            PartialChargeSet(method="am1bcc", values=[0.25, 0.75])
                        ],
                    )
                ],
            )
        )

        assert len(store) == 1
        assert {*store.charge_methods} == {"am1", "am1bcc"}

        record = store.retrieve()[0]
        assert len(record.conformers) == 1

        with pytest.raises(
            RuntimeError,
            match=re.escape("am1bcc charges already stored for [Cl:1][H:2]"),
        ):

            store.store(
                MoleculeRecord(
                    smiles="[Cl:2][H:1]",
                    conformers=[
                        ConformerRecord(
                            coordinates=numpy.arange(6).reshape((2, 3)),
                            partial_charges=[
                                PartialChargeSet(method="am1bcc", values=[0.25, 0.75])
                            ],
                        )
                    ],
                )
            )

        assert len(store) == 1
        assert {*store.charge_methods} == {"am1", "am1bcc"}

        record = store.retrieve()[0]
        assert len(record.conformers) == 1

    def test_store_bond_order_data(self, tmp_path):

        store = MoleculeStore(f"{tmp_path}.sqlite")

        store.store(
            MoleculeRecord(
                smiles="[Cl:1][H:2]",
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.arange(6).reshape((2, 3)),
                        bond_orders=[
                            WibergBondOrderSet(method="am1", values=[(0, 1, 0.5)])
                        ],
                    )
                ],
            )
        )
        assert len(store) == 1

        with pytest.raises(
            RuntimeError, match=re.escape("am1 WBOs already stored for [Cl:1][H:2]")
        ):
            store.store(
                MoleculeRecord(
                    smiles="[Cl:2][H:1]",
                    conformers=[
                        ConformerRecord(
                            coordinates=numpy.arange(6).reshape((2, 3)),
                            bond_orders=[
                                WibergBondOrderSet(method="am1", values=[(0, 1, 0.5)])
                            ],
                        )
                    ],
                )
            )

        store.store(
            MoleculeRecord(
                smiles="[Cl:2][H:1]",
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.zeros((2, 3)),
                        bond_orders=[
                            WibergBondOrderSet(method="am1", values=[(0, 1, 0.5)])
                        ],
                    )
                ],
            )
        )

        assert len(store) == 1
        assert {*store.wbo_methods} == {"am1"}

        record = store.retrieve()[0]
        assert len(record.conformers) == 2

    @pytest.mark.parametrize(
        "partial_charge_method,bond_order_method,n_expected",
        [
            (None, None, 3),
            ("am1", None, 2),
            ("am1bcc", None, 2),
            ([], "am1", 1),
        ],
    )
    def test_retrieve_data(
        self, partial_charge_method, bond_order_method, n_expected, tmp_molecule_store
    ):

        retrieved_records = tmp_molecule_store.retrieve(
            partial_charge_methods=partial_charge_method,
            bond_order_methods=bond_order_method,
        )
        assert len(retrieved_records) == n_expected

        for record in retrieved_records:

            for conformer in record.conformers:

                assert partial_charge_method is None or all(
                    partial_charges.method == partial_charge_method
                    for partial_charges in conformer.partial_charges
                )

                assert bond_order_method is None or all(
                    bond_orders.method == bond_order_method
                    for bond_orders in conformer.bond_orders
                )

    def test_len(self, tmp_molecule_store):
        assert len(tmp_molecule_store) == 3
