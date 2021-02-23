import numpy
import pytest

from nagl.storage.db import DB_VERSION, DBInformation
from nagl.storage.exceptions import IncompatibleDBVersion
from nagl.storage.storage import (
    ConformerRecord,
    MoleculeRecord,
    MoleculeStore,
    PartialChargeSet,
    WibergBondOrderSet,
)


def test_db_version(tmp_path):
    """Tests that a version is correctly added to a new store."""

    store = MoleculeStore(f"{tmp_path}.sqlite")

    with store._get_session() as db:

        db_info = db.query(DBInformation).first()

        assert db_info is not None
        assert db_info.version == DB_VERSION

    assert store.db_version == DB_VERSION


def test_provenance(tmp_path):
    """Tests that a stores provenance can be set / retrieved."""

    store = MoleculeStore(f"{tmp_path}.sqlite")

    assert store.general_provenance == {}
    assert store.software_provenance == {}

    general_provenance = {"author": "Author 1"}
    software_provenance = {"psi4": "0.1.0"}

    store.set_provenance(general_provenance, software_provenance)

    assert store.general_provenance == general_provenance
    assert store.software_provenance == software_provenance


def test_db_invalid_version(tmp_path):
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


def test_store_retrieve_data(tmp_path):

    tmp_path = "x"

    store = MoleculeStore(f"{tmp_path}.sqlite")

    expected_records = [
        MoleculeRecord(
            indexed_smiles="[Ar:1]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[0.0, 0.0, 0.0]]),
                    partial_charges=[PartialChargeSet(method="am1", values=[0.5])],
                    bond_orders=[],
                )
            ],
        ),
        MoleculeRecord(
            indexed_smiles="[He:1]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[0.0, 0.0, 0.0]]),
                    partial_charges=[PartialChargeSet(method="am1bcc", values=[-0.5])],
                    bond_orders=[],
                )
            ],
        ),
        MoleculeRecord(
            indexed_smiles="[Cl:1][Cl:2]",
            conformers=[
                ConformerRecord(
                    coordinates=numpy.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
                    partial_charges=[
                        PartialChargeSet(method="am1", values=[0.5, -0.5]),
                        PartialChargeSet(method="am1bcc", values=[0.75, -0.75]),
                    ],
                    bond_orders=[WibergBondOrderSet(method="am1", values=[1.2])],
                )
            ],
        ),
    ]

    store.store(*expected_records)

    retrieved_records = store.retrieve()
    assert len(retrieved_records) == 3

    retrieved_records = store.retrieve(bond_order_method="am1")
    assert len(retrieved_records) == 1

    retrieved_records = store.retrieve(partial_charge_method="am1")
    assert len(retrieved_records) == 2

    retrieved_records = store.retrieve(partial_charge_method="am1bcc")
    assert len(retrieved_records) == 2
