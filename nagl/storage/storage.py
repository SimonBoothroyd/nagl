"""This module contains classes which are able to store and retrieve
calculated sets of partial charges and Wiberg bond orders.
"""
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import TYPE_CHECKING, ContextManager, Dict, List, Literal, Optional

import numpy
from pydantic import BaseModel, Field
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from nagl.storage.db import (
    DB_VERSION,
    DBBase,
    DBConformerRecord,
    DBCoordinate,
    DBGeneralProvenance,
    DBInformation,
    DBMoleculeRecord,
    DBPartialCharge,
    DBPartialChargeSet,
    DBSoftwareProvenance,
    DBWibergBondOrder,
    DBWibergBondOrderSet,
)
from nagl.storage.exceptions import IncompatibleDBVersion
from nagl.utilities.openeye import requires_oe_package, smiles_to_molecule

if TYPE_CHECKING:
    Array = numpy.ndarray
else:
    from nagl.utilities.pydantic import Array


ChargeMethod = Literal["am1", "am1bcc"]
WBOMethod = Literal["am1"]


class _BaseStoredModel(BaseModel):
    class Config:
        orm_mode = True


class PartialChargeSet(_BaseStoredModel):
    """A class which stores a set of partial charges computed using a particular
    conformer and charge method."""

    method: ChargeMethod = Field(
        ..., description="The method used to compute these partial charges."
    )

    values: Array[float] = Field(
        ..., description="The values [e] of the partial charges with shape=(n_atoms,)."
    )


class WibergBondOrderSet(_BaseStoredModel):
    """A class which stores a set of Wiberg bond orders computed using a particular
    conformer and calculation method."""

    method: WBOMethod = Field(
        ..., description="The method used to compute these bond orders."
    )

    values: Array[float] = Field(
        ..., description="The values of the WBOs with shape=(n_atoms,)."
    )


class ConformerRecord(_BaseStoredModel):
    """A record which stores the coordinates of a molecule in a particular conformer,
    as well as sets of partial charges and WBOs computed using this conformer and
    for different methods."""

    coordinates: Array[float] = Field(
        ...,
        description="The coordinates [Angstrom] of this conformer with "
        "shape=(n_atoms, 3).",
    )

    partial_charges: List[PartialChargeSet] = Field(
        ...,
        description="Sets of partial charges computed using this conformer and "
        "different charge methods (e.g. am1, am1bcc).",
    )
    bond_orders: List[WibergBondOrderSet] = Field(
        ...,
        description="Sets of partial charges computed using this conformer and "
        "different computation methods (e.g. am1).",
    )


class MoleculeRecord(_BaseStoredModel):
    """A record which contains information for a specific molecule, including the
    coordinates of the molecule in different conformers, and partial charges / WBOs
    computed for those conformers."""

    indexed_smiles: str = Field(
        ...,
        description="The indexed SMILES patterns which encodes both the molecule stored "
        "in this record and a map between the atoms and the molecule and their "
        "coordinates.",
    )

    conformers: List[ConformerRecord] = Field(
        ...,
        description="Different conformers of the molecule, including sets of partial "
        "charges and WBOs computed using each conformer.",
    )


class MoleculeStore:
    """A class used to to store and retrieve calculated sets of partial charges and
    Wiberg bond orders for sets of molecules.
    """

    @property
    def db_version(self) -> int:
        """Returns the version of the underlying database."""
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            return db_info.version

    @property
    def general_provenance(self) -> Dict[str, str]:
        """Returns a dictionary containing general provenance about the store such as
        the author and when it was generated.
        """

        with self._get_session() as db:

            db_info = db.query(DBInformation).first()

            return {
                provenance.key: provenance.value
                for provenance in db_info.general_provenance
            }

    @property
    def software_provenance(self) -> Dict[str, str]:
        """A dictionary containing provenance of the software and packages used
        to generate the data in the store.
        """

        with self._get_session() as db:

            db_info = db.query(DBInformation).first()

            return {
                provenance.key: provenance.value
                for provenance in db_info.software_provenance
            }

    @property
    def smiles(self) -> List[str]:
        """A list of SMILES representations of the unique molecules in the store."""

        with self._get_session() as db:
            return [
                smiles for (smiles,) in db.query(DBMoleculeRecord.smiles).distinct()
            ]

    @property
    def charge_methods(self) -> List[str]:
        """A list of the methods used to compute the partial charges in the store."""

        with self._get_session() as db:
            return [
                method for (method,) in db.query(DBPartialChargeSet.method).distinct()
            ]

    @property
    def wbo_methods(self) -> List[str]:
        """A list of the methods used to compute the WBOs in the store."""

        with self._get_session() as db:
            return [
                method for (method,) in db.query(DBWibergBondOrderSet.method).distinct()
            ]

    def __init__(self, database_path: str = "esp-store.sqlite"):
        """

        Parameters
        ----------
        database_path
            The path to the SQLite database to store to and retrieve data from.
        """
        self._database_url = f"sqlite:///{database_path}"

        self._engine = create_engine(self._database_url, echo=False)
        DBBase.metadata.create_all(self._engine)

        self._session_maker = sessionmaker(
            autocommit=False, autoflush=False, bind=self._engine
        )

        # Validate the DB version if present, or add one if not.
        with self._get_session() as db:
            db_info = db.query(DBInformation).first()

            if not db_info:
                db_info = DBInformation(version=DB_VERSION)
                db.add(db_info)

            if db_info.version != DB_VERSION:
                raise IncompatibleDBVersion(db_info.version, DB_VERSION)

    def set_provenance(
        self,
        general_provenance: Dict[str, str],
        software_provenance: Dict[str, str],
    ):
        """Set the stores provenance information.

        Parameters
        ----------
        general_provenance
            A dictionary storing provenance about the store such as the author,
            when it was generated etc.
        software_provenance
            A dictionary storing the provenance of the software and packages used
            to generate the data in the store.
        """

        with self._get_session() as db:

            db_info: DBInformation = db.query(DBInformation).first()
            db_info.general_provenance = [
                DBGeneralProvenance(key=key, value=value)
                for key, value in general_provenance.items()
            ]
            db_info.software_provenance = [
                DBSoftwareProvenance(key=key, value=value)
                for key, value in software_provenance.items()
            ]

    @contextmanager
    def _get_session(self) -> ContextManager[Session]:

        session = self._session_maker()

        try:
            yield session
            session.commit()
        except BaseException as e:
            session.rollback()
            raise e
        finally:
            session.close()

    @classmethod
    def _db_records_to_model(
        cls, db_records: List[DBMoleculeRecord]
    ) -> List[MoleculeRecord]:
        """Maps a set of database records into their corresponding
        data models.

        Parameters
        ----------
        db_records
            The records to map.

        Returns
        -------
            The mapped data models.
        """
        # noinspection PyTypeChecker
        return [
            MoleculeRecord(
                indexed_smiles=db_record.smiles,
                conformers=[
                    ConformerRecord(
                        coordinates=numpy.array(
                            [
                                [db_coordinate.x, db_coordinate.y, db_coordinate.z]
                                for db_coordinate in db_conformer.coordinates
                            ]
                        ),
                        partial_charges=[
                            PartialChargeSet(
                                method=db_partial_charges.method,
                                values=[
                                    value.value for value in db_partial_charges.values
                                ],
                            )
                            for db_partial_charges in db_conformer.partial_charges
                        ],
                        bond_orders=[
                            WibergBondOrderSet(
                                method=db_bond_orders.method,
                                values=[value.value for value in db_bond_orders.values],
                            )
                            for db_bond_orders in db_conformer.bond_orders
                        ],
                    )
                    for db_conformer in db_record.conformers
                ],
            )
            for db_record in db_records
        ]

    @classmethod
    def _store_smiles_records(
        cls, db: Session, indexed_smiles: str, records: List[MoleculeRecord]
    ) -> DBMoleculeRecord:
        """Stores a set of records which all store information for the same
        molecule.

        Parameters
        ----------
        db
            The current database session.
        indexed_smiles
            The indexed SMILES representation of the molecule.
        records
            The records to store.
        """

        db_record = DBMoleculeRecord(smiles=indexed_smiles)

        # noinspection PyTypeChecker
        # noinspection PyUnresolvedReferences
        db_record.conformers.extend(
            DBConformerRecord(
                coordinates=[
                    DBCoordinate(x=coordinate[0], y=coordinate[1], z=coordinate[2])
                    for coordinate in conformer.coordinates
                ],
                partial_charges=[
                    DBPartialChargeSet(
                        method=partial_charges.method,
                        values=[
                            DBPartialCharge(value=value)
                            for value in partial_charges.values
                        ],
                    )
                    for partial_charges in conformer.partial_charges
                ],
                bond_orders=[
                    DBWibergBondOrderSet(
                        method=bond_orders.method,
                        values=[
                            DBWibergBondOrder(value=value)
                            for value in bond_orders.values
                        ],
                    )
                    for bond_orders in conformer.bond_orders
                ],
            )
            for record in records
            for conformer in record.conformers
        )

        db.add(db_record)
        return db_record

    @classmethod
    @functools.lru_cache(128)
    @requires_oe_package("oechem")
    def _tagged_to_canonical_smiles(cls, indexed_smiles: str) -> str:
        """Converts a smiles pattern which contains atom indices into
        a canonical smiles pattern without indices.

        Parameters
        ----------
        indexed_smiles
            The tagged smiles pattern to convert.

        Returns
        -------
            The canonical smiles pattern.
        """

        from openeye import oechem

        oe_molecule = smiles_to_molecule(indexed_smiles)

        for atom in oe_molecule.GetAtoms():
            atom.SetMapIdx(0)

        return oechem.OECreateCanSmiString(oe_molecule)

    def store(self, *records: MoleculeRecord):
        """Store the molecules and their computed properties in the data store.

        Parameters
        ----------
        records
            The records to store.
        """

        # Validate and re-partition the records by their smiles patterns.
        records_by_smiles: Dict[str, List[MoleculeRecord]] = defaultdict(list)

        for record in records:

            record = record.copy(deep=True)
            smiles = self._tagged_to_canonical_smiles(record.indexed_smiles)

            records_by_smiles[smiles].append(record)

        # Store the records.
        with self._get_session() as db:

            for smiles in records_by_smiles:
                self._store_smiles_records(db, smiles, records_by_smiles[smiles])

    def retrieve(
        self,
        partial_charge_method: Optional[ChargeMethod] = None,
        bond_order_method: Optional[WBOMethod] = None,
    ) -> List[MoleculeRecord]:
        """Retrieve records stored in this data store, optionally
        according to a set of filters."""

        with self._get_session() as db:

            db_records = db.query(DBMoleculeRecord).join(DBConformerRecord)

            if partial_charge_method is not None:
                db_records = db_records.join(DBPartialChargeSet)
                db_records = db_records.filter(
                    DBPartialChargeSet.method == partial_charge_method
                )
            if bond_order_method is not None:
                db_records = db_records.join(DBWibergBondOrderSet)
                db_records = db_records.filter(
                    DBWibergBondOrderSet.method == bond_order_method
                )

            db_records = db_records.all()

            records = self._db_records_to_model(db_records)
            return records
