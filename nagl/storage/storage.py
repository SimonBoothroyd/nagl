"""This module contains classes to store labelled molecules within a compact database
structure.
"""
import functools
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Collection,
    ContextManager,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import numpy
from openff.utilities import requires_package
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from nagl.storage.db import (
    DB_VERSION,
    DBBase,
    DBConformerRecord,
    DBGeneralProvenance,
    DBInformation,
    DBMoleculeRecord,
    DBPartialChargeSet,
    DBSoftwareProvenance,
    DBWibergBondOrderSet,
)
from nagl.storage.exceptions import IncompatibleDBVersion
from nagl.utilities.rmsd import are_conformers_identical
from nagl.utilities.smiles import map_indexed_smiles

if TYPE_CHECKING:
    Array = numpy.ndarray
else:
    from nagl.utilities.pydantic import Array


ChargeMethod = Literal["am1", "am1bcc"]
WBOMethod = Literal["am1"]


class _BaseStoredModel(BaseModel):
    class Config:
        orm_mode = True
        allow_mutation = False


class PartialChargeSet(_BaseStoredModel):
    """A class which stores a set of partial charges computed using a particular
    conformer and charge method."""

    method: ChargeMethod = Field(
        ..., description="The method used to compute these partial charges."
    )

    values: Union[Tuple[float, ...], List[float]] = Field(
        ..., description="The values [e] of the partial charges."
    )


class WibergBondOrderSet(_BaseStoredModel):
    """A class which stores a set of Wiberg bond orders computed using a particular
    conformer and calculation method."""

    method: WBOMethod = Field(
        ..., description="The method used to compute these bond orders."
    )

    values: Union[
        Tuple[Tuple[int, int, float], ...], List[Tuple[int, int, float]]
    ] = Field(
        ...,
        description="The values of the WBOs stored as tuples of the form "
        "`(index_a, index_b, value)` where `index_a` and `index_b` are the indices "
        "of the atoms involved in the bond associated with the WBO value.",
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

    partial_charges: Union[
        Tuple[PartialChargeSet, ...], List[PartialChargeSet]
    ] = Field(
        tuple(),
        description="Sets of partial charges computed using this conformer and "
        "different charge methods (e.g. am1, am1bcc).",
    )
    bond_orders: Union[
        Tuple[WibergBondOrderSet, ...], List[WibergBondOrderSet]
    ] = Field(
        tuple(),
        description="Sets of partial charges computed using this conformer and "
        "different computation methods (e.g. am1).",
    )

    @property
    def partial_charges_by_method(self) -> Dict[ChargeMethod, Tuple[float, ...]]:
        """Returns the values of partial charges [e] computed for this conformer using a
        specific method."""
        return {
            charge_set.method: charge_set.values for charge_set in self.partial_charges
        }

    @property
    def bond_orders_by_method(
        self,
    ) -> Dict[WBOMethod, Tuple[Tuple[int, int, float], ...]]:
        """Returns the values of the bond orders computed for this conformer using a
        specific method."""

        return {wbo_set.method: wbo_set.values for wbo_set in self.bond_orders}

    @validator("coordinates")
    def _validate_coordinates(cls, value):

        assert (value.ndim == 2 and value.shape[1] == 3) or (
            value.ndim == 1 and len(value) % 3 == 0
        ), "coordinates must be re-shapable to `(n_atoms, 3)`"

        value = value.reshape((-1, 3))
        value.flags.writeable = False

        return value

    @validator("partial_charges")
    def _validate_partial_charges(cls, value, values):

        assert len({x.method for x in value}) == len(
            value
        ), "multiple charge sets computed using the same method are not allowed"

        assert all(
            len(x.values) == len(values["coordinates"]) for x in value
        ), "the number of partial charges must match the number of coordinates"

        return value

    @validator("bond_orders")
    def _validate_bond_orders(cls, value, values):

        assert len({x.method for x in value}) == len(
            value
        ), "multiple bond order sets computed using the same method are not allowed"

        return value


class MoleculeRecord(_BaseStoredModel):
    """A record which contains information for a labelled molecule. This may include the
    coordinates of the molecule in different conformers, and partial charges / WBOs
    computed for those conformers."""

    smiles: str = Field(
        ...,
        description="The SMILES patterns which encodes both the molecule stored "
        "in this record as well as the unique index assigned to each atom (including "
        "hydrogen).",
    )

    conformers: List[ConformerRecord] = Field(
        ...,
        description="Conformers associated with this molecule. Each conformer will "
        "contain labelled properties, such as sets of partial charges and WBOs computed "
        "from the conformer.",
    )

    def average_partial_charges(self, method: ChargeMethod) -> Tuple[float, ...]:
        """Computes the average partial charges [e] over all stored values."""

        return tuple(
            numpy.mean(
                [
                    conformer.partial_charges_by_method[method]
                    for conformer in self.conformers
                    if method in conformer.partial_charges_by_method
                ],
                axis=0,
            )
        )

    def reorder(self, expected_smiles: str) -> "MoleculeRecord":
        """Reorders is data stored in this record so that the atom ordering matches
        the ordering of the specified indexed SMILES pattern.

        Args:
            expected_smiles: The indexed SMILES pattern which encodes the desired atom
                ordering.

        Returns
            The reordered record, or the existing record if its order already matches.
        """

        if self.smiles == expected_smiles:
            return self

        map_indices = map_indexed_smiles(expected_smiles, self.smiles)
        inverse_map = {j: i for i, j in map_indices.items()}

        map_indices_array = numpy.array(
            [map_indices[i] for i in range(len(map_indices))]
        )

        reordered_record = MoleculeRecord(
            smiles=expected_smiles,
            conformers=[
                ConformerRecord(
                    coordinates=conformer.coordinates[map_indices_array],
                    partial_charges=tuple(
                        PartialChargeSet(
                            method=charge_set.method,
                            values=tuple(
                                charge_set.values[map_indices[i]]
                                for i in range(len(map_indices))
                            ),
                        )
                        for charge_set in conformer.partial_charges
                    ),
                    bond_orders=tuple(
                        WibergBondOrderSet(
                            method=bond_order_set.method,
                            values=tuple(
                                (inverse_map[index_a], inverse_map[index_b], value)
                                for (index_a, index_b, value) in bond_order_set.values
                            ),
                        )
                        for bond_order_set in conformer.bond_orders
                    ),
                )
                for conformer in self.conformers
            ],
        )

        return reordered_record


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

    def __init__(self, database_path: str = "molecule-store.sqlite"):
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
    @functools.lru_cache(2048)
    @requires_package("openff.toolkit")
    def _to_canonical_smiles(cls, smiles: str) -> str:
        """Converts a SMILES pattern which may contain atom indices into
        a canonical SMILES pattern without indices.

        Parameters
        ----------
        smiles
            The smiles pattern to convert.

        Returns
        -------
            The canonical smiles pattern.
        """
        from openff.toolkit.topology import Molecule

        return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_smiles(
            isomeric=False, explicit_hydrogens=False
        )

    @classmethod
    @functools.lru_cache(2048)
    @requires_package("openff.toolkit")
    def _to_inchi_key(cls, smiles: str) -> str:
        """Converts a SMILES pattern to a InChI key representation.

        Parameters
        ----------
        smiles
            The smiles pattern to convert.

        Returns
        -------
            The InChI key representation.
        """
        from openff.toolkit.topology import Molecule

        return Molecule.from_smiles(smiles, allow_undefined_stereo=True).to_inchikey(
            fixed_hydrogens=True
        )

    @classmethod
    @requires_package("openff.toolkit")
    def _match_conformers(
        cls,
        indexed_smiles: str,
        db_conformers: List[DBConformerRecord],
        conformers: List[ConformerRecord],
    ) -> Dict[int, int]:
        """A method which attempts to match a set of new conformers to store with
        conformers already present in the database by comparing the RMS of the
        two sets.

        Args:
            indexed_smiles: The indexed SMILES pattern associated with the conformers.
            db_conformers: The database conformers.
            conformers: The conformers to store.

        Returns:
            A dictionary which maps the index of a conformer to the index of a database
            conformer. The indices of conformers which do not match an existing database
            conformer are not included.
        """

        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_mapped_smiles(
            indexed_smiles, allow_undefined_stereo=True
        )

        # See if any of the conformers to add are already in the DB.
        matches = {}

        for index, conformer in enumerate(conformers):

            db_match = None

            for db_index, db_conformer in enumerate(db_conformers):

                are_identical = are_conformers_identical(
                    molecule,
                    conformer.coordinates,
                    db_conformer.coordinates,
                )

                if not are_identical:
                    continue

                db_match = db_index
                break

            if db_match is None:
                continue

            matches[index] = db_match

        return matches

    @classmethod
    def _store_conformer_records(
        cls, smiles: str, db_parent: DBMoleculeRecord, records: List[ConformerRecord]
    ):
        """Store a set of conformer records in an existing DB molecule record."""

        conformer_matches = cls._match_conformers(smiles, db_parent.conformers, records)

        # Create new database conformers for those unmatched conformers.
        missing_indices = {*range(len(records))} - {*conformer_matches}

        for index in missing_indices:
            # noinspection PyTypeChecker
            db_parent.conformers.append(
                DBConformerRecord(coordinates=records[index].coordinates)
            )
            conformer_matches[index] = len(db_parent.conformers) - 1

        for index, db_index in conformer_matches.items():

            db_record = db_parent.conformers[db_index]
            record = records[index]

            existing_charge_methods = [x.method for x in db_record.partial_charges]

            for partial_charges in record.partial_charges:

                if partial_charges.method in existing_charge_methods:
                    raise RuntimeError(
                        f"{partial_charges.method} charges already stored for {smiles} "
                        f"with coordinates {record.coordinates}"
                    )

                db_record.partial_charges.append(
                    DBPartialChargeSet(
                        method=partial_charges.method,
                        values=partial_charges.values,
                    )
                )

            existing_bond_methods = [x.method for x in db_record.bond_orders]

            for bond_orders in record.bond_orders:

                if bond_orders.method in existing_bond_methods:

                    raise RuntimeError(
                        f"{bond_orders.method} WBOs already stored for {smiles} "
                        f"with coordinates {record.coordinates}"
                    )

                db_record.bond_orders.append(
                    DBWibergBondOrderSet(
                        method=bond_orders.method, values=bond_orders.values
                    )
                )

    @classmethod
    def _store_records_with_inchi_key(
        cls, db: Session, inchi_key: str, records: List[MoleculeRecord]
    ):
        """Stores a set of records which all store information for molecules with the
        same hill formula.

        Notes
        -----
        * We split by the hill formula to speed up finding molecules that already exist
        in the data

        Parameters
        ----------
        db
            The current database session.
        inchi_key
            The **fixed hydrogen** InChI key representation of the molecule stored in
            the records.
        records
            The records to store.
        """

        existing_db_records: Collection[DBMoleculeRecord] = (
            db.query(DBMoleculeRecord)
            .filter(DBMoleculeRecord.inchi_key == inchi_key)
            .all()
        )

        if len(existing_db_records) > 1:
            # Sanity check that no two DB records have the same InChI key
            raise RuntimeError("The database is not self consistent.")

        db_record = (
            DBMoleculeRecord(inchi_key=inchi_key, smiles=records[0].smiles)
            if len(existing_db_records) == 0
            else next(iter(existing_db_records))
        )

        # Retrieve the DB indexed SMILES that defines the ordering the atoms in each
        # record should have and re-order the incoming records to match.
        expected_smiles = db_record.smiles

        conformer_records = [
            conformer_record
            for record in records
            for conformer_record in record.reorder(expected_smiles).conformers
        ]

        cls._store_conformer_records(expected_smiles, db_record, conformer_records)

        db.add(db_record)

    def store(self, *records: MoleculeRecord):
        """Store the molecules and their computed properties in the data store.

        Parameters
        ----------
        records
            The records to store.
        """

        records_by_inchi_key: Dict[str, List[MoleculeRecord]] = defaultdict(list)

        for record in records:
            records_by_inchi_key[self._to_inchi_key(record.smiles)].append(record)

        with self._get_session() as db:

            for inchi_key, inchi_records in records_by_inchi_key.items():
                self._store_records_with_inchi_key(db, inchi_key, inchi_records)

    @classmethod
    def _db_records_to_model(
        cls,
        db_records: List[DBMoleculeRecord],
        partial_charge_method: Optional[ChargeMethod] = None,
        bond_order_method: Optional[WBOMethod] = None,
    ) -> List[MoleculeRecord]:
        """Maps a set of database records into their corresponding data models,
        optionally retaining only partial charge sets and WBO sets computed with a
        specified method.

        Args
            db_records: The records to map.
            partial_charge_method: The (optional) partial charge method to filter by.
            bond_order_method: The (optional) WBO method to filter by.

        Returns
            The mapped data models.
        """

        # noinspection PyTypeChecker
        return [
            MoleculeRecord(
                smiles=db_record.smiles,
                conformers=[
                    ConformerRecord(
                        coordinates=db_conformer.coordinates,
                        partial_charges=[
                            PartialChargeSet(
                                method=db_partial_charges.method,
                                values=db_partial_charges.values,
                            )
                            for db_partial_charges in db_conformer.partial_charges
                            if partial_charge_method is None
                            or db_partial_charges.method == partial_charge_method
                        ],
                        bond_orders=[
                            WibergBondOrderSet(
                                method=db_bond_orders.method,
                                values=db_bond_orders.values,
                            )
                            for db_bond_orders in db_conformer.bond_orders
                            if bond_order_method is None
                            or db_bond_orders.method == bond_order_method
                        ],
                    )
                    for db_conformer in db_record.conformers
                ],
            )
            for db_record in db_records
        ]

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

            records = self._db_records_to_model(
                db_records, partial_charge_method, bond_order_method
            )
            return records

    def __len__(self):

        with self._get_session() as db:
            return db.query(DBMoleculeRecord.smiles).count()
