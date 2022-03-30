"""This module contains classes to store labelled molecules within a compact database
structure.
"""
import logging
import time
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
    Type,
    Union,
)

import numpy
from openff.utilities import requires_package
from pydantic import BaseModel, Field, validator
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from tqdm import tqdm

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
from nagl.utilities.toolkits import capture_toolkit_warnings, smiles_to_inchi_key

if TYPE_CHECKING:
    Array = numpy.ndarray
else:
    from nagl.utilities.pydantic import Array

_logger = logging.getLogger(__name__)


ChargeMethod = Union[Literal["am1", "am1bcc"], str]
WBOMethod = Union[Literal["am1"], str]

DBQueryResult = Tuple[
    int,  # DBMoleculeRecord.id
    str,  # DBMoleculeRecord.smiles,
    int,  # DBConformerRecord.id,
    numpy.ndarray,  # DBConformerRecord.coordinates,
    str,  # model_type.method,
    numpy.ndarray,  # model_type.values,
]


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

        if len(db_parent.conformers) > 0:

            _logger.warning(
                f"An entry for {smiles} is already present in the molecule store. "
                f"Trying to find matching conformers."
            )

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
    def _store_records_with_smiles(
        cls,
        db: Session,
        inchi_key: str,
        records: List[MoleculeRecord],
        existing_db_record: Optional[DBMoleculeRecord],
    ):
        """Stores a set of records which all store information for molecules with the
        same SMILES representation AND the same fixed hydrogen InChI key.

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

        db_record = (
            DBMoleculeRecord(inchi_key=inchi_key, smiles=records[0].smiles)
            if existing_db_record is None
            else existing_db_record
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

    @classmethod
    def _store_records_with_inchi_key(
        cls, db: Session, inchi_key: str, records: List[MoleculeRecord]
    ):
        """Stores a set of records which all store information for molecules with the
        same fixed hydrogen InChI key.

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

        from openff.toolkit.topology import Molecule

        existing_db_records: Collection[DBMoleculeRecord] = (
            db.query(DBMoleculeRecord)
            .filter(DBMoleculeRecord.inchi_key == inchi_key)
            .all()
        )

        existing_db_records_by_smiles = {}

        for existing_db_record in existing_db_records:

            molecule: Molecule = Molecule.from_smiles(
                existing_db_record.smiles, allow_undefined_stereo=True
            )
            smiles = molecule.to_smiles(mapped=False)

            if smiles in existing_db_records_by_smiles:
                # Sanity check that no two DB records have the same InChI key AND the
                # same canonical SMILES pattern.
                raise RuntimeError("The database is not self consistent.")

            existing_db_records_by_smiles[smiles] = existing_db_record

        records_by_smiles = defaultdict(list)

        for record in records:

            molecule: Molecule = Molecule.from_smiles(
                record.smiles, allow_undefined_stereo=True
            )
            records_by_smiles[molecule.to_smiles(mapped=False)].append(record)

        for smiles, smiles_records in records_by_smiles.items():

            cls._store_records_with_smiles(
                db,
                inchi_key,
                smiles_records,
                existing_db_records_by_smiles.get(smiles, None),
            )

    def store(self, *records: MoleculeRecord):
        """Store the molecules and their computed properties in the data store.

        Parameters
        ----------
        records
            The records to store.
        """

        with capture_toolkit_warnings():

            records_by_inchi_key: Dict[str, List[MoleculeRecord]] = defaultdict(list)

            for record in tqdm(records, desc="grouping records to store by InChI key"):
                records_by_inchi_key[smiles_to_inchi_key(record.smiles)].append(record)

            with self._get_session() as db:

                for inchi_key, inchi_records in tqdm(
                    records_by_inchi_key.items(), desc="storing grouped records"
                ):
                    self._store_records_with_inchi_key(db, inchi_key, inchi_records)

    @classmethod
    def _db_query_by_method(
        cls,
        db: Session,
        model_type: Union[Type[DBPartialChargeSet], Type[DBWibergBondOrderSet]],
        allowed_methods: List[str],
    ) -> List[DBQueryResult]:
        """Returns the results of querying the DB for records associated with either a
        set of partial charge or bond order methods

        Returns:
            A list of tuples of the form::

                (
                    DBMoleculeRecord.id,
                    DBMoleculeRecord.smiles,
                    DBConformerRecord.id,
                    DBConformerRecord.coordinates,
                    model_type.method,
                    model_type.values
                )
        """

        return (
            db.query(
                DBMoleculeRecord.id,
                DBMoleculeRecord.smiles,
                DBConformerRecord.id,
                DBConformerRecord.coordinates,
                model_type.method,
                model_type.values,
            )
            .order_by(DBMoleculeRecord.id)
            .join(
                DBConformerRecord,
                DBConformerRecord.parent_id == DBMoleculeRecord.id,
            )
            .join(
                model_type,
                model_type.parent_id == DBConformerRecord.id,
            )
            .filter(model_type.method.in_(allowed_methods))
            .all()
        )

    @classmethod
    def _db_columns_to_models(
        cls,
        db_partial_charge_columns: List[DBQueryResult],
        db_bond_order_columns: List[DBQueryResult],
    ) -> List[MoleculeRecord]:
        """Maps a set of database records into their corresponding data models,
        optionally retaining only partial charge sets and WBO sets computed with a
        specified method.

        Args:

        Returns:
            The mapped data models.
        """

        raw_objects = defaultdict(
            lambda: {
                "smiles": None,
                "conformers": defaultdict(
                    lambda: {
                        "coordinates": None,
                        "partial_charges": {},
                        "bond_orders": {},
                    }
                ),
            }
        )

        for value_type, db_columns, model_type in [
            ("partial_charges", db_partial_charge_columns, PartialChargeSet),
            ("bond_orders", db_bond_order_columns, WibergBondOrderSet),
        ]:

            for (
                db_molecule_id,
                db_molecule_smiles,
                db_conformer_id,
                db_conformer_coordinates,
                db_method,
                db_values,
            ) in db_columns:

                raw_objects[db_molecule_id]["smiles"] = db_molecule_smiles
                raw_objects[db_molecule_id]["conformers"][db_conformer_id][
                    "coordinates"
                ] = db_conformer_coordinates
                raw_objects[db_molecule_id]["conformers"][db_conformer_id][value_type][
                    db_method
                ] = PartialChargeSet.construct(method=db_method, values=db_values)

        records = [
            MoleculeRecord.construct(
                smiles=raw_molecule["smiles"],
                conformers=[
                    ConformerRecord.construct(
                        coordinates=raw_conformer["coordinates"],
                        partial_charges=[*raw_conformer["partial_charges"].values()],
                        bond_orders=[*raw_conformer["bond_orders"].values()],
                    )
                    for raw_conformer in raw_molecule["conformers"].values()
                ],
            )
            for raw_molecule in raw_objects.values()
        ]

        return records

    def retrieve(
        self,
        partial_charge_methods: Optional[
            Union[ChargeMethod, List[ChargeMethod]]
        ] = None,
        bond_order_methods: Optional[Union[WBOMethod, List[WBOMethod]]] = None,
    ) -> List[MoleculeRecord]:
        """Retrieve records stored in this data store

        Args:
            partial_charge_methods: The (optional) list of charge methods to retrieve
                from the store. By default (`None`) all charges will be returned.
            bond_order_methods: The (optional) list of bond order methods to retrieve
                from the store. By default (`None`) all bond orders will be returned.

        Returns:
            The retrieved records.
        """

        if isinstance(partial_charge_methods, str):
            partial_charge_methods = [partial_charge_methods]
        elif partial_charge_methods is None:
            partial_charge_methods = self.charge_methods

        if isinstance(bond_order_methods, str):
            bond_order_methods = [bond_order_methods]
        elif bond_order_methods is None:
            bond_order_methods = self.wbo_methods

        with self._get_session() as db:

            db_partial_charge_columns = []
            db_bond_order_columns = []

            _logger.debug("performing SQL queries")
            start = time.perf_counter()

            if len(partial_charge_methods) > 0:

                db_partial_charge_columns = self._db_query_by_method(
                    db, DBPartialChargeSet, partial_charge_methods
                )

            if len(bond_order_methods) > 0:

                db_bond_order_columns = self._db_query_by_method(
                    db, DBWibergBondOrderSet, bond_order_methods
                )

            _logger.debug(f"performed SQL query {time.perf_counter() - start}s")

            _logger.debug("converting SQL columns to entries")
            start = time.perf_counter()
            records = self._db_columns_to_models(
                db_partial_charge_columns, db_bond_order_columns
            )
            _logger.debug(
                f"converted SQL columns to entries {time.perf_counter() - start}s"
            )

        return records

    def __len__(self):

        with self._get_session() as db:
            return db.query(DBMoleculeRecord.smiles).count()
