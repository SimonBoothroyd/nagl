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
)

import numpy
from pydantic import BaseModel, Field, validator
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
from nagl.utilities.openeye import (
    map_indexed_smiles,
    requires_oe_package,
    smiles_to_molecule,
)
from nagl.utilities.utilities import requires_package

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

    values: List[float] = Field(
        ..., description="The values [e] of the partial charges."
    )


class WibergBondOrderSet(_BaseStoredModel):
    """A class which stores a set of Wiberg bond orders computed using a particular
    conformer and calculation method."""

    method: WBOMethod = Field(
        ..., description="The method used to compute these bond orders."
    )

    values: List[Tuple[int, int, float]] = Field(
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

    partial_charges: List[PartialChargeSet] = Field(
        [],
        description="Sets of partial charges computed using this conformer and "
        "different charge methods (e.g. am1, am1bcc).",
    )
    bond_orders: List[WibergBondOrderSet] = Field(
        [],
        description="Sets of partial charges computed using this conformer and "
        "different computation methods (e.g. am1).",
    )

    @validator("partial_charges")
    def validate_partial_charges(cls, values):

        assert len({value.method for value in values}) == len(
            values
        ), "multiple charge sets computed using the same method are not allowed"

        return values

    @validator("bond_orders")
    def validate_bond_orders(cls, values):

        assert len({value.method for value in values}) == len(
            values
        ), "multiple bond order sets computed using the same method are not allowed"

        return values


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

    def reorder(self, expected_smiles: str) -> "MoleculeRecord":
        """Reorders is data stored in this record so that the atom ordering matches
        the ordering of the specified indexed SMILES pattern.

        Args:
            expected_smiles: The indexed SMILES pattern which encodes the desired atom
                ordering.

        Returns
            The reordered record.
        """

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
                    partial_charges=[
                        PartialChargeSet(
                            method=charge_set.method,
                            values=[
                                charge_set.values[map_indices[i]]
                                for i in range(len(map_indices))
                            ],
                        )
                        for charge_set in conformer.partial_charges
                    ],
                    bond_orders=[
                        WibergBondOrderSet(
                            method=bond_order_set.method,
                            values=[
                                (inverse_map[index_a], inverse_map[index_b], value)
                                for (index_a, index_b, value) in bond_order_set.values
                            ],
                        )
                        for bond_order_set in conformer.bond_orders
                    ],
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
    @functools.lru_cache(512)
    @requires_oe_package("oechem")
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

        from openeye import oechem

        oe_molecule = smiles_to_molecule(smiles)

        for atom in oe_molecule.GetAtoms():
            atom.SetMapIdx(0)

        return oechem.OECreateCanSmiString(oe_molecule)

    @classmethod
    @functools.lru_cache(512)
    @requires_oe_package("oechem")
    def _to_hill_formula(cls, smiles: str) -> str:
        """Converts a SMILES pattern to a Hill ordered molecular formula.

        Parameters
        ----------
        smiles
            The smiles pattern to convert.

        Returns
        -------
            The Hill ordered molecular formula.
        """

        from openeye import oechem

        oe_molecule = oechem.OEMol()
        oechem.OESmilesToMol(oe_molecule, smiles)

        output_stream = oechem.oemolostream()
        output_stream.SetFormat(oechem.OEFormat_MF)

        output_stream.openstring()

        oechem.OEWriteMolecule(output_stream, oe_molecule)

        formula = output_stream.GetString().decode().strip().replace("\n", "")
        output_stream.close()

        return formula

    @classmethod
    @requires_package("rdkit")
    def _to_rdkit(cls, smiles: str):
        """Creates an RDKit molecule object from an indexed SMILES pattern."""
        from rdkit import Chem

        rdkit_smiles_options = Chem.SmilesParserParams()
        rdkit_smiles_options.removeHs = False

        rdkit_molecule: Chem.RWMol = Chem.MolFromSmiles(smiles, rdkit_smiles_options)
        rdkit_atoms = {atom.GetIdx(): atom for atom in rdkit_molecule.GetAtoms()}

        new_order = [
            rdkit_atoms[i].GetAtomMapNum() - 1 for i in range(len(rdkit_atoms))
        ]

        return Chem.RenumberAtoms(rdkit_molecule, new_order)

    @classmethod
    @requires_package("rdkit")
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

        from rdkit import Chem, Geometry
        from rdkit.Chem import AllChem

        rdkit_molecule_a = cls._to_rdkit(indexed_smiles)
        rdkit_molecule_b = Chem.Mol(rdkit_molecule_a)

        # Add the DB conformers to the molecule.
        conformer_ids_a = []

        for db_conformer in db_conformers:

            rdkit_conformer = Chem.Conformer()

            for i, db_coordinate in enumerate(db_conformer.coordinates):

                rdkit_conformer.SetAtomPosition(
                    i,
                    Geometry.Point3D(db_coordinate.x, db_coordinate.y, db_coordinate.z),
                )

            conformer_ids_a.append(
                rdkit_molecule_a.AddConformer(rdkit_conformer, assignId=True)
            )

        # Add the conformers to add to the molecule.
        conformer_ids_b = []

        for record in conformers:

            rdkit_conformer = Chem.Conformer()

            for i in range(len(record.coordinates)):

                rdkit_conformer.SetAtomPosition(
                    i,
                    Geometry.Point3D(
                        record.coordinates[i, 0],
                        record.coordinates[i, 1],
                        record.coordinates[i, 2],
                    ),
                )

            conformer_ids_b.append(
                rdkit_molecule_b.AddConformer(rdkit_conformer, assignId=True)
            )

        # See if any of the conformers to add are already in the DB.
        matches = {}

        for j, id_b in enumerate(conformer_ids_b):

            matched_index = None

            for i, id_a in enumerate(conformer_ids_a):

                rms = AllChem.GetBestRMS(rdkit_molecule_a, rdkit_molecule_b, id_a, id_b)

                if rms >= 0.001:
                    continue

                matched_index = i
                break

            if matched_index is None:
                continue

            matches[j] = matched_index

        return matches

    @classmethod
    def _store_conformer_records(
        cls, smiles: str, db_parent: DBMoleculeRecord, records: List[ConformerRecord]
    ):

        # Find any matching conformers
        conformer_matches = cls._match_conformers(smiles, db_parent.conformers, records)

        # Create new database conformers for those unmatched conformers.
        missing_indices = {*range(len(records))} - {*conformer_matches}

        for index in missing_indices:
            # noinspection PyTypeChecker
            db_parent.conformers.append(
                DBConformerRecord(
                    coordinates=[
                        DBCoordinate(
                            x=records[index].coordinates[i, 0],
                            y=records[index].coordinates[i, 1],
                            z=records[index].coordinates[i, 2],
                        )
                        for i in range(len(records[index].coordinates))
                    ]
                )
            )
            conformer_matches[index] = len(db_parent.conformers) - 1

        for index, db_index in conformer_matches.items():

            db_record = db_parent.conformers[db_index]
            record = records[index]

            existing_charge_methods = [x.method for x in db_record.partial_charges]

            for partial_charges in record.partial_charges:

                if partial_charges.method in existing_charge_methods:
                    raise RuntimeError(
                        f"{partial_charges.method} charges already stored for {smiles}"
                    )

                db_record.partial_charges.append(
                    DBPartialChargeSet(
                        method=partial_charges.method,
                        values=[
                            DBPartialCharge(value=value)
                            for value in partial_charges.values
                        ],
                    )
                )

            existing_bond_methods = [x.method for x in db_record.bond_orders]

            for bond_orders in record.bond_orders:

                if bond_orders.method in existing_bond_methods:

                    raise RuntimeError(
                        f"{bond_orders.method} WBOs already stored for {smiles}"
                    )

                db_record.bond_orders.append(
                    DBWibergBondOrderSet(
                        method=bond_orders.method,
                        values=[
                            DBWibergBondOrder(
                                index_a=index_a, index_b=index_b, value=value
                            )
                            for (index_a, index_b, value) in bond_orders.values
                        ],
                    )
                )

    @classmethod
    def _store_records_with_formula(
        cls, db: Session, hill_formula: str, records: List[MoleculeRecord]
    ):
        """Stores a set of records which all store information for the same
        molecule.

        Parameters
        ----------
        db
            The current database session.
        hill_formula
            The indexed SMILES representation of the molecule.
        records
            The records to store.
        """

        # Partition the records by their unique SMILES patterns.
        records_by_smiles = defaultdict(list)

        for record in records:

            canonical_smiles = cls._to_canonical_smiles(record.smiles)
            records_by_smiles[canonical_smiles].append(record)

        # Find all of the records which have the same hill formula and partition them
        # by their unique, index-less SMILES patterns.
        existing_db_records: Collection[DBMoleculeRecord] = (
            db.query(DBMoleculeRecord)
            .filter(DBMoleculeRecord.hill_formula == hill_formula)
            .all()
        )

        db_records_by_smiles = {
            cls._to_canonical_smiles(db_record.smiles): db_record
            for db_record in existing_db_records
        }

        if len(db_records_by_smiles) != len(existing_db_records):
            # Sanity check that no two DB records have the same SMILES patterns
            # i.e. differ only by atom ordering.
            raise RuntimeError("The database is not self consistent.")

        # Create new DB records for molecules not yet in the DB.
        for smiles in {*records_by_smiles} - {*db_records_by_smiles}:

            db_records_by_smiles[smiles] = DBMoleculeRecord(
                hill_formula=hill_formula, smiles=records[0].smiles
            )

        # Handle the re-ordering of conformers to match the expected DB order.
        for smiles in records_by_smiles:

            db_record = db_records_by_smiles[smiles]

            # Retrieve the DB indexed SMILES which defines what ordering the
            # atoms in each record should have.
            expected_smiles = db_record.smiles

            # Re-order the records to match the database.
            conformer_records = [
                conformer_record
                for record in records_by_smiles[smiles]
                for conformer_record in record.reorder(expected_smiles).conformers
            ]

            cls._store_conformer_records(expected_smiles, db_record, conformer_records)

        for db_record in db_records_by_smiles.values():
            db.add(db_record)

    def store(self, *records: MoleculeRecord):
        """Store the molecules and their computed properties in the data store.

        Parameters
        ----------
        records
            The records to store.
        """

        # Validate and re-partition the records by their Hill formula.
        records_by_formula: Dict[str, List[MoleculeRecord]] = defaultdict(list)

        for record in records:

            record = record.copy(deep=True)
            records_by_formula[self._to_hill_formula(record.smiles)].append(record)

        # Store the records.
        with self._get_session() as db:

            for hill_formula in records_by_formula:

                self._store_records_with_formula(
                    db, hill_formula, records_by_formula[hill_formula]
                )

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
                            if partial_charge_method is None
                            or db_partial_charges.method == partial_charge_method
                        ],
                        bond_orders=[
                            WibergBondOrderSet(
                                method=db_bond_orders.method,
                                values=[
                                    (value.index_a, value.index_b, value.value)
                                    for value in db_bond_orders.values
                                ],
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
