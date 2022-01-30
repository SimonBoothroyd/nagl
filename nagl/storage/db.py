from sqlalchemy import Column, ForeignKey, Integer, PickleType, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

DBBase = declarative_base()

DB_VERSION = 1


class DBPartialChargeSet(DBBase):

    __tablename__ = "partial_charge_sets"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False, index=True)

    method = Column(String(10), nullable=False)
    values = Column(PickleType, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "method", name="_pc_parent_method_uc"),
    )


class DBWibergBondOrderSet(DBBase):

    __tablename__ = "wiberg_bond_order_sets"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False, index=True)

    method = Column(String(10), nullable=False)
    values = Column(PickleType, nullable=False)

    __table_args__ = (
        UniqueConstraint("parent_id", "method", name="_wbo_parent_method_uc"),
    )


class DBConformerRecord(DBBase):

    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("molecules.id"), nullable=False, index=True)

    coordinates = Column(PickleType, nullable=False)

    partial_charges = relationship("DBPartialChargeSet", cascade="all, delete-orphan")
    bond_orders = relationship("DBWibergBondOrderSet", cascade="all, delete-orphan")


class DBMoleculeRecord(DBBase):

    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    inchi_key = Column(String(20), nullable=False)
    smiles = Column(String, nullable=False)

    conformers = relationship("DBConformerRecord", cascade="all, delete-orphan")


class DBGeneralProvenance(DBBase):

    __tablename__ = "general_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBSoftwareProvenance(DBBase):

    __tablename__ = "software_provenance"

    key = Column(String, primary_key=True, index=True, unique=True)
    value = Column(String, nullable=False)

    parent_id = Column(Integer, ForeignKey("db_info.version"))


class DBInformation(DBBase):
    """A class which keeps track of the current database
    settings.
    """

    __tablename__ = "db_info"

    version = Column(Integer, primary_key=True)

    general_provenance = relationship(
        "DBGeneralProvenance", cascade="all, delete-orphan"
    )
    software_provenance = relationship(
        "DBSoftwareProvenance", cascade="all, delete-orphan"
    )
