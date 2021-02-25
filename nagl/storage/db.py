from sqlalchemy import Column, Float, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

DBBase = declarative_base()

DB_VERSION = 1


class DBCoordinate(DBBase):

    __tablename__ = "coordinates"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False)

    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    z = Column(Float, nullable=False)


class DBPartialCharge(DBBase):

    __tablename__ = "partial_charges"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("partial_charge_sets.id"), nullable=False)

    value = Column(Float, nullable=False)


class DBPartialChargeSet(DBBase):

    __tablename__ = "partial_charge_sets"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False)

    method = Column(String(10), nullable=False)

    values = relationship("DBPartialCharge")

    __table_args__ = (
        UniqueConstraint("parent_id", "method", name="_pc_parent_method_uc"),
    )


class DBWibergBondOrder(DBBase):

    __tablename__ = "wiberg_bond_orders"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("wiberg_bond_order_sets.id"), nullable=False)

    index_a = Column(Integer, nullable=False)
    index_b = Column(Integer, nullable=False)

    value = Column(Float, nullable=False)


class DBWibergBondOrderSet(DBBase):

    __tablename__ = "wiberg_bond_order_sets"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("conformers.id"), nullable=False)

    method = Column(String(10), nullable=False)

    values = relationship("DBWibergBondOrder")

    __table_args__ = (
        UniqueConstraint("parent_id", "method", name="_wbo_parent_method_uc"),
    )


class DBConformerRecord(DBBase):

    __tablename__ = "conformers"

    id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(String, ForeignKey("molecules.id"), nullable=False)

    coordinates = relationship("DBCoordinate")

    partial_charges = relationship("DBPartialChargeSet")
    bond_orders = relationship("DBWibergBondOrderSet")


class DBMoleculeRecord(DBBase):

    __tablename__ = "molecules"

    id = Column(Integer, primary_key=True, index=True)

    hill_formula = Column(String(20), nullable=False)
    smiles = Column(String, nullable=False)

    conformers = relationship("DBConformerRecord")


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
