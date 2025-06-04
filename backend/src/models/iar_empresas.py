from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class IarEmpresas(Base):
    __tablename__ = 'iar_empresas'

    ID_empresa = Column('ID_empresa', Integer, primary_key=True, autoincrement=False)
    Empresa_cve = Column('Empresa_cve', Text)
    Empresa_nombre = Column('Empresa_nombre', Text)
    Habitaciones_tot = Column('Habitaciones_tot', Integer)
    ID_clht = Column('ID_clht', Integer)
    Franquicia = Column('Franquicia', Text)
