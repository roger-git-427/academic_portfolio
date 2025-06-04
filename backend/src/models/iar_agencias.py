from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class IarAgencias(Base):
    __tablename__ = 'iar_agencias'

    ID_Agencia = Column('ID_Agencia', Integer, primary_key=True, autoincrement=False)
    Hotel_cve = Column('Hotel_cve', Text)
    Agencia_cve = Column('Agencia_cve', Text)
    Agencia_nombre = Column('Agencia_nombre', Text)
    Ciudad_Nombre = Column('Ciudad_Nombre', Text)
    Estado_cve = Column('Estado_cve', Text)
    Pais_cod = Column('Pais_cod', Text)
    ID_Entidad_Fed_Agencia = Column('ID_Entidad_Fed_Agencia', Integer)
    ID_Vendedor = Column('ID_Vendedor', Integer)
