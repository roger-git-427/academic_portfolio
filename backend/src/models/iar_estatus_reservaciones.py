from sqlalchemy import Column, Integer, Text
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class IarEstatusReservaciones(Base):
    __tablename__ = 'iar_estatus_reservaciones'

    ID_estatus_reservaciones = Column('ID_estatus_reservaciones', Integer, primary_key=True, autoincrement=False)
    estatus_reservaciones_cve = Column('estatus_reservaciones_cve', Text)
    estatus_reservaciones = Column('estatus_reservaciones', Text)

