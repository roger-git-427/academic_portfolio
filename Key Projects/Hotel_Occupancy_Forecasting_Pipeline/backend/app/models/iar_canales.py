from sqlalchemy import Column, Integer, Text
from app.db.base import Base

class IarCanales(Base):
    __tablename__ = 'iar_canales'

    ID_canal = Column('ID_canal', Integer, primary_key=True, autoincrement=False)
    Canal_cve = Column('Canal_cve', Text)
    Canal_nombre = Column('Canal_nombre', Text)
