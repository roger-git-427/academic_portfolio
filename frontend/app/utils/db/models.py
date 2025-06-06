from sqlalchemy import Column, Integer, Text, Float
from .connection import Base

class Reservaciones(Base):
    __tablename__ = 'reservaciones'

    ID_Reserva = Column(Integer, primary_key=True)
    Fecha_hoy = Column(Text)
    h_res_fec = Column(Integer)
    h_res_fec_ok = Column(Text)
    h_res_fec_okt = Column(Text)
    h_num_per = Column(Integer)
    h_num_adu = Column(Integer)
    h_num_men = Column(Integer)
    h_num_noc = Column(Integer)
    h_tot_hab = Column(Integer)
    h_tfa_total = Column(Float)
    ID_Programa = Column(Integer)
    ID_Paquete = Column(Integer)
    ID_Segmento_Comp = Column(Integer)
    ID_Agencia = Column(Integer)
    ID_empresa = Column(Integer)
    ID_Tipo_Habitacion = Column(Integer)
    ID_canal = Column(Integer)
    ID_Pais_Origen = Column(Integer)
    ID_estatus_reservaciones = Column(Integer)
    h_fec_lld = Column(Text)
    h_fec_reg = Column(Text)
    h_fec_sda = Column(Text)

class IarCanales(Base):
    __tablename__ = 'iar_canales'

    ID_canal = Column(Integer, primary_key=True)
    Canal_nombre = Column(Text)

class IarEmpresas(Base):
    __tablename__ = 'iar_empresas'

    ID_empresa = Column(Integer, primary_key=True)
    Empresa_nombre = Column(Text)
    Habitaciones_tot = Column(Integer)

class IarAgencias(Base):
    __tablename__ = 'iar_agencias'

    ID_Agencia = Column(Integer, primary_key=True)
    Agencia_nombre = Column(Text)
    Ciudad_Nombre = Column(Text)

class IarEstatusReservaciones(Base):
    __tablename__ = 'iar_estatus_reservaciones'

    ID_estatus_reservaciones = Column(Integer, primary_key=True)
    estatus_reservaciones = Column(Text)