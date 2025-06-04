# app/db/__init__.py

from .base import Base
from .session import engine, AsyncSessionLocal, get_db

# Importa aquí cada módulo donde definas tus modelos,
# para que SQLAlchemy los «registre» en Base.metadata.
from app.models.iar_agencias import IarAgencias
from app.models.iar_canales import IarCanales
from app.models.iar_estatus_reservaciones import IarEstatusReservaciones
from app.models.iar_empresas import IarEmpresas
from app.models.reservaciones import Reservaciones


__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "IarAgencias",
    "IarCanales",
    "IarEstatusReservaciones",
    "IarEmpresas",
    "Reservaciones"
]
