# app/db/__init__.py

from .base import Base
from .session import engine, AsyncSessionLocal, get_db

# Importa aquí cada módulo donde definas tus modelos,
# para que SQLAlchemy los «registre» en Base.metadata.
from app.models.reservations import ReservationClean
from app.models.rooms import RoomsByDate

__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_db",
    "ReservationClean", # modelos
    "RoomsByDate",
]
