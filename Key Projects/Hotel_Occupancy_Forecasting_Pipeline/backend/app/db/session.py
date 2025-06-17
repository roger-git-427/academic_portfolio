# app/db/session.py
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# Usar la URL montada en settings (PostgreSQL Azure)
DATABASE_URL = settings.DB_URL

# Crear motor asíncrono
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
)

# Sesiones asíncronas
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Dependencia para inyectar la sesión en los endpoints
async def get_db():
    """
    Dependency - proporciona una sesión de base de datos por petición
    """
    async with AsyncSessionLocal() as session:
        yield session