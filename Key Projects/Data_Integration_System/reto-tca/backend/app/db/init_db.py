# app/db/init_db.py
import asyncio
from .session import engine
from .base import Base

import app.db  # esto arrastra los modelos

async def init_models():
    """
    Crea todas las tablas definidas en Base.metadata.
    Úsalo sólo en desarrollo o pruebas; en producción, mejor usa Alembic.
    """
    async with engine.begin() as conn:
        # Ejecuta sincronously el método create_all()
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    asyncio.run(init_models())
