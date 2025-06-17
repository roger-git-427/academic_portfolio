from pydantic_settings import BaseSettings
from pydantic import field_validator
from typing import List, Optional, Union


class Settings(BaseSettings):
    # Información de la aplicación
    PROJECT_NAME: str = "MiAPI"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    # Configuración de Base de Datos
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: str = "5432"
    DB_NAME: str
    DB_URL: Optional[str] = None  # Se ensambla a partir de las otras vars si no se proporciona

    # Seguridad
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60  # Tiempo de expiración del token JWT en minutos

    # CORS
    BACKEND_CORS_ORIGINS_STR: str = ""

    # Ruta al modelo de ML
    MODEL_PATH: str = "app/utils/models/prophet_model.pkl"

    model_config = {
        "env_file": ".env",
        "case_sensitive": True
    }

    @field_validator("DB_URL", mode="before")
    @classmethod
    def assemble_db_url(cls, v, info):
        """
        Si no se proporciona DB_URL directamente, ensamblar a partir de vars individuales.
        """
        if isinstance(v, str):
            return v
            
        values = info.data
        
        return f"postgresql+asyncpg://{values.get('DB_USER')}:{values.get('DB_PASSWORD')}@{values.get('DB_HOST')}:{values.get('DB_PORT')}/{values.get('DB_NAME')}"


    @property
    def BACKEND_CORS_ORIGINS(self) -> List[str]:
        if not self.BACKEND_CORS_ORIGINS_STR:
            return []
        return [i.strip() for i in self.BACKEND_CORS_ORIGINS_STR.split(",")]


settings = Settings()
