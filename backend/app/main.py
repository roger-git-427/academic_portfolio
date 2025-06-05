# backend/main.py

import uvicorn
import asyncio
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.utils.schema import InferenceRequest, InferenceResponse, RawDataResponse
from app.services.inference import infer_model
from app.services.raw_data import get_all_raw_data 
from app.services.pipeline import run_full_pipeline

app = FastAPI()

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, db: AsyncSession = Depends(get_db)):
    try:
        # Ahora infer_model regresa (intermediate_data, predictions)
        preds = await infer_model(request, db)
        return InferenceResponse(
            predictions=preds
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        # En caso de error, devolvemos un objeto con listas vacías
        return InferenceResponse(predictions=[])

@app.get("/raw_data", response_model=RawDataResponse)
async def raw_data(db: AsyncSession = Depends(get_db)):
    """
    Retorna el contenido completo (raw) de todas las tablas del modelo,
    serializado como JSON.
    """
    try:
        return await get_all_raw_data(db)
    except Exception as e:
        print(f"Error al extraer raw data: {e}")
        # En caso de error, devolvemos listas vacías para cada tabla
        return RawDataResponse(
            reservaciones=[],
            iar_canales=[],
            iar_empresas=[],
            iar_agencias=[],
            iar_estatus_reservaciones=[]
        )

@app.get("/full_pipeline", response_model=FullPipelineResponse)
async def full_pipeline(db: AsyncSession = Depends(get_db)):
    """
    1. Obtiene raw data de todas las tablas
    2. Aplica limpieza y pipeline sin merges
    3. Genera predicciones
    4. Devuelve todos los DataFrames (raw, limpio, occupancy, features) y las predicciones
    """
    try:
        return await run_full_pipeline(db)
    except Exception as e:
        print(f"Error en full_pipeline: {e}")
        # En caso de fallo, devolvemos objetos vacíos para cada sección
        return FullPipelineResponse(
            raw_reservaciones=[],
            raw_iar_canales=[],
            raw_iar_empresas=[],
            raw_iar_agencias=[],
            raw_iar_estatus_reservaciones=[],
            clean_reservaciones=[],
            daily_occupancy=[],
            features=[],
            predictions=[]
        )
        
async def run_server():
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8080,
        workers=1,
        log_config=None,
        access_log=True,
    )
    await uvicorn.Server(config).serve()


if __name__ == "__main__":
    asyncio.run(run_server())
