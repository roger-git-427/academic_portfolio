# backend/main.py

import uvicorn
import asyncio
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.utils.schema import InferenceRequest, InferenceResponse
from app.services.inference import infer_model

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
