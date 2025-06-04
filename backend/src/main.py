# main.py
import uvicorn
import asyncio
from fastapi import FastAPI
from src.utils.schema import InferenceRequest, InferenceResponse
from src.inference import infer_model

app = FastAPI()


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    try:
        result = await infer_model(request)
        return InferenceResponse(predictions=result)
    except Exception as e:
        print(f"Error during prediction: {e}")
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
