# schema.py

from typing import List, Dict, Union
from pydantic import BaseModel


class PredictionResult(BaseModel):
    ds: str
    yhat: float
    yhat_lower: float
    yhat_upper: float


class InferenceRequest(BaseModel):
    data: List[Dict[str, Union[float, int, str]]]  # JSON rows from DataFrame


class InferenceResponse(BaseModel):
    predictions: List[PredictionResult]
