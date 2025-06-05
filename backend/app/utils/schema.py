# schema.py

from typing import List, Dict, Union, Any, Optional
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


class RawDataResponse(BaseModel):
    reservaciones: List[Dict[str, Any]]
    iar_canales: List[Dict[str, Any]]
    iar_empresas: List[Dict[str, Any]]
    iar_agencias: List[Dict[str, Any]]
    iar_estatus_reservaciones: List[Dict[str, Any]]
    processed_data: List[Dict[str, Any]] 
    