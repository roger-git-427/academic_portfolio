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
    
class FullPipelineResponse(BaseModel):
    # 1) Raw data (tal cual llega de /raw_data)
    raw_reservaciones: List[Dict[str, Any]]
    raw_iar_canales:   List[Dict[str, Any]]
    raw_iar_empresas:  List[Dict[str, Any]]
    raw_iar_agencias:  List[Dict[str, Any]]
    raw_iar_estatus_reservaciones: List[Dict[str, Any]]

    # 2) DataFrame “limpio” de reservaciones (tras aplicar convert_dates, enforce_types, normalise_city, etc.)
    clean_reservaciones: List[Dict[str, Any]]

    # 3) Ocupación diaria construida
    daily_occupancy: List[Dict[str, Any]]

    # 4) DataFrame de features (tras prepare_features)
    features: List[Dict[str, Any]]

    # 5) Predicciones finales
    predictions: List[Dict[str, Union[str, float]]]