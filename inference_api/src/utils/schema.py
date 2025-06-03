# schema.py

from typing import List, Dict, Union
from pydantic import BaseModel


# Entrada esperada: lista de diccionarios con las features
class InferenceRequest(BaseModel):
    data: List[Dict[str, Union[int, float]]]


# Salida esperada: una lista de predicciones (una por fila)
class InferenceResponse(BaseModel):
    predictions: List[float]
