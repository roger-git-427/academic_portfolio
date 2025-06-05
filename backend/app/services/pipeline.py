# backend/src/services/pipeline.py

import pandas as pd
import numpy as np
import math
import os
import pickle
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.raw_data import get_all_raw_data     # Para obtener raw tables :contentReference[oaicite:0]{index=0}
from app.services.preprocessing import (
    convert_dates,
    enforce_types_and_basic_filters,
    normalise_city,
    replace_h_num_persons,
    filtered_df,
    build_daily_occupancy,
    prepare_features,
    prepare_prophet_features_for_inference
)
from app.utils.schema import FullPipelineResponse, PredictionResult

async def run_full_pipeline(db: AsyncSession) -> FullPipelineResponse:
    """
    1. Obtiene raw data de todas las tablas via get_all_raw_data(db)
    2. Convierte cada lista de dicts a DataFrame
    3. Sobre df_reserv (raw), aplica convert_dates, enforce_types..., etc (sin merge)
    4. Genera daily_occupancy y features
    5. Carga Prophet y predice
    6. Devuelve un FullPipelineResponse con todos los DF serializados + predictions
    """

    # --------------------------------------------------------
    # 1) OBTENER RAW DATA
    # --------------------------------------------------------
    raw_data_obj = await get_all_raw_data(db)

    # Convertimos cada lista de dicts a DataFrame
    df_reserv_raw   = pd.DataFrame(raw_data_obj.reservaciones)
    df_canales_raw  = pd.DataFrame(raw_data_obj.iar_canales)
    df_empresas_raw = pd.DataFrame(raw_data_obj.iar_empresas)
    df_agencias_raw = pd.DataFrame(raw_data_obj.iar_agencias)
    df_estatus_raw  = pd.DataFrame(raw_data_obj.iar_estatus_reservaciones)

    # --------------------------------------------------------
    # 2) APLICAR PIPELINE DE LIMPIEZA SOLO A reservaciones
    #    (sin hacer merge con lookups)
    # --------------------------------------------------------
    # 2.1) Convertir fechas
    df_reserv_dates = convert_dates(df_reserv_raw)
    # 2.2) Typing + filtros básicos
    df_reserv_typed = enforce_types_and_basic_filters(df_reserv_dates)
    # 2.3) Normalizar ciudad
    df_reserv_norm  = normalise_city(df_reserv_typed)
    # 2.4) Ajustar h_num_per
    df_reserv_group = replace_h_num_persons(df_reserv_norm)
    # 2.5) Filtrar canceladas / habitaciones = 0
    df_reserv_filt  = filtered_df(df_reserv_group)

    # 2.6) Construir daily occupancy
    df_daily = build_daily_occupancy(df_reserv_filt, None, None)

    # 2.7) Preparar features finales
    df_features = prepare_features(df_daily)

    # --------------------------------------------------------
    # 3) GENERAR PREDICCIONES CON PROPHET
    # --------------------------------------------------------
    forecast_periods = 90
    df_future, regressors = prepare_prophet_features_for_inference(df_features, forecast_periods)

    model_path = "app/utils/models/prophet_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    with open(model_path, "rb") as f:
        prophet_model = pickle.load(f)

    forecast = prophet_model.predict(df_future)

    results: List[Dict[str, Any]] = []
    for _, row in forecast.iterrows():
        ds_str = row["ds"].strftime("%Y-%m-%d")
        yhat = row["yhat"]
        yhat_lower = row["yhat_lower"]
        yhat_upper = row["yhat_upper"]

        # Filtrar infinitos o NaN
        if not (np.isfinite(yhat) and np.isfinite(yhat_lower) and np.isfinite(yhat_upper)):
            continue

        results.append({
            "ds": ds_str,
            "yhat": float(yhat),
            "yhat_lower": float(yhat_lower),
            "yhat_upper": float(yhat_upper),
        })

    # --------------------------------------------------------
    # 4) SERIALIZAR TODOS LOS DataFrames A LISTA DE DICTS
    # --------------------------------------------------------
    def df_to_serializable(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Reemplaza inf/NaN por None y convierte tipos numpy a nativos,
        retornando una lista de diccionarios.
        """
        df2 = df.replace([np.inf, -np.inf], None)
        df2 = df2.where(pd.notnull(df2), None)
        raw_recs = df2.to_dict(orient="records")

        serializable: List[Dict[str, Any]] = []
        for rec in raw_recs:
            new_rec: Dict[str, Any] = {}
            for k, v in rec.items():
                if v is None:
                    new_rec[k] = None
                else:
                    try:
                        fval = float(v)
                        if not np.isfinite(fval):
                            new_rec[k] = None
                        else:
                            if isinstance(v, (int, np.integer)):
                                new_rec[k] = int(v)
                            else:
                                new_rec[k] = fval
                    except:
                        new_rec[k] = v
            serializable.append(new_rec)
        return serializable

    # Serializamos cada DataFrame
    raw_reserv_serial  = df_to_serializable(df_reserv_raw)
    raw_canales_serial  = df_to_serializable(df_canales_raw)
    raw_empresas_serial = df_to_serializable(df_empresas_raw)
    raw_agencias_serial = df_to_serializable(df_agencias_raw)
    raw_estatus_serial  = df_to_serializable(df_estatus_raw)

    clean_reserv_serial = df_to_serializable(df_reserv_filt)
    daily_serial        = df_to_serializable(df_daily)
    features_serial     = df_to_serializable(df_features)

    # --------------------------------------------------------
    # 5) CONSTRUIR OBJETO FullPipelineResponse
    # --------------------------------------------------------
    return FullPipelineResponse(
        raw_reservaciones = raw_reserv_serial,
        raw_iar_canales   = raw_canales_serial,
        raw_iar_empresas  = raw_empresas_serial,
        raw_iar_agencias  = raw_agencias_serial,
        raw_iar_estatus_reservaciones = raw_estatus_serial,

        clean_reservaciones = clean_reserv_serial,
        daily_occupancy     = daily_serial,
        features            = features_serial,
        predictions         = results
    )
