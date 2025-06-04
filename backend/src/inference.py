import pandas as pd
import pickle
import os
import asyncio
from typing import List

from src.preprocessing import (
    select_columns,
    merge_lookup_tables,
    convert_dates,
    enforce_types_and_basic_filters,
    normalise_city,
    replace_h_num_persons,
    filtered_df,
    build_daily_occupancy,
    prepare_features,
    prepare_prophet_features_for_inference
)
from src.utils.schema import InferenceRequest, PredictionResult


async def infer_model(request: InferenceRequest) -> List[PredictionResult]:
    """
    1) Toma request.data → DataFrame raw_reservaciones
    2) Carga los lookup tables desde CSV
    3) Aplica en serie cada nodo del pipeline para generar df_features
    4) Usa prepare_prophet_features_for_inference(...) para el DataFrame final
    5) Carga modelo Prophet entrenado y hace predict
    6) Devuelve la lista de PredictionResult(ds, yhat, yhat_lower, yhat_upper)
    """
    df_raw_reserv = pd.DataFrame(request.data)

    raw_canales = pd.read_csv("src/data/01_raw/iar_canales.csv")
    raw_empresas = pd.read_csv("src/data/01_raw/iar_empresas.csv")
    raw_agencias = pd.read_csv("src/data/01_raw/iar_Agencias.csv")
    raw_estatus = pd.read_csv("src/data/01_raw/iar_estatus_reservaciones.csv")


    df_selected = select_columns(df_raw_reserv)

    df_merged = merge_lookup_tables(
        reservations=df_selected,
        canales=raw_canales,
        empresas=raw_empresas,
        agencias=raw_agencias,
        estatus=raw_estatus,
    )

    df_dates = convert_dates(df_merged)
    df_typed = enforce_types_and_basic_filters(df_dates)
    df_norm = normalise_city(df_typed)
    df_grouped = replace_h_num_persons(df_norm)
    df_filtered = filtered_df(df_grouped)
    df_occupancy = build_daily_occupancy(df_filtered, None, None)
    df_features = prepare_features(df_occupancy)
    forecast_periods = 90  # ajusta según usaste en entrenamiento
    df_future, _ = prepare_prophet_features_for_inference(df_features, forecast_periods)

    print("[DEBUG] ran prepare_prophet_features_for_inference")

    model_path = "src/models/prophet_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    print("[DEBUG] found model path")

    with open(model_path, "rb") as f:
        prophet_model = pickle.load(f)

    print(f"[DEBUG] Loaded object from '{model_path}' is: {type(prophet_model)}")

    forecast = prophet_model.predict(df_future)

    print(f"[DEBUG] forecast '{forecast}' is type {type(forecast)}")

    results: List[PredictionResult] = []
    for _, row in forecast.iterrows():
        ds_str = row["ds"].strftime("%Y-%m-%d")
        results.append(
            PredictionResult(
                ds=ds_str,
                yhat=float(row["yhat"]),
                yhat_lower=float(row["yhat_lower"]),
                yhat_upper=float(row["yhat_upper"]),
            )
        )

    print(f"results {results}")

    return results




