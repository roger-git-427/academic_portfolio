# backend/src/services/inference.py

import pandas as pd
import numpy as np
import math
import numbers
import pickle
import os
import asyncio
from typing import List

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Importamos las funciones del pipeline sin cambios
from app.services.preprocessing import (
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
from app.utils.schema import InferenceRequest, PredictionResult


async def infer_model(request: InferenceRequest, db: AsyncSession) -> (List[PredictionResult]):
    """
    1) Consulta la tabla `reservaciones` con h_fec_reg < '2021-03-01'
    2) Consulta los lookup tables (iar_canales, iar_empresas, iar_agencias, iar_estatus_reservaciones)
    3) Arma DataFrames y ejecuta el pipeline completo
    4) Genera forecast con Prophet
    5) Devuelve:
       - intermediate_data: lista de dicts para el DataFrame final de preprocesamiento (df_features)
       - results: lista de PredictionResult
    """

    # --------------------------------------------------------
    # 1) CONSULTA A LA TABLA `reservaciones` (columnas con comillas dobles)
    # --------------------------------------------------------
    sql_reserv = text("""
        SELECT
          "ID_Reserva",
          "Fecha_hoy",
          "h_res_fec",
          "h_num_per",
          "h_num_adu",
          "h_num_men",
          "h_num_noc",
          "h_tot_hab",
          "ID_Programa",
          "ID_Paquete",
          "ID_Segmento_Comp",
          "ID_Agencia",
          "ID_empresa",
          "ID_Tipo_Habitacion",
          "ID_canal",
          "ID_Pais_Origen",
          "ID_estatus_reservaciones",
          "h_fec_lld",
          "h_fec_reg",
          "h_fec_sda"
        FROM "reservaciones"
        WHERE "h_fec_reg" < '2021-03-01';
    """)
    result_reserv = await db.execute(sql_reserv)
    rows_reserv = result_reserv.fetchall()
    cols_reserv = result_reserv.keys()

    df_raw_reserv = pd.DataFrame(rows_reserv, columns=cols_reserv)
    print(f"[DEBUG] df_raw_reserv.shape = {df_raw_reserv.shape}")
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 2) CONSULTA A LOS LOOKUP TABLES (también con comillas dobles)
    # --------------------------------------------------------
    sql_canales = text("""
        SELECT "ID_canal", "Canal_nombre"
        FROM "iar_canales";
    """)
    result_canales = await db.execute(sql_canales)
    rows_canales = result_canales.fetchall()
    cols_canales = result_canales.keys()
    df_canales = pd.DataFrame(rows_canales, columns=cols_canales)
    print(f"[DEBUG] df_canales.shape     = {df_canales.shape}")

    sql_empresas = text("""
        SELECT "ID_empresa", "Empresa_nombre", "Habitaciones_tot"
        FROM "iar_empresas";
    """)
    result_empresas = await db.execute(sql_empresas)
    rows_empresas = result_empresas.fetchall()
    cols_empresas = result_empresas.keys()
    df_empresas = pd.DataFrame(rows_empresas, columns=cols_empresas)
    print(f"[DEBUG] df_empresas.shape    = {df_empresas.shape}")

    sql_agencias = text("""
        SELECT "ID_Agencia", "Agencia_nombre", "Ciudad_Nombre"
        FROM "iar_agencias";
    """)
    result_agencias = await db.execute(sql_agencias)
    rows_agencias = result_agencias.fetchall()
    cols_agencias = result_agencias.keys()
    df_agencias = pd.DataFrame(rows_agencias, columns=cols_agencias)
    print(f"[DEBUG] df_agencias.shape    = {df_agencias.shape}")

    sql_estatus = text("""
        SELECT "ID_estatus_reservaciones", "estatus_reservaciones"
        FROM "iar_estatus_reservaciones";
    """)
    result_estatus = await db.execute(sql_estatus)
    rows_estatus = result_estatus.fetchall()
    cols_estatus = result_estatus.keys()
    df_estatus = pd.DataFrame(rows_estatus, columns=cols_estatus)
    print(f"[DEBUG] df_estatus.shape     = {df_estatus.shape}")
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 3) EJECUTAMOS EL PIPELINE DE PREPROCESAMIENTO
    # --------------------------------------------------------
    df_selected = select_columns(df_raw_reserv)
    print(f"[DEBUG] df_selected.shape    = {df_selected.shape}")

    df_merged = merge_lookup_tables(
        reservations=df_selected,
        canales=df_canales,
        empresas=df_empresas,
        agencias=df_agencias,
        estatus=df_estatus,
    )
    print(f"[DEBUG] df_merged.shape      = {df_merged.shape}")

    df_dates = convert_dates(df_merged)
    print(f"[DEBUG] df_dates.shape       = {df_dates.shape}")

    df_typed = enforce_types_and_basic_filters(df_dates)
    print(f"[DEBUG] df_typed.shape       = {df_typed.shape}")

    df_norm = normalise_city(df_typed)
    print(f"[DEBUG] df_norm.shape        = {df_norm.shape}")

    df_grouped = replace_h_num_persons(df_norm)
    print(f"[DEBUG] df_grouped.shape     = {df_grouped.shape}")

    df_filtered = filtered_df(df_grouped)
    print(f"[DEBUG] df_filtered.shape    = {df_filtered.shape}")

    df_occupancy = build_daily_occupancy(df_filtered, None, None)
    print(f"[DEBUG] df_occupancy.shape   = {df_occupancy.shape}")

    df_features = prepare_features(df_occupancy)
    print(f"[DEBUG] df_features.shape    = {df_features.shape}")
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 4) PREPARAR DATOS PARA PROPHET Y HACER PREDICCIÓN
    # --------------------------------------------------------
    forecast_periods = 90
    df_future, regressors = prepare_prophet_features_for_inference(df_features, forecast_periods)

    model_path = "app/utils/models/prophet_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    with open(model_path, "rb") as f:
        prophet_model = pickle.load(f)

    forecast = prophet_model.predict(df_future)

    results: List[PredictionResult] = []
    for _, row in forecast.iterrows():
        # Convertir columna ds a string
        ds_str = row["ds"].strftime("%Y-%m-%d")
        yhat = row["yhat"]
        yhat_lower = row["yhat_lower"]
        yhat_upper = row["yhat_upper"]

        # Saltar cualquier fila con valores infinitos o NaN
        if not (np.isfinite(yhat) and np.isfinite(yhat_lower) and np.isfinite(yhat_upper)):
            continue

        results.append(
            PredictionResult(
                ds=ds_str,
                yhat=float(yhat),
                yhat_lower=float(yhat_lower),
                yhat_upper=float(yhat_upper),
            )
        )


    # Devuelve una tupla: (intermediate_data, lista de PredictionResult)
    return results
