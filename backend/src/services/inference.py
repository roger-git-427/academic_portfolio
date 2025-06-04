# backend/src/services/inference.py

import pandas as pd
import pickle
import os
import asyncio
from typing import List

from sqlalchemy import text                       # <-- [CAMBIO] para ejecutar queries raw
from sqlalchemy.ext.asyncio import AsyncSession  # <-- [CAMBIO] para tipar la sesión asíncrona

# Importamos las funciones del pipeline sin cambios
from backend.src.services.preprocessing import (
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


# -- Firma modificada: ahora recibe también db: AsyncSession --
async def infer_model(request: InferenceRequest, db: AsyncSession) -> List[PredictionResult]:
    """
    1) Obtiene request.data → NO se usa: ahora vamos a consultar la tabla directamente.
    2) Ejecuta query a la tabla `reservaciones` para obtener todos los registros con h_fec_reg < '2021-03-01'
    3) Ejecuta queries a los lookup tables:
         - iar_canales
         - iar_empresas
         - iar_agencias
         - iar_estatus_reservaciones
    4) Arma los DataFrames a partir de los ResultProxy de SQLAlchemy.
    5) Aplica en serie cada nodo del pipeline para generar df_features.
    6) Usa prepare_prophet_features_for_inference(...) para el DataFrame final.
    7) Carga modelo Prophet entrenado y hace predict.
    8) Devuelve la lista de PredictionResult(ds, yhat, yhat_lower, yhat_upper)
    """

    # --------------------------------------------------------
    # 1) CONSULTA A LA TABLA `reservaciones`
    # --------------------------------------------------------
    # Hacemos una consulta asíncrona para traer todos los campos necesarios
    # Solo obtenemos aquellos registros con h_fec_reg < '2021-03-01'.
    #
    # Nota: ajusta el nombre exacto de la columna (h_fec_reg) si en tu BD es DATE o TEXT.
    #
    sql_reserv = text("""
        SELECT
          ID_Reserva,
          Fecha_hoy,
          h_res_fec,
          h_num_per,
          h_num_adu,
          h_num_men,
          h_num_noc,
          h_tot_hab,
          ID_Programa,
          ID_Paquete,
          ID_Segmento_Comp,
          ID_Agencia,
          ID_empresa,
          ID_Tipo_Habitacion,
          ID_canal,
          ID_Pais_Origen,
          ID_estatus_reservaciones,
          h_fec_lld,
          h_fec_reg,
          h_fec_sda
        FROM reservaciones
        WHERE h_fec_reg < '2021-03-01';
    """)
    result_reserv = await db.execute(sql_reserv)
    rows_reserv = result_reserv.fetchall()
    cols_reserv = result_reserv.keys()

    # Construimos un DataFrame de pandas con el resultado de la consulta
    df_raw_reserv = pd.DataFrame(rows_reserv, columns=cols_reserv)
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 2) CONSULTA A LOS LOOKUP TABLES
    # --------------------------------------------------------
    # 2.1) iar_canales: necesitamos solo ID_canal y Canal_nombre
    sql_canales = text("SELECT ID_canal, Canal_nombre FROM iar_canales;")
    result_canales = await db.execute(sql_canales)
    rows_canales = result_canales.fetchall()
    cols_canales = result_canales.keys()
    df_canales = pd.DataFrame(rows_canales, columns=cols_canales)

    # 2.2) iar_empresas: necesitamos ID_empresa, Empresa_nombre, Habitaciones_tot
    sql_empresas = text("""
        SELECT ID_empresa, Empresa_nombre, Habitaciones_tot
        FROM iar_empresas;
    """)
    result_empresas = await db.execute(sql_empresas)
    rows_empresas = result_empresas.fetchall()
    cols_empresas = result_empresas.keys()
    df_empresas = pd.DataFrame(rows_empresas, columns=cols_empresas)

    # 2.3) iar_agencias: necesitamos ID_Agencia, Agencia_nombre, Ciudad_Nombre
    sql_agencias = text("""
        SELECT ID_Agencia, Agencia_nombre, Ciudad_Nombre
        FROM iar_agencias;
    """)
    result_agencias = await db.execute(sql_agencias)
    rows_agencias = result_agencias.fetchall()
    cols_agencias = result_agencias.keys()
    df_agencias = pd.DataFrame(rows_agencias, columns=cols_agencias)

    # 2.4) iar_estatus_reservaciones: necesitamos ID_estatus_reservaciones, estatus_reservaciones
    sql_estatus = text("""
        SELECT ID_estatus_reservaciones, estatus_reservaciones
        FROM iar_estatus_reservaciones;
    """)
    result_estatus = await db.execute(sql_estatus)
    rows_estatus = result_estatus.fetchall()
    cols_estatus = result_estatus.keys()
    df_estatus = pd.DataFrame(rows_estatus, columns=cols_estatus)
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 3) EJECUTAMOS EL PIPELINE DE PREPROCESAMIENTO
    # --------------------------------------------------------
    # 3.1) Seleccionar columnas básicas de reservaciones
    df_selected = select_columns(df_raw_reserv)

    # 3.2) Merge con lookup tables
    df_merged = merge_lookup_tables(
        reservations=df_selected,
        canales=df_canales,
        empresas=df_empresas,
        agencias=df_agencias,
        estatus=df_estatus,
    )

    # 3.3) Convertir fechas
    df_dates = convert_dates(df_merged)

    # 3.4) Aplicar tipado y filtros básicos
    df_typed = enforce_types_and_basic_filters(df_dates)

    # 3.5) Normalizar ciudad
    df_norm = normalise_city(df_typed)

    # 3.6) Ajustar "h_num_per" si es menor que suma de adultos+menores
    df_grouped = replace_h_num_persons(df_norm)

    # 3.7) Filtrar reservaciones canceladas, etc.
    df_filtered = filtered_df(df_grouped)

    # 3.8) Construir daily occupancy
    df_occupancy = build_daily_occupancy(df_filtered, None, None)

    # 3.9) Preparar features finales
    df_features = prepare_features(df_occupancy)
    # --------------------------------------------------------

    # --------------------------------------------------------
    # 4) PREPARAMOS EL DATAFRAME PARA INFERENCIA CON PROPHET
    # --------------------------------------------------------
    forecast_periods = 90  # mismo valor que en entrenamiento
    df_future, regressors = prepare_prophet_features_for_inference(df_features, forecast_periods)

    # Ruta fija al modelo Prophet entrenado
    model_path = "src/models/prophet_model.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}")

    with open(model_path, "rb") as f:
        prophet_model = pickle.load(f)

    # 4.1) Generamos el forecast
    forecast = prophet_model.predict(df_future)

    # 4.2) Construir lista de PredictionResult
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

    return results
