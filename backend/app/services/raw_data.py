# backend/src/services/raw_data.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.schema import RawDataResponse

from app.services.preprocessing import (
    merge_lookup_tables,
    convert_dates,
    enforce_types_and_basic_filters,
    normalise_city,
    replace_h_num_persons,
    filtered_df,
)

async def get_all_raw_data(db: AsyncSession) -> RawDataResponse:
    """
    Consulta todas las tablas y retorna un objeto RawDataResponse,
    donde cada atributo es una lista de diccionarios con tipos nativos de Python.
    """

    # --------------------------------------------------------
    # 1) Consultar la tabla `reservaciones`
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
        "h_tfa_total",
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
        WHERE "h_fec_lld" < '2020-06-01';
    """)
    result_reserv = await db.execute(sql_reserv)
    rows_reserv   = result_reserv.fetchall()
    cols_reserv   = result_reserv.keys()
    df_reserv     = pd.DataFrame(rows_reserv, columns=cols_reserv)
    print(f"[DEBUG] df_reserv.shape      = {df_reserv.shape}")
    print(f"[DEBUG] df_reserv.dtypes:\n{df_reserv.dtypes}\n")

    # --------------------------------------------------------
    # 2) Consultar lookup tables
    # --------------------------------------------------------
    sql_canales = text('SELECT "ID_canal", "Canal_nombre" FROM "iar_canales";')
    result_canales = await db.execute(sql_canales)
    df_canales     = pd.DataFrame(result_canales.fetchall(), columns=result_canales.keys())
    print(f"[DEBUG] df_canales.shape    = {df_canales.shape}")

    sql_empresas = text('SELECT "ID_empresa", "Empresa_nombre", "Habitaciones_tot" FROM "iar_empresas";')
    result_empresas = await db.execute(sql_empresas)
    df_empresas     = pd.DataFrame(result_empresas.fetchall(), columns=result_empresas.keys())
    print(f"[DEBUG] df_empresas.shape   = {df_empresas.shape}")

    sql_agencias = text('SELECT "ID_Agencia", "Agencia_nombre", "Ciudad_Nombre" FROM "iar_agencias";')
    result_agencias = await db.execute(sql_agencias)
    df_agencias     = pd.DataFrame(result_agencias.fetchall(), columns=result_agencias.keys())
    print(f"[DEBUG] df_agencias.shape   = {df_agencias.shape}")

    sql_estatus = text('SELECT "ID_estatus_reservaciones", "estatus_reservaciones" FROM "iar_estatus_reservaciones";')
    result_estatus = await db.execute(sql_estatus)
    df_estatus     = pd.DataFrame(result_estatus.fetchall(), columns=result_estatus.keys())
    print(f"[DEBUG] df_estatus.shape    = {df_estatus.shape}\n")

    # --------------------------------------------------------
    # 3) Preprocesar todo para obtener `processed_data`
    # --------------------------------------------------------
    df_merged = merge_lookup_tables(
        reservations= df_reserv,
        canales      = df_canales,
        empresas     = df_empresas,
        agencias     = df_agencias,
        estatus      = df_estatus,
    )
    print(f"[DEBUG] df_merged.shape      = {df_merged.shape}")

    df_dates = convert_dates(df_merged)
    print(f"[DEBUG] df_dates.shape       = {df_dates.shape}")
    print(f"[DEBUG] df_dates.dtypes:\n{df_dates.dtypes}\n")

    df_typed = enforce_types_and_basic_filters(df_dates)
    print(f"[DEBUG] df_typed.shape       = {df_typed.shape}")
    print(f"[DEBUG] df_typed.dtypes:\n{df_typed.dtypes}\n")

    df_norm = normalise_city(df_typed)
    print(f"[DEBUG] df_norm.shape        = {df_norm.shape}")

    df_grouped = replace_h_num_persons(df_norm)
    print(f"[DEBUG] df_grouped.shape     = {df_grouped.shape}")

    df_filtered = filtered_df(df_grouped)
    print(f"[DEBUG] df_filtered.shape    = {df_filtered.shape}\n")

    # --------------------------------------------------------
    # 4) Preparar la parte “raw” de reservaciones: parsear fechas en df_reserv
    # --------------------------------------------------------
    df_reserv_parsed = df_reserv.copy()
    df_reserv_parsed = convert_dates(df_reserv_parsed)
    print(f"[DEBUG] df_reserv_parsed.shape   = {df_reserv_parsed.shape}")
    print(f"[DEBUG] df_reserv_parsed.dtypes:\n{df_reserv_parsed.dtypes}\n")
    print(f"[DEBUG] Ejemplo de filas dates raw:\n{df_reserv_parsed[['h_fec_lld','h_fec_reg','h_fec_sda']].head(3)}\n")

    # --------------------------------------------------------
    # 5) Convertir todas las columnas datetime64[ns] a string ISO y todas las categorical → str
    # --------------------------------------------------------
    from pandas.api.types import (
        is_datetime64_any_dtype,
        is_categorical_dtype,
        is_integer_dtype,
        is_float_dtype,
    )

    # 5.1) Fechas a ISO-string
    for col in df_reserv_parsed.columns:
        if is_datetime64_any_dtype(df_reserv_parsed[col].dtype):
            df_reserv_parsed[col] = df_reserv_parsed[col].dt.strftime("%Y-%m-%d")
            print(f"[DEBUG] Columna '{col}' convertida a str de fecha (ISO).")
        elif is_categorical_dtype(df_reserv_parsed[col].dtype):
            df_reserv_parsed[col] = df_reserv_parsed[col].astype(str)
            print(f"[DEBUG] Columna '{col}' era Categorical → convertida a str.")
    print()

    # 5.2) Desplegar numpy.int64/yfloat64 → Python int/float y NaN → None
    def to_python_scalar(x: Any) -> Any:
        if isinstance(x, (np.integer, np.floating)):
            return x.item()
        if isinstance(x, float) and np.isnan(x):
            return None
        return x

    # Primero, reemplazar NaN (numpy) por None
    df_reserv_parsed = df_reserv_parsed.where(pd.notnull(df_reserv_parsed), None)
    # Luego aplicar to_python_scalar a cada celda
    df_reserv_parsed = df_reserv_parsed.applymap(to_python_scalar)

    # 5.3) Ahora convierto a lista de dicts
    reservaciones_list = df_reserv_parsed.to_dict(orient="records")
    print(f"[DEBUG] reservaciones_list[0] (ejemplo):\n{reservaciones_list[0]}\n")

    # --------------------------------------------------------
    # 6) Hacer lo mismo con las tablas lookup (aunque usualmente no hay datetime allí)
    # --------------------------------------------------------
    def clean_df(df: pd.DataFrame) -> List[Dict[str,Any]]:
        df2 = df.copy()
        for c in df2.columns:
            if is_datetime64_any_dtype(df2[c].dtype):
                df2[c] = df2[c].dt.strftime("%Y-%m-%d")
            elif is_categorical_dtype(df2[c].dtype):
                df2[c] = df2[c].astype(str)

            # Reemplazar NaN → None
            df2[c] = df2[c].where(pd.notnull(df2[c]), None)
            # Convertir numpy.* → Python
            df2[c] = df2[c].map(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)

        return df2.to_dict(orient="records")

    canales_list   = clean_df(df_canales)
    empresas_list  = clean_df(df_empresas)
    agencias_list  = clean_df(df_agencias)
    estatus_list   = clean_df(df_estatus)

    # --------------------------------------------------------
    # 7) Y también limpiar `df_filtered` para enviarlo como processed_data
    # --------------------------------------------------------
    df_proc = df_filtered.copy()
    for c in df_proc.columns:
        if is_datetime64_any_dtype(df_proc[c].dtype):
            df_proc[c] = df_proc[c].dt.strftime("%Y-%m-%d")
        elif is_categorical_dtype(df_proc[c].dtype):
            df_proc[c] = df_proc[c].astype(str)

        df_proc[c] = df_proc[c].where(pd.notnull(df_proc[c]), None)
        df_proc[c] = df_proc[c].map(lambda x: x.item() if isinstance(x, (np.integer, np.floating)) else x)

    processed_list = df_proc.to_dict(orient="records")
    print(f"[DEBUG] reservaciones_list[0] (ejemplo):\n{reservaciones_list[0]}\n")
    print(f"[DEBUG] canales_list[0] (ejemplo):\n{canales_list[0]}\n")
    print(f"[DEBUG] empresas_list[0] (ejemplo):\n{empresas_list[0]}\n")
    print(f"[DEBUG] agencias_list[0] (ejemplo):\n{agencias_list[0]}\n")
    print(f"[DEBUG] estatus_list[0] (ejemplo):\n{estatus_list[0]}\n")
    print(f"[DEBUG] processed_list[0] (ejemplo):\n{processed_list[0]}\n")
    # --------------------------------------------------------
    # 8) Devolver el RawDataResponse con listas “limpias”
    # --------------------------------------------------------
    return RawDataResponse(
        reservaciones= reservaciones_list,
        iar_canales  = canales_list,
        iar_empresas = empresas_list,
        iar_agencias = agencias_list,
        iar_estatus_reservaciones = estatus_list,
        processed_data= processed_list,
    )