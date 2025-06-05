# frontend/utils/data_extractor.py

from pathlib import Path
from functools import lru_cache
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests

# Directorio de datos por defecto (carpeta 'data' en el nivel superior del proyecto)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
API_BASE_URL = "http://127.0.0.1:8000"


@lru_cache(maxsize=1)
def load_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Carga todos los DataFrames vía GET /raw_data de la API y
    devuelve un diccionario con DataFrames ya listos para usar.
    """
    url = f"{API_BASE_URL}/raw_data"
    resp = requests.get(url)
    resp.raise_for_status()
    raw_json = resp.json()

    # 1) Convertir cada lista de JSON a DataFrame
    df_res = pd.DataFrame(raw_json.get("reservaciones", []))
    print(f"[DEBUG] Reservaciones cargadas: {len(df_res)} filas")
    df_est = pd.DataFrame(raw_json.get("iar_estatus_reservaciones", []))
    df_ages = pd.DataFrame(raw_json.get("iar_agencias", []))
    df_emp = pd.DataFrame(raw_json.get("iar_empresas", []))
    df_can = pd.DataFrame(raw_json.get("iar_canales", []))

    # 2) Convertir todas las columnas que contienen 'fec' a datetime
    cols_fec = [col for col in df_res.columns if "fec" in col.lower()]
    print(f"[DEBUG load_data] Columnas con 'fec' detectadas: {cols_fec}")
    for col in cols_fec:
        df_res[col] = pd.to_datetime(df_res[col], errors="coerce")
        n_null = df_res[col].isna().sum()
        print(f"   [DEBUG load_data] Columna '{col}' a datetime. Nulos:**{n_null}**")

    # 3) Definir 'fecha_checkin' — AHORA solo existe 'h_fec_lld'
    if "h_fec_lld" in df_res.columns:
        df_res["fecha_checkin"] = df_res["h_fec_lld"]
        print(f"[DEBUG load_data] Usando 'h_fec_lld' como fecha_checkin → filas válidas: {df_res['fecha_checkin'].notna().sum()}")
    else:
        df_res["fecha_checkin"] = pd.NaT
        print("[WARN load_data] No existe 'h_fec_lld'; todas NaT en fecha_checkin")

    # 4) Eliminar filas sin fecha_checkin válida
    df_res = df_res.dropna(subset=["fecha_checkin"]).reset_index(drop=True)
    print(f"[DEBUG load_data] Reservaciones tras dropna(fecha_checkin): {len(df_res)} filas")

    # 5) Forzar tipos de ID a Int16(Nullable)
    for col in ["ID_empresa", "ID_canal", "ID_Agencia", "ID_estatus_reservaciones"]:
        if col in df_res.columns:
            df_res[col] = pd.to_numeric(df_res[col], errors="coerce").astype("Int16")
            print(f"[DEBUG load_data] Columna '{col}' → Int16(nullable)")

    # 6) Normalizar IDs también para los lookup tables
    if "ID_empresa" in df_emp.columns:
        df_emp["ID_empresa"] = pd.to_numeric(df_emp["ID_empresa"], errors="coerce").astype("Int16")
    if "ID_canal" in df_can.columns:
        df_can["ID_canal"] = pd.to_numeric(df_can["ID_canal"], errors="coerce").astype("Int16")
    if "ID_Agencia" in df_ages.columns:
        df_ages["ID_Agencia"] = pd.to_numeric(df_ages["ID_Agencia"], errors="coerce").astype("Int16")
    print("[DEBUG load_data] Tipos de ID convertidos en lookup tables donde corresponde")

    # 7) Unir descripción del estatus
    if not df_est.empty and "ID_estatus_reservaciones" in df_res.columns:
        df_res = df_res.merge(
            df_est[["ID_estatus_reservaciones", "estatus_reservaciones"]],
            on="ID_estatus_reservaciones",
            how="left"
        )
        df_res = df_res.rename(columns={"estatus_reservaciones": "descripcion"})
        print(f"[DEBUG load_data] Descripciones añadidas (non-null): {df_res['descripcion'].notna().sum()} filas")
    else:
        print("[WARN load_data] No se pudo unir descripción de estatus")

    return {
        "reservaciones": df_res,
        "estatus": df_est,
        "agencias": df_ages,
        "empresas": df_emp,
        "canales": df_can
    }


def filtrar_datos(
    df: pd.DataFrame,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    empresas_sel: Optional[List[Any]] = None,
    canales_sel: Optional[List[Any]] = None,
    agencias_sel: Optional[List[Any]] = None
) -> pd.DataFrame:
    """
    Filtra el DataFrame de reservaciones según rango de fechas y selecciones.
    Args:
        df: DataFrame de reservaciones (debe contener 'fecha_checkin').
        start_date: Fecha mínima de check-in.
        end_date: Fecha máxima de check-in.
        empresas_sel: Lista de IDs de empresa a filtrar.
        canales_sel: Lista de IDs de canal a filtrar.
        agencias_sel: Lista de IDs de agencia a filtrar.
    Returns:
        DataFrame filtrado.
    """
    df_f = df.copy()
    if start_date is not None:
        df_f = df_f[df_f["fecha_checkin"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_f = df_f[df_f["fecha_checkin"] <= pd.to_datetime(end_date)]
    if empresas_sel:
        df_f = df_f[df_f["ID_empresa"].isin(empresas_sel)]
    if canales_sel:
        df_f = df_f[df_f["ID_canal"].isin(canales_sel)]
    if agencias_sel:
        df_f = df_f[df_f["ID_Agencia"].isin(agencias_sel)]
    return df_f


def calcular_kpis(
    df: pd.DataFrame,
    freq: str = "D",
    total_habs: int = 735
) -> Dict[str, Any]:
    """
    Calcula KPI temporales y globales sobre el DataFrame filtrado.
    Ahora con prints de debug en pasos clave para diagnosticar valores y columnas.
    """
    print("▶️ [DEBUG] Entrando a calcular_kpis. Filas recibidas:", len(df))
    if df.empty:
        print("    [DEBUG] DataFrame vacío. Retornando indicadores cero.")
        empty_series = pd.Series(dtype=float)
        global_vals = {
            "total_reservas": 0,
            "room_nights": 0,
            "ocupacion": 0,
            "adr": 0,
            "revpar": 0,
            "avg_stay": 0,
            "tasa_cancel": 0,
            "tasa_noshow": 0
        }
        return {
            **{k: empty_series for k in ["volumen", "room_nights", "ocupacion", "adr", "revpar"]},
            "global": global_vals
        }

    # 1) Verificar si existen las columnas requeridas
    columnas_necesarias = ["fecha_checkin", "h_num_noc", "h_tot_hab", "h_tfa_total", "descripcion"]
    for col in columnas_necesarias:
        print(f"    [DEBUG] Columna '{col}' presente:" , col in df.columns,
              f"| Nulos en '{col}':", df[col].isna().sum() if col in df.columns else "N/A")

    # Indexar por fecha_checkin para agrupaciones
    df_k = df.copy()
    try:
        df_k.set_index("fecha_checkin", inplace=True)
        print("    [DEBUG] Índice cambiado a 'fecha_checkin'. Primeras fechas:", df_k.index.min(), "->", df_k.index.max())
    except KeyError:
        print("    [ERROR] No se encontró la columna 'fecha_checkin' en el DataFrame.")
        raise

    # 2) Calcular métricas por periodo
    grp = df_k.groupby(pd.Grouper(freq=freq))
    volumen = grp.size()
    print("    [DEBUG] Serie 'volumen' obtenida. Primeros valores:\n", volumen.head())

    room_nights = grp.apply(lambda x: (x["h_num_noc"] * x["h_tot_hab"]).sum())
    print("    [DEBUG] Serie 'room_nights' obtenida. Primeros valores:\n", room_nights.head())

    ingreso_total = grp["h_tfa_total"].sum()
    print("    [DEBUG] Serie 'ingreso_total' obtenida. Primeros valores:\n", ingreso_total.head())

    adr = ingreso_total.div(room_nights.replace({0: pd.NA}))
    print("    [DEBUG] Serie 'adr' (Ingreso ÷ Room Nights). Primeros valores:\n", adr.head())

    # 3) Calcular días en cada periodo
    if freq == "D":
        dias_en_periodo = pd.Series(1, index=volumen.index)
    elif freq == "W":
        dias_en_periodo = pd.Series(7, index=volumen.index)
    elif freq == "M":
        dias_en_periodo = pd.to_datetime(volumen.index).to_series().dt.days_in_month
    else:
        dias_en_periodo = pd.Series(1, index=volumen.index)
    print("    [DEBUG] 'dias_en_periodo' para frecuencia", freq, "Primeros valores:\n", dias_en_periodo.head())

    ocupacion = room_nights.div(total_habs * dias_en_periodo) * 100
    print("    [DEBUG] Serie 'ocupacion' (%). Primeros valores:\n", ocupacion.head())

    revpar = adr.mul(ocupacion.div(100))
    print("    [DEBUG] Serie 'revpar' (ADR × Ocupación). Primeros valores:\n", revpar.head())

    # 4) KPI globales
    total_res = len(df_k)
    total_rn = room_nights.sum()
    span_days = (df_k.index.max() - df_k.index.min()).days + 1
    ocupacion_glob = (total_rn / (total_habs * span_days)) * 100 if span_days > 0 else 0
    adr_glob = ingreso_total.sum() / total_rn if total_rn > 0 else 0
    revpar_glob = adr_glob * (ocupacion_glob / 100)
    avg_stay = df_k["h_num_noc"].mean()

    print(f"    [DEBUG] KPI globales → total_res: {total_res}, total_rn: {total_rn}, span_days: {span_days}")
    print(f"             → ocupacion_glob: {ocupacion_glob:.2f}%, adr_glob: {adr_glob:.2f}, revpar_glob: {revpar_glob:.2f}, avg_stay: {avg_stay:.2f}")

    # 5) Tasas de cancelación y no-show
    # Revisar conteo de filas que tengan 'descripcion' nula o en blanco
    null_descr = df_k["descripcion"].isna().sum()
    print("    [DEBUG] Filas con 'descripcion' nulo antes de conteo:", null_descr)

    cancelados = df_k["descripcion"].str.contains("cancel", case=False, na=False).sum()
    noshows = df_k["descripcion"].str.contains("no show", case=False, na=False).sum()
    print(f"    [DEBUG] Reservas con 'cancel': {cancelados}, con 'no show': {noshows}")

    tasa_cancel = cancelados / total_res * 100 if total_res > 0 else 0
    tasa_noshow = noshows / total_res * 100 if total_res > 0 else 0
    print(f"    [DEBUG] Tasa cancelación: {tasa_cancel:.2f}%, Tasa no-show: {tasa_noshow:.2f}%")

    return {
        "volumen": volumen,
        "room_nights": room_nights,
        "ocupacion": ocupacion,
        "adr": adr,
        "revpar": revpar,
        "global": {
            "total_reservas": total_res,
            "room_nights": total_rn,
            "ocupacion": ocupacion_glob,
            "adr": adr_glob,
            "revpar": revpar_glob,
            "avg_stay": avg_stay,
            "tasa_cancel": tasa_cancel,
            "tasa_noshow": tasa_noshow
        }
    }
