# frontend/utils/data_extractor.py
import os
from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

import requests

# Directorio de datos por defecto (carpeta 'data' en el nivel superior del proyecto)


# Load environment variables from .env file
load_dotenv()

# Access the API_BASE_URL environment variable
api_base_url = os.getenv("API_BASE_URL")
print(f"API_BASE_UR:{api_base_url}")

@lru_cache(maxsize=1)
def load_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Carga todos los DataFrames vía GET /raw_data de la API y
    devuelve un diccionario con DataFrames ya listos para usar.
    """
    url = f"{api_base_url}/raw_data"
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
    Calcula KPI temporales y globales a partir de la ocupación diaria “explodida”.
    """
    if df.empty:
        empty = pd.Series(dtype=float)
        return {
            **{k: empty for k in ["volumen", "room_nights", "ocupacion", "adr", "revpar"]},
            "global": {k: 0 for k in [
                "total_reservas","room_nights","ocupacion","adr","revpar","avg_stay","tasa_cancel","tasa_noshow"
            ]}
        }

    # 1) Verificamos columnas y creamos df_k indexado por fecha_checkin
    columnas_necesarias = ["fecha_checkin", "h_num_noc", "h_tot_hab", "h_tfa_total", "descripcion"]
    for col in columnas_necesarias:
        if col not in df.columns:
            raise KeyError(f"No se encontró la columna '{col}'")
    df_k = df.copy()
    df_k["fecha_checkin"] = pd.to_datetime(df_k["fecha_checkin"], errors="coerce")
    df_k = df_k.dropna(subset=["fecha_checkin"])
    df_k.set_index("fecha_checkin", inplace=True)

    # 2) Filtramos filas inválidas y calculamos ingreso_por_noche
    df_k = df_k.dropna(subset=["h_num_noc", "h_tot_hab", "h_tfa_total"])
    df_k["h_num_noc"] = df_k["h_num_noc"].astype(int)
    df_k = df_k[df_k["h_num_noc"] > 0]
    df_k["ingreso_por_noche"] = df_k["h_tfa_total"] / df_k["h_num_noc"]

    # 3) Creamos stay_dates para cada reserva y explotamos
    df_k["stay_dates"] = df_k.apply(
        lambda row: list(pd.date_range(
            start=row.name,
            periods=row["h_num_noc"],
            freq="D"
        )),
        axis=1
    )
    df_exploded = df_k.explode("stay_dates").rename(columns={"stay_dates": "fecha_ocupacion"})

    # 4) Agrupamos por día: habitaciones e ingreso
    diario = (
        df_exploded
        .groupby("fecha_ocupacion")
        .agg({
            "h_tot_hab": "sum",
            "ingreso_por_noche": "sum"
        })
        .rename(columns={
            "h_tot_hab": "habitaciones_ocupadas",
            "ingreso_por_noche": "ingreso_total"
        })
    )
    diario.index.name = None

    # 5) Calculamos KPIs diarios
    diario["ocupacion_%"] = diario["habitaciones_ocupadas"] / total_habs * 100
    diario["adr"] = diario.apply(
        lambda r: r["ingreso_total"] / r["habitaciones_ocupadas"]
                  if r["habitaciones_ocupadas"] > 0 else 0,
        axis=1
    )
    diario["revpar"] = diario["ingreso_total"] / total_habs

    # 6) Reagrupamos a la frecuencia solicitada (D, W o M)
    def agrupar_a_periodo(diario_df, freq, total_habs):
        agr = diario_df.resample(freq).agg({
            "habitaciones_ocupadas": "sum",
            "ingreso_total": "sum"
        }).rename(columns={
            "habitaciones_ocupadas": "room_nights_periodo",
            "ingreso_total": "ingreso_total_periodo"
        })

        if freq == "D":
            dias = pd.Series(1, index=agr.index)
        elif freq == "W":
            dias = pd.Series(7, index=agr.index)
        elif freq == "M":
            dias = pd.to_datetime(agr.index).to_series().dt.days_in_month
        else:
            dias = pd.Series(1, index=agr.index)

        agr["ocupacion"] = agr["room_nights_periodo"] / (total_habs * dias) * 100
        agr["adr"] = agr.apply(
            lambda r: r["ingreso_total_periodo"] / r["room_nights_periodo"]
                      if r["room_nights_periodo"] > 0 else 0,
            axis=1
        )
        agr["revpar"] = agr["ingreso_total_periodo"] / (total_habs * dias)

        return agr

    kpis_periodo = agrupar_a_periodo(diario, freq, total_habs)

    # 7) Extraemos las series que queremos devolver:
    volumen     = kpis_periodo["room_nights_periodo"]   # antes era grp.size(), ahora es suma diaria/periodo
    room_nights = kpis_periodo["room_nights_periodo"]
    ocupacion   = kpis_periodo["ocupacion"]
    adr         = kpis_periodo["adr"]
    revpar      = kpis_periodo["revpar"]

    # 8) KPI globales (igual que antes, pero calculados sobre df_k “raw”, no resampleado)
    total_res     = len(df_k)
    total_roomn   = df_k["h_num_noc"].mul(df_k["h_tot_hab"]).sum()
    span_days     = (df_k.index.max() - df_k.index.min()).days + 1
    ocup_glob     = (total_roomn / (total_habs * span_days)) * 100 if span_days > 0 else 0
    ingreso_glob  = df_k["h_tfa_total"].sum()
    adr_glob      = ingreso_glob / total_roomn if total_roomn > 0 else 0
    revpar_glob   = adr_glob * (ocup_glob / 100)
    avg_stay      = df_k["h_num_noc"].mean()

    # 9) Tasas de cancelación y no-show (sobre df_k original indexado)
    cancelados = df_k["descripcion"].str.contains("cancel", case=False, na=False).sum()
    noshows    = df_k["descripcion"].str.contains("no show", case=False, na=False).sum()
    tasa_cancel = cancelados / total_res * 100 if total_res > 0 else 0
    tasa_noshow = noshows / total_res * 100 if total_res > 0 else 0

    global_vals = {
        "total_reservas": total_res,
        "room_nights": total_roomn,
        "ocupacion": ocup_glob,
        "adr": adr_glob,
        "revpar": revpar_glob,
        "avg_stay": avg_stay,
        "tasa_cancel": tasa_cancel,
        "tasa_noshow": tasa_noshow
    }

    return {
        "volumen":     volumen,
        "room_nights": room_nights,
        "ocupacion":   ocupacion,
        "adr":         adr,
        "revpar":      revpar,
        "global":      global_vals
    }
