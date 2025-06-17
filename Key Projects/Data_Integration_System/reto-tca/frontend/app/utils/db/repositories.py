import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import Dict, Optional
from pathlib import Path

from .connection import get_db_session_sync
from .models import Reservaciones, IarCanales, IarEmpresas, IarAgencias, IarEstatusReservaciones

def load_data_from_db() -> Dict[str, pd.DataFrame]:
    """
    Load all data directly from the database and return a dictionary of DataFrames
    """
    session = get_db_session_sync()
    
    try:
        
        print("[DEBUG] Loading data from database...")
        # 1) Query reservaciones table
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
        result_reserv = session.execute(sql_reserv)
        df_res = pd.DataFrame(result_reserv.fetchall(), columns=result_reserv.keys())
        
        # 2) Query lookup tables
        sql_canales = text('SELECT "ID_canal", "Canal_nombre" FROM "iar_canales";')
        result_canales = session.execute(sql_canales)
        df_can = pd.DataFrame(result_canales.fetchall(), columns=result_canales.keys())
        
        sql_empresas = text('SELECT "ID_empresa", "Empresa_nombre", "Habitaciones_tot" FROM "iar_empresas";')
        result_empresas = session.execute(sql_empresas)
        df_emp = pd.DataFrame(result_empresas.fetchall(), columns=result_empresas.keys())
        
        sql_agencias = text('SELECT "ID_Agencia", "Agencia_nombre", "Ciudad_Nombre" FROM "iar_agencias";')
        result_agencias = session.execute(sql_agencias)
        df_ages = pd.DataFrame(result_agencias.fetchall(), columns=result_agencias.keys())
        
        sql_estatus = text('SELECT "ID_estatus_reservaciones", "estatus_reservaciones" FROM "iar_estatus_reservaciones";')
        result_estatus = session.execute(sql_estatus)
        df_est = pd.DataFrame(result_estatus.fetchall(), columns=result_estatus.keys())
        
        # 3) Convert date columns to datetime
        cols_fec = [col for col in df_res.columns if "fec" in col.lower()]
        for col in cols_fec:
            df_res[col] = pd.to_datetime(df_res[col], errors="coerce")
        
        # 4) Add fecha_checkin column
        if "h_fec_lld" in df_res.columns:
            df_res["fecha_checkin"] = df_res["h_fec_lld"]
        else:
            df_res["fecha_checkin"] = pd.NaT
        
        # 5) Drop rows with invalid fecha_checkin
        df_res = df_res.dropna(subset=["fecha_checkin"]).reset_index(drop=True)
        
        # 6) Convert ID columns to Int16
        for col in ["ID_empresa", "ID_canal", "ID_Agencia", "ID_estatus_reservaciones"]:
            if col in df_res.columns:
                df_res[col] = pd.to_numeric(df_res[col], errors="coerce").astype("Int16")
        
        # 7) Normalize IDs in lookup tables
        if "ID_empresa" in df_emp.columns:
            df_emp["ID_empresa"] = pd.to_numeric(df_emp["ID_empresa"], errors="coerce").astype("Int16")
        if "ID_canal" in df_can.columns:
            df_can["ID_canal"] = pd.to_numeric(df_can["ID_canal"], errors="coerce").astype("Int16")
        if "ID_Agencia" in df_ages.columns:
            df_ages["ID_Agencia"] = pd.to_numeric(df_ages["ID_Agencia"], errors="coerce").astype("Int16")
            
        # 8) Merge description from estatus
        if not df_est.empty and "ID_estatus_reservaciones" in df_res.columns:
            df_res = df_res.merge(
                df_est[["ID_estatus_reservaciones", "estatus_reservaciones"]],
                on="ID_estatus_reservaciones",
                how="left"
            )
            df_res = df_res.rename(columns={"estatus_reservaciones": "descripcion"})
        
        print("[DEBUG] Data loaded successfully.")
        return {
            "reservaciones": df_res,
            "estatus": df_est,
            "agencias": df_ages,
            "empresas": df_emp,
            "canales": df_can
        }
    
    finally:
        session.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()