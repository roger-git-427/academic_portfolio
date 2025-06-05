# backend/src/services/raw_data.py

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.schema import RawDataResponse

async def get_all_raw_data(db: AsyncSession) -> RawDataResponse:
    """
    Consulta todas las tablas del modelo y retorna un objeto RawDataResponse,
    donde cada atributo es la lista de registros de esa tabla en forma de dict.
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
        FROM "reservaciones";
    """)
    result_reserv = await db.execute(sql_reserv)
    rows_reserv = result_reserv.fetchall()
    cols_reserv = result_reserv.keys()
    df_reserv = pd.DataFrame(rows_reserv, columns=cols_reserv)

    # --------------------------------------------------------
    # 2) Consultar la tabla `iar_canales`
    # --------------------------------------------------------
    sql_canales = text('SELECT "ID_canal", "Canal_nombre" FROM "iar_canales";')
    result_canales = await db.execute(sql_canales)
    rows_canales = result_canales.fetchall()
    cols_canales = result_canales.keys()
    df_canales = pd.DataFrame(rows_canales, columns=cols_canales)

    # --------------------------------------------------------
    # 3) Consultar la tabla `iar_empresas`
    # --------------------------------------------------------
    sql_empresas = text('SELECT "ID_empresa", "Empresa_nombre", "Habitaciones_tot" FROM "iar_empresas";')
    result_empresas = await db.execute(sql_empresas)
    rows_empresas = result_empresas.fetchall()
    cols_empresas = result_empresas.keys()
    df_empresas = pd.DataFrame(rows_empresas, columns=cols_empresas)

    # --------------------------------------------------------
    # 4) Consultar la tabla `iar_agencias`
    # --------------------------------------------------------
    sql_agencias = text('SELECT "ID_Agencia", "Agencia_nombre", "Ciudad_Nombre" FROM "iar_agencias";')
    result_agencias = await db.execute(sql_agencias)
    rows_agencias = result_agencias.fetchall()
    cols_agencias = result_agencias.keys()
    df_agencias = pd.DataFrame(rows_agencias, columns=cols_agencias)

    # --------------------------------------------------------
    # 5) Consultar la tabla `iar_estatus_reservaciones`
    # --------------------------------------------------------
    sql_estatus = text('SELECT "ID_estatus_reservaciones", "estatus_reservaciones" FROM "iar_estatus_reservaciones";')
    result_estatus = await db.execute(sql_estatus)
    rows_estatus = result_estatus.fetchall()
    cols_estatus = result_estatus.keys()
    df_estatus = pd.DataFrame(rows_estatus, columns=cols_estatus)

    # --------------------------------------------------------
    # 6) Convertir cada DataFrame a una lista de diccionarios serializable
    # --------------------------------------------------------
    def df_to_serializable(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Reemplaza inf/NaN por None y convierte tipos numpy a nativos,
        retornando una lista de registros.
        """
        # 1) Reemplazar infinitos por None
        df2 = df.replace([np.inf, -np.inf], None)
        # 2) Reemplazar NaN por None
        df2 = df2.where(pd.notnull(df2), None)

        # 3) Convertir a lista de diccionarios
        raw_records = df2.to_dict(orient="records")

        serializable = []
        for rec in raw_records:
            new_rec: Dict[str, Any] = {}
            for k, v in rec.items():
                if v is None:
                    new_rec[k] = None
                else:
                    # Intentar convertir a float para detectar inf/NaN
                    try:
                        fval = float(v)
                        if not np.isfinite(fval):
                            new_rec[k] = None
                        else:
                            # Si es numpy integer
                            if isinstance(v, (int, np.integer)):
                                new_rec[k] = int(v)
                            else:
                                new_rec[k] = fval
                    except:
                        # No es numérico (p.ej. string), se deja tal cual
                        new_rec[k] = v
            serializable.append(new_rec)
        return serializable

    return RawDataResponse(
        reservaciones=df_to_serializable(df_reserv),
        iar_canales=df_to_serializable(df_canales),
        iar_empresas=df_to_serializable(df_empresas),
        iar_agencias=df_to_serializable(df_agencias),
        iar_estatus_reservaciones=df_to_serializable(df_estatus),
    )
