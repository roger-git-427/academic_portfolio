from pathlib import Path
from functools import lru_cache
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict, Any

# Directorio de datos por defecto (carpeta 'data' en el nivel superior del proyecto)
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

@lru_cache(maxsize=1)
def load_data(data_dir: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    """
    Carga los datos de reservaciones y catálogos desde archivos CSV.
    Usa caching para evitar recargas innecesarias.

    Args:
        data_dir: Ruta opcional al directorio de datos; si es None, usa DATA_DIR.

    Returns:
        Diccionario con DataFrames: reservaciones, agencias, empresas, canales, estatus.
    """
    base = Path(data_dir) if data_dir else DATA_DIR

    # Definir rutas de archivos
    paths = {
        'reservaciones': base / 'reservaciones_tcabdfront2.csv',
        'estatus': base / 'iar_estatus_reservaciones.csv',
        'agencias': base / 'iar_Agencias.csv',
        'empresas': base / 'iar_empresas.csv',
        'canales': base / 'iar_canales.csv'
    }

    # Leer CSVs
    df_res = pd.read_csv(
        paths['reservaciones'],
        low_memory=False,
        parse_dates=['h_res_fec_ok', 'h_fec_lld_ok']
    )
    df_est = pd.read_csv(paths['estatus'])
    df_est.rename(columns={'estatus_reservaciones': 'descripcion'}, inplace=True)
    df_ages = pd.read_csv(paths['agencias'])
    df_emp = pd.read_csv(paths['empresas'])
    df_can = pd.read_csv(paths['canales'])
    
    df_res['ID_empresa']  = df_res['ID_empresa'].astype(int)
    df_res['ID_canal']    = df_res['ID_canal'].astype(int)
    df_res['ID_Agencia']  = df_res['ID_Agencia'].astype(int)

    df_emp['ID_empresa']  = df_emp['ID_empresa'].astype(int)
    df_can['ID_canal']    = df_can['ID_canal'].astype(int)
    df_ages['ID_Agencia'] = df_ages['ID_Agencia'].astype(int)

    # Merge estatus en reservaciones para simplificar posterior filtrado
    if 'ID_estatus_reservaciones' in df_res.columns:
        df_res = df_res.merge(
            df_est[['ID_estatus_reservaciones', 'descripcion']],
            on='ID_estatus_reservaciones', how='left'
        )
    # Renombrar fechas de check-in para mayor claridad
    df_res['fecha_checkin'] = pd.to_datetime(df_res['h_fec_lld_ok'])

    return {
        'reservaciones': df_res,
        'estatus': df_est,
        'agencias': df_ages,
        'empresas': df_emp,
        'canales': df_can
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
        df: DataFrame de reservaciones (con columna fecha_checkin).
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
        df_f = df_f[df_f['fecha_checkin'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df_f = df_f[df_f['fecha_checkin'] <= pd.to_datetime(end_date)]
    if empresas_sel:
        df_f = df_f[df_f['ID_empresa'].isin(empresas_sel)]
    if canales_sel:
        df_f = df_f[df_f['ID_canal'].isin(canales_sel)]
    if agencias_sel:
        df_f = df_f[df_f['ID_Agencia'].isin(agencias_sel)]
    return df_f


def calcular_kpis(
    df: pd.DataFrame,
    freq: str = 'D',
    total_habs: int = 735
) -> Dict[str, Any]:
    """
    Calcula KPI temporales y globales sobre el DataFrame filtrado.

    Args:
        df: DataFrame filtrado de reservaciones.
        freq: Frecuencia de agregación ('D', 'W', 'M').
        total_habs: Número total de habitaciones disponibles.

    Returns:
        Diccionario con series de métricas (volumen, room_nights, ocupacion, adr, revpar)
        y valores globales (total_reservas, tasa_cancel, tasa_noshow, etc.).
    """
    if df.empty:
        empty_series = pd.Series(dtype=float)
        global_vals = { 'total_reservas': 0, 'room_nights': 0, 'ocupacion': 0,
                        'adr': 0, 'revpar': 0, 'avg_stay': 0,
                        'tasa_cancel': 0, 'tasa_noshow': 0 }
        return { **{k: empty_series for k in ['volumen','room_nights','ocupacion','adr','revpar']}, 'global': global_vals }

    # Indexar por fecha_checkin para agrupaciones
    df_k = df.copy()
    df_k.set_index('fecha_checkin', inplace=True)

    # Calcular métricas por periodo
    grp = df_k.groupby(pd.Grouper(freq=freq))
    volumen = grp.size()
    room_nights = grp.apply(lambda x: (x['h_num_noc'] * x['h_tot_hab']).sum())
    ingreso_total = grp['h_tfa_total'].sum()
    adr = ingreso_total.div(room_nights.replace({0: pd.NA}))

    # Calcular días en cada periodo
    if freq == 'D':
        dias_en_periodo = pd.Series(1, index=volumen.index)
    elif freq == 'W':
        dias_en_periodo = pd.Series(7, index=volumen.index)
    elif freq == 'M':
        # Para mensual, obtener días del mes de cada timestamp
        dias_en_periodo = pd.to_datetime(volumen.index).to_series().dt.days_in_month
    else:
        dias_en_periodo = pd.Series(1, index=volumen.index)

    ocupacion = room_nights.div(total_habs * dias_en_periodo) * 100
    revpar = adr.mul(ocupacion.div(100))

    # KPI globales
    total_res = len(df_k)
    total_rn = room_nights.sum()
    span_days = (df_k.index.max() - df_k.index.min()).days + 1
    ocupacion_glob = (total_rn / (total_habs * span_days)) * 100
    adr_glob = ingreso_total.sum() / total_rn if total_rn > 0 else 0
    revpar_glob = adr_glob * (ocupacion_glob / 100)
    avg_stay = df_k['h_num_noc'].mean()

    # Tasas de cancelación y no-show
    cancelados = df_k['descripcion'].str.contains('cancel', case=False, na=False).sum()
    noshows = df_k['descripcion'].str.contains('no show', case=False, na=False).sum()
    tasa_cancel = cancelados / total_res * 100
    tasa_noshow = noshows / total_res * 100

    return {
        'volumen': volumen,
        'room_nights': room_nights,
        'ocupacion': ocupacion,
        'adr': adr,
        'revpar': revpar,
        'global': {
            'total_reservas': total_res,
            'room_nights': total_rn,
            'ocupacion': ocupacion_glob,
            'adr': adr_glob,
            'revpar': revpar_glob,
            'avg_stay': avg_stay,
            'tasa_cancel': tasa_cancel,
            'tasa_noshow': tasa_noshow
        }
    }
