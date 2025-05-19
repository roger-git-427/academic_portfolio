"""
This is a boilerplate pipeline 'reservations_preprocessing'
generated using Kedro 0.19.12
"""

import pandas as pd

# 1.  INITIAL SELECTION 

KEEP_COLS = [
    "ID_Reserva", "Fecha_hoy", "h_res_fec", "h_num_per", "h_num_adu",
    "h_num_men", "h_num_noc", "h_tot_hab", "ID_Programa", "ID_Paquete",
    "ID_Segmento_Comp", "ID_Agencia", "ID_empresa", "ID_Tipo_Habitacion",
    "ID_canal", "ID_Pais_Origen", "ID_estatus_reservaciones",
    "h_fec_lld", "h_fec_reg", "h_fec_sda",
]

def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only required columns and rename if desired."""
    df = df[KEEP_COLS]
    return df

# 2.  JOIN LOOK-UP TABLES

def merge_lookup_tables(
    reservations: pd.DataFrame,
    canales: pd.DataFrame,
    empresas: pd.DataFrame,
    agencias: pd.DataFrame,
    estatus: pd.DataFrame
) -> pd.DataFrame:
    df = (
        reservations
        .merge(canales[["ID_canal", "Canal_nombre"]],           on="ID_canal",              how="left")
        .merge(empresas[["ID_empresa", "Empresa_nombre", "Habitaciones_tot"]],
                                                               on="ID_empresa",             how="left")
        .merge(agencias[["ID_Agencia", "Agencia_nombre", "Ciudad_Nombre"]],
                                                               on="ID_Agencia",             how="left")
        .merge(estatus[["ID_estatus_reservaciones", "estatus_reservaciones"]],
                                                               on="ID_estatus_reservaciones", how="left")
    )
    return df

# 3.  DATE CONVERSION 

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    fec_cols = [c for c in df.columns if c.endswith("_fec")] + [
        "Fecha_hoy", "h_fec_lld", "h_fec_reg", "h_fec_sda"
    ]
    for col in fec_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    return df


# 4.  TYPE ENFORCEMENT / DUPLICATE DROP / NEGATIVE FILTER

CATEGORICAL = [
    "ID_Reserva", "ID_Programa", "ID_Paquete",
    "ID_Segmento_Comp", "ID_Agencia", "ID_empresa", "ID_Tipo_Habitacion",
    "ID_canal", "ID_Pais_Origen", "ID_estatus_reservaciones",
    "Canal_nombre", "Empresa_nombre", "Agencia_nombre",
    "Ciudad_Nombre", "estatus_reservaciones"
]

def enforce_types_and_basic_filters(df: pd.DataFrame) -> pd.DataFrame:
    # duplicates
    df = df.drop_duplicates(subset=["ID_Reserva"])
    # categorical
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")
    # negatives impossible
    num_check = [
        "h_num_per", "h_num_adu", "h_num_men",
        "h_num_noc", "h_tot_hab"
    ]
    for col in num_check:
        df = df[df[col] >= 0]
    return df.reset_index(drop=True)

# 5.   CITY NORMALISATION

CORRECCIONES_CIUDADES = {
    "CANCUN": "CANCÚN", "IXTAP": "IXTAPA", "CD.DEMEXICO": "CD.MEXICO",
    "CDDEMEXICO": "CD.MEXICO", "CDMEXICO": "CD.MEXICO",
    "CIUDADDEMEXIC": "CD.MEXICO", "CIUDADMEXICO": "CD.MEXICO",
    "ESTADODEMEXIC": "CD.MEXICO", "MEXICO": "CD.MEXICO",
    "MEXICOCITY": "CD.MEXICO", "CD.MEXICO": "CD.MEXICO",
    "DEL.CUAUHTEMOC": "CUAUHTÉMOC", "CDCUAUHTEMOC": "CUAUHTÉMOC",
    "DEL.COYOACAN": "COYOACÁN", "COYOACAN": "COYOACÁN",
    "GUADALAJARA,JA": "GUADALAJARA", "ESTADODEMEXIC": "ESTADO DE MÉXICO",
}

def normalise_city(df: pd.DataFrame) -> pd.DataFrame:
    ser = (
        df["Ciudad_Nombre"]
        .astype(str)
        .str.upper()
        .str.replace(r"[\s,]+", "", regex=True)
        .replace(CORRECCIONES_CIUDADES)
    )
    df["Ciudad_Normalizada"] = ser
    return df

# 6.  OUTLIER REMOVAL

def remove_outliers_percentile(
    df: pd.DataFrame,
    exclude: list[str] | None = None,
    pct: float = 0.01
) -> pd.DataFrame:
    exclude = exclude or []
    lower_q = pct / 2
    upper_q = 1 - lower_q
    mask = pd.Series(True, index=df.index)
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        if col in exclude:
            continue
        lo, hi = df[col].quantile([lower_q, upper_q])
        mask &= df[col].between(lo, hi)
    return df[mask].reset_index(drop=True)

# 7.  EXPLODE BY DATE  &  AGGREGATE ROOMS

def explode_and_sum_rooms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["h_fec_lld", "h_fec_sda"])
    df["fecha"] = df.apply(
        lambda r: pd.date_range(r["h_fec_lld"], r["h_fec_sda"] - pd.Timedelta(days=1)),
        axis=1,
    )
    exploded = df.explode("fecha")
    out = (
        exploded[["fecha", "h_tot_hab"]]
        .groupby("fecha", as_index=False)
        .sum()
        .rename(columns={"h_tot_hab": "rooms_reserved"})
        .sort_values("fecha")
    )
    return out