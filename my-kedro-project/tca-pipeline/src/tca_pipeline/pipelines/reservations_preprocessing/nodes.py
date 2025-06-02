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
    print(df.shape)
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
    print(df.shape)
    return df

# 3.  DATE CONVERSION 

def convert_dates(df: pd.DataFrame) -> pd.DataFrame:
    fec_cols = [col for col in df.columns if '_fec' in col]
    for col in fec_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format="%Y%m%d", errors="coerce")
    df['Fecha_hoy'] = pd.to_datetime(df['Fecha_hoy'], errors='coerce')
    print(df.shape)
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
    print(df.shape)
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
    print(df.shape)
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

# 7. CORRECT NUMBER OF PERSONS

def replace_h_num_persons(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {'h_num_per', 'h_num_adu', 'h_num_men'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener las columnas: {required_columns}")

    condition = df['h_num_per'] < (df['h_num_adu'] + df['h_num_men'])
    df.loc[condition, 'h_num_per'] = df['h_num_adu'] + df['h_num_men']

    print("Se actualizaron los valores de 'h_num_per' donde eran menores que 'h_num_adu + h_num_men'")
    print(df.shape)
    return df


# 8.  FILTER RESERVATIONS

def filtered_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure date columns are datetime
    df['h_fec_lld'] = pd.to_datetime(df['h_fec_lld'], errors='coerce')
    df['h_fec_sda'] = pd.to_datetime(df['h_fec_sda'], errors='coerce')

    # Filter out rows where total rooms are zero
    df = df[df['h_tot_hab'] != 0]
    print(f"Registros con h_tot_hab != 0: {len(df)}")

    # Drop reservations marked as cancelled with missing check-in or check-out
    mask_cancelled_missing_dates = (
        (df['estatus_reservaciones'] == 'RESERVACION CANCELADA') &
        (df['h_fec_lld'].isna() | df['h_fec_sda'].isna())
    )
    df = df[~mask_cancelled_missing_dates]
    print(df.shape)
    return df


# 9. BUILD DAILY OCCUPANCY

def build_daily_occupancy(
    df: pd.DataFrame, START_DATE: str | None = None, END_DATE: str | None = None) -> pd.DataFrame:
    print(f"[DEBUG] df input shape: {df.shape}")

    # 1) Ensure dates are datetime
    df['h_fec_lld'] = pd.to_datetime(df['h_fec_lld'], errors='coerce')
    df['h_fec_sda'] = pd.to_datetime(df['h_fec_sda'], errors='coerce')

    # 2) Drop invalid or missing
    df = df.dropna(subset=['h_fec_lld', 'h_fec_sda'])
    print(f"[DEBUG] df after dropna shape: {df.shape}")

    df = df[df['h_fec_sda'] > df['h_fec_lld']]
    print(f"[DEBUG] df df['h_fec_sda'] > df['h_fec_lld'] shape: {df.shape}")

    # 3) Build stay_dates as LISTS, not Index objects
    df['stay_dates'] = df.apply(
        lambda row: list(
            pd.date_range(
                start=row['h_fec_lld'],
                end=row['h_fec_sda'] - pd.Timedelta(days=1)
            )
        ),
        axis=1
    )
    print(f"[DEBUG] df stay dates shape: {df.shape}")

    # 4) Explode into one date per row
    df_exploded = df.explode('stay_dates')
    print(f"[DEBUG] df explode shape: {df_exploded.shape}")

    # 5) Aggregate
    daily_occupancy = (
        df_exploded
        .groupby('stay_dates')[['h_tot_hab','h_num_per','h_num_adu','h_num_men']]
        .sum()
        .reset_index()
        .rename(columns={'stay_dates':'Fecha'})
        .sort_values('Fecha')
    )
    print(f"[DEBUG] df daily_occupancy shape: {daily_occupancy.shape}")

    # 6) Fill gaps
    full_dates = pd.DataFrame({
        'Fecha': pd.date_range(
            start=daily_occupancy['Fecha'].min(),
            end=daily_occupancy['Fecha'].max()
        )
    })
    print(f"[DEBUG] df full_dates shape: {full_dates.shape}")

    daily_occupancy = (
        full_dates
        .merge(daily_occupancy, on='Fecha', how='left')
    )
    print(f"[DEBUG] df daily_occupancy shape: {daily_occupancy.shape}")

    occupancy_cols = ['h_tot_hab', 'h_num_per', 'h_num_adu', 'h_num_men']
    daily_occupancy[occupancy_cols] = (
        daily_occupancy[occupancy_cols].fillna(0).astype(int)
    )

    print(f"[DEBUG] df daily_occupancy filter shape: {daily_occupancy.shape}")

    # To-DO: revisar lo siguiente para funcionalidad completa con la inferencia
    if START_DATE is not None and END_DATE is not None:
        daily_occupancy = daily_occupancy[
            (daily_occupancy['Fecha'] >= pd.to_datetime(START_DATE)) &
            (daily_occupancy['Fecha'] <= pd.to_datetime(END_DATE))
        ]
        print(f"[DEBUG] After date filtering: shape = {daily_occupancy.shape}")



    return daily_occupancy
