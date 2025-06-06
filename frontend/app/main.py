
import os
from dotenv import load_dotenv
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
import utils.data_extractor_v2 as de  # Use the updated data extractor
import components.filters as filters
import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
import numpy as np
import traceback

# ---------------------------------------------------
# Load environment variables before attempting data load
# ---------------------------------------------------
load_dotenv()

# Access environment variables
api_base_url = os.getenv("API_BASE_URL")
use_direct_db = os.getenv("USE_DIRECT_DB", "false").lower() == "true"

# ---------------------------------------------------
# Initialize data loading with error handling
# ---------------------------------------------------
try:
    print(f"[INFO] Loading data using {'direct database' if use_direct_db else 'API'}")
    data = de.load_data()
    df_reservas = data['reservaciones']
    df_empresas = data['empresas']
    df_canales = data['canales']
    df_agencias = data['agencias']
    print(f"[INFO] Data loaded successfully: {len(df_reservas)} reservations")
except Exception as e:
    print(f"[ERROR] Failed to load data: {str(e)}")
    print(traceback.format_exc())
    # Create empty DataFrames as fallback
    df_reservas = pd.DataFrame()
    df_empresas = pd.DataFrame()
    df_canales = pd.DataFrame() 
    df_agencias = pd.DataFrame()

# ---------------------------------------------------
# Iniciar aplicación Dash
# ---------------------------------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Dashboard KPI Hoteleros'

# ---------------------------------------------------
# Definir header (logo + filtros)
# ---------------------------------------------------
header = html.Div(
    style={
        'position': 'fixed', 'top': '0', 'left': '0', 'right': '0', 'zIndex': '999',
        'backgroundColor': '#262626', 'boxSizing': 'border-box', 'padding': '10px 40px 5px 40px'
    },
    children=[
        # primera fila: logo + título
        html.Div(
            style={'position': 'relative', 'width': '100%', 'height': '40px', 'marginTop': '10px'},
            children=[
                html.Img(
                    src=app.get_asset_url('tca_logo.png'),
                    style={'position': 'absolute', 'left': '0px', 'height': '50px'},
                    alt='Logo TCA'
                ),
                html.H4(
                    'Dashboard de Reservas Hoteleras',
                    style={
                        'position': 'absolute', 'left': '50%', 'transform': 'translateX(-50%)',
                        'color': 'white', 'margin': '0', 'fontSize': '32px'
                    }
                )
            ]
        ),
        # segunda fila: filtros
        html.Div(
            style={'marginTop': '30px', 'paddingBottom': '0px', 'width': '100%', 'backgroundColor': '#262626'},
            children=[
                dbc.Container(
                    fluid=True,
                    style={'padding': '0'},
                    children=[
                        filters.crear_controles(df_reservas, df_empresas, df_canales, df_agencias)
                    ]
                )
            ]
        )
    ]
)

# ---------------------------------------------------
# Layout principal
# ---------------------------------------------------
app.layout = html.Div(
    style={'margin': '0', 'padding': '0'},
    children=[
        header,  # header fijo

        # container principal (dejamos espacio arriba para el header)
        dbc.Container(
            fluid=True,
            style={'paddingTop': '200px'},
            children=[
                html.Hr(style={'borderColor': '#444444', 'margin': '0'}),

                # KPI Cards
                dbc.Row([
                    dbc.Col(dcc.Graph(id='indicador-adr',       config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-revpar',    config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-ocupacion', config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-estancia',  config={'displayModeBar': False}), xs=12, sm=6, md=3),
                ], className='mb-4 mt-2'),

                # Serie temporal de Volumen
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-volumen', config={'responsive': True}), xs=12, md=12)
                ], className='mb-4'),

                # Ocupación y RevPAR
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-ocupacion', config={'responsive': True}), xs=12, md=6),
                    dbc.Col(dcc.Graph(id='grafico-revpar',     config={'responsive': True}), xs=12, md=6),
                ], className='mb-4'),

                # Distribuciones: Leadtime, Estancia, Cancel/No-Show
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-leadtime', config={'responsive': True}), xs=12, md=4),
                    dbc.Col(dcc.Graph(id='grafico-estancia', config={'responsive': True}), xs=12, md=4),
                    dbc.Col(dcc.Graph(id='grafico-cancel',    config={'responsive': True}), xs=12, md=4),
                ], className='mb-4'),
            ]
        )
    ]
)

# ---------------------------------------------------
# CALLBACK PRINCIPAL: se dispara cuando cambian los filtros (incluido el slider de fechas).
# ---------------------------------------------------
@app.callback(
    # Salidas: figuras de cada gráfico + texto para el label de fechas
    Output('grafico-volumen',     'figure'),
    Output('grafico-ocupacion',   'figure'),
    Output('grafico-revpar',      'figure'),
    Output('grafico-leadtime',    'figure'),
    Output('grafico-estancia',    'figure'),
    Output('grafico-cancel',      'figure'),
    Output('indicador-adr',       'figure'),
    Output('indicador-revpar',    'figure'),
    Output('indicador-ocupacion', 'figure'),
    Output('indicador-estancia',  'figure'),
    Output('label-fechas',        'children'),
    # Entradas: valores de los filtros
    Input('filter-fechas-slider', 'value'),
    Input('filter-empresa',       'value'),
    Input('filter-canal',         'value'),
    Input('filter-agencia',       'value'),
    Input('filter-freq',          'value'),
)
def actualizar_dashboard(fecha_offset, empresas, canales, agencias, freq):
    """
    Cada vez que el usuario mueva el slider o cambie cualquiera de los filtros,
    esta función recalcula:
      1) sd, ed a partir de fecha_offset (slider).
      2) Filtra df_reservas y calcula KPIs.
      3) Recorta cada serie histórica al rango [sd, ed].
      4) Construye los gráficos históricos.
      5) Reconstruye la ventana de datos para entrenar hasta la fecha ed, llama a /predict.
      6) Agrupa la predicción según freq y superpone sobre los gráficos.
      7) Devuelve todos los gráficos y el texto del label-fechas.
    """

    # --------------------------------------
    # 1) Reconstruir fechas reales (sd, ed)
    # --------------------------------------
    fecha_base = dt.date(2019, 1, 1)
    offset_inicio, offset_fin = fecha_offset
    sd = fecha_base + dt.timedelta(days=offset_inicio)
    ed = fecha_base + dt.timedelta(days=offset_fin)

    # Etiqueta que veremos arriba de los gráficos
    texto_rango_label = f"Rango de fechas: Del {sd:%Y-%m-%d} al {ed:%Y-%m-%d}"

    # --------------------------------------
    # 2) Filtrar datos y calcular KPIs
    # --------------------------------------
    df_fil = de.filtrar_datos(
        df_reservas,
        sd, ed,
        empresas or None,
        canales  or None,
        agencias or None
    )
    print(f"[DEBUG callback] Rango: {sd} a {ed} | Filtrado result: {len(df_fil)} filas")
    kpis = de.calcular_kpis(df_fil, freq, total_habs=735)

    # --------------------------------------
    # 3) Recortar las series históricas
    # --------------------------------------
    sd_ts = pd.to_datetime(sd)
    ed_ts = pd.to_datetime(ed)

    series_volumen   = kpis['volumen']  [(kpis['volumen']  .index >= sd_ts) & (kpis['volumen']  .index <= ed_ts)]
    series_ocupacion = kpis['ocupacion'][(kpis['ocupacion'].index >= sd_ts) & (kpis['ocupacion'].index <= ed_ts)]
    series_revpar    = kpis['revpar']   [(kpis['revpar']   .index >= sd_ts) & (kpis['revpar']   .index <= ed_ts)]

    # --------------------------------------
    # 4) Construir las figuras históricas
    # --------------------------------------
    # — Volumen histórico
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=series_volumen.index, y=series_volumen.values,
        mode='lines',
        name='Volumen Histórico',
        line=dict(color='lightgrey')
    ))

    # — Leadtime (histograma)
    lead = ((df_fil['h_fec_lld'] - df_fil['h_res_fec']).dt.days if not df_fil.empty else [])
    fig_lead = filters.grafica_histograma(
        lead,
        titulo='Anticipación de Reserva',
        xaxis_title='Días de anticipación'
    )

    # — Duración de estancia (boxplot)
    if not df_fil.empty:
        df_aux = df_fil.dropna(subset=["h_num_noc", "h_tot_hab", "h_tfa_total"]).copy()
        df_aux["h_num_noc"] = df_aux["h_num_noc"].astype(int)
        df_aux = df_aux[df_aux["h_num_noc"] > 0]
        series_estancia = df_aux["h_num_noc"].astype(float)
    else:
        series_estancia = []

    print(f"[DEBUG Avg Stay] KPI global avg_stay = {kpis['global']['avg_stay']:.2f}")
    if hasattr(series_estancia, "mean"):
        print(f"[DEBUG Avg Stay] Media en series_estancia = {series_estancia.mean():.2f}")
    else:
        print("[DEBUG Avg Stay] series_estancia vacío → media = 0.00")

    fig_stay = filters.grafica_boxplot(
        series_estancia,
        titulo='Duración de Estancia',
        yaxis_title='Noches por reserva'
    )

    # — Tasa de Cancelación y No-Show
    tasas = {
        'Cancelación': kpis['global']['tasa_cancel'],
        'No Show':    kpis['global']['tasa_noshow']
    }
    fig_cancel = px.bar(
        x=list(tasas.keys()),
        y=list(tasas.values()),
        title='Tasa de Cancelación y No-Show',
        color_discrete_sequence=[filters.PRIMARY_COLOR]
    )
    fig_cancel.update_layout(
        yaxis_range=[0, 30],
        template='plotly_white',
        yaxis_title='Tasa (%)',
        plot_bgcolor=filters.BACKGROUND_COLOR,
        paper_bgcolor=filters.BACKGROUND_COLOR,
        font_color=filters.TEXT_COLOR,
        margin=dict(l=20, r=20, t=30, b=20),
        height=350
    )

    # — KPI Cards (ADR, RevPAR, Ocupación, Estancia Prom.)
    fig_ind_adr      = filters.grafica_indicador(
        kpis['global']['adr'],       titulo='ADR',       sufijo=' MXN'
    )
    fig_ind_revpar   = filters.grafica_indicador(
        kpis['global']['revpar'],    titulo='RevPAR',    sufijo=' MXN'
    )
    fig_ind_ocup     = filters.grafica_indicador(
        kpis['global']['ocupacion'], titulo='Ocupación', sufijo='%'
    )
    fig_ind_estancia = filters.grafica_indicador(
        kpis['global']['avg_stay'],  titulo='Estancia Prom.', sufijo=' noches'
    )

    # --------------------------------------
    # 5) Reconstruir la ventana para entrenar hasta ed
    # --------------------------------------
    df_raw = df_reservas.copy()
    df_raw['fecha_checkin'] = pd.to_datetime(df_raw['fecha_checkin'], errors='coerce')

    # Tomamos como punto final de entrenamiento la fecha “ed” que viene del slider:
    window_end = pd.to_datetime(ed)

    # Definimos ventana de entrenamiento: últimos 180 días hasta ed
    window_start = window_end - pd.Timedelta(days=180)
    train_end = window_end  # ahora entrenamos hasta la fecha seleccionada

    print(f"[DEBUG] Ventana de predicción (entrenamiento): {window_start:%Y-%m-%d} a {train_end:%Y-%m-%d}")

    # Filtramos reservas con check-in en [window_start, train_end]
    df_pred_window = df_raw[
        (df_raw['fecha_checkin'] >= window_start) &
        (df_raw['fecha_checkin'] <= train_end)
    ].copy()

    # =====≪ Formatear columnas datetime a "%Y%m%d" para JSON ≫=====
    for col in df_pred_window.select_dtypes(include=['datetime64']).columns:
        df_pred_window[col] = df_pred_window[col].dt.strftime('%Y%m%d')
    if 'h_fec_reg' in df_pred_window.columns:
        df_pred_window['h_fec_reg'] = (
            pd.to_datetime(df_pred_window['h_fec_reg'], errors='coerce')
            .dt.strftime('%Y%m%d')
        )

    # Reemplazar inf/-inf → NaN → None
    df_pred_window = df_pred_window.replace([np.inf, -np.inf], np.nan)
    df_pred_window = df_pred_window.where(pd.notnull(df_pred_window), None)

    # --------------------------------------
    # 6) Llamada a /predict con la ventana hasta ed
    # --------------------------------------
    try:
        payload = {"data": df_pred_window.to_dict(orient='records')}
        resp = requests.post(f"{api_base_url}/predict", json=payload)
        resp.raise_for_status()

        resp_json = resp.json()
        lista_predicciones = resp_json["predictions"]
        df_pred = pd.DataFrame(lista_predicciones)
        df_pred['ds'] = pd.to_datetime(df_pred['ds'])
        df_pred.set_index('ds', inplace=True)

        print(f"[DEBUG /predict] {len(df_pred)} predicciones obtenidas.")
    except requests.exceptions.HTTPError as http_err:
        if resp.status_code == 422:
            print("[ERROR /predict] Código 422 – detalles del error de validación:")
            print(resp.json())
        else:
            print(f"[ERROR /predict] HTTP {resp.status_code} – {resp.text}")
        df_pred = pd.DataFrame(columns=['yhat', 'yhat_lower', 'yhat_upper'])
    except Exception as e:
        print("[ERROR /predict] No se pudo obtener predicción:", str(e))
        df_pred = pd.DataFrame(columns=['yhat', 'yhat_lower', 'yhat_upper'])
        
    # --------------------------------------
    # 7) Agrupar pronóstico según freq y superponer curva
    # --------------------------------------

    # — Preparar variables globales para agrupar predicción
    total_habs = 735
    adr_glob = kpis['global']['adr']

    # — Volumen histórico se añadió arriba (fig_vol).
    #    Ahora agrupamos el pronóstico:
    if not df_pred.empty:
        # Agrupar y sumar 'yhat' según la frecuencia seleccionada
        pred_vol_agg = df_pred['yhat'].resample(freq).sum()

        # Trazar volumen pronosticado en freq
        fig_vol.add_trace(go.Scatter(
            x=pred_vol_agg.index,
            y=pred_vol_agg.values,
            mode='lines',
            name='Volumen Pronosticado',
            line=dict(color=filters.PRIMARY_COLOR, dash='dash')
        ))
    fig_vol.update_layout(
        title='Volumen de Reservas (Histórico + Pronóstico)',
        xaxis_title='Fecha',
        yaxis_title='Volumen',
        template='plotly_white',
        plot_bgcolor=filters.BACKGROUND_COLOR,
        paper_bgcolor=filters.BACKGROUND_COLOR,
        font_color=filters.TEXT_COLOR,
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    # — Ocupación (histórico + pronóstico)
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(
        x=series_ocupacion.index, y=series_ocupacion.values,
        mode='lines',
        name='Ocupación Histórica',
        line=dict(color='lightgrey')
    ))
    if not df_pred.empty:
        # Agrupar volumen pronosticado según freq
        pred_vol_agg = df_pred['yhat'].resample(freq).sum()

        # Calcular días por período según freq
        if freq == "D":
            dias_periodo = 1
        elif freq == "W":
            dias_periodo = 7
        else:  # freq == "M"
            # Para mensual, extraemos días en mes desde el índice
            dias_periodo = pred_vol_agg.index.to_series().dt.days_in_month

        # Ocupación pronosticada (%) = (volumen_agrupado / (total_habs * días_periodo)) * 100
        ocup_pred = pred_vol_agg / (total_habs * dias_periodo) * 100

        fig_occ.add_trace(go.Scatter(
            x=ocup_pred.index,
            y=ocup_pred.values,
            mode='lines',
            name='Ocupación Pronosticada',
            line=dict(color=filters.PRIMARY_COLOR, dash='dash')
        ))
    fig_occ.update_layout(
        title='Tasa de Ocupación (Histórico + Pronóstico)',
        xaxis_title='Fecha',
        yaxis_title='Ocupación (%)',
        template='plotly_white',
        plot_bgcolor=filters.BACKGROUND_COLOR,
        paper_bgcolor=filters.BACKGROUND_COLOR,
        font_color=filters.TEXT_COLOR,
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    # — RevPAR (histórico + pronóstico)
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatter(
        x=series_revpar.index, y=series_revpar.values,
        mode='lines',
        name='RevPAR Histórico',
        line=dict(color='lightgrey')
    ))
    if not df_pred.empty:
        # Usar la misma agregación de volumen pronosticado
        pred_vol_agg = df_pred['yhat'].resample(freq).sum()

        # Reutilizar dias_periodo calculados arriba
        if freq == "D":
            dias_periodo = 1
        elif freq == "W":
            dias_periodo = 7
        else:
            dias_periodo = pred_vol_agg.index.to_series().dt.days_in_month

        # Ocupación pronosticada en fracción (sin %) para RevPAR
        ocup_frac = (pred_vol_agg / (total_habs * dias_periodo))

        # RevPAR pronosticado = ADR_global * ocup_frac
        rp_pred = adr_glob * ocup_frac

        fig_rp.add_trace(go.Scatter(
            x=rp_pred.index,
            y=rp_pred.values,
            mode='lines',
            name='RevPAR Pronosticado',
            line=dict(color=filters.PRIMARY_COLOR, dash='dash')
        ))
    fig_rp.update_layout(
        title='RevPAR (Histórico + Pronóstico)',
        xaxis_title='Fecha',
        yaxis_title='RevPAR (MXN)',
        template='plotly_white',
        plot_bgcolor=filters.BACKGROUND_COLOR,
        paper_bgcolor=filters.BACKGROUND_COLOR,
        font_color=filters.TEXT_COLOR,
        margin=dict(l=20, r=20, t=40, b=20),
        height=350
    )

    # 8) Retornar todas las figuras y el texto del label-fechas
    print(f"[DEBUG RETURN] texto_rango_label = {texto_rango_label}")
    return (
        fig_vol,
        fig_occ,
        fig_rp,
        fig_lead,
        fig_stay,
        fig_cancel,
        fig_ind_adr,
        fig_ind_revpar,
        fig_ind_ocup,
        fig_ind_estancia,
        texto_rango_label
    )

if __name__ == '__main__':
    app.run(debug=True)
