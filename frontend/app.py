import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
import utils.data_extractor as de      # módulo refactorizado para carga y filtrado de datos
import components.filters as filters   # módulo de filtros y funciones de graficación
import datetime as dt
import plotly.express as px
import pandas as pd

# Cargar datos al iniciar la aplicación
data = de.load_data()
df_reservas = data['reservaciones']
df_empresas = data['empresas']
df_canales = data['canales']
df_agencias = data['agencias']

# Inicializar aplicación Dash con Bootstrap
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Dashboard KPI Hoteleros'

# ---------------------------------------------------
# HEADER FIJO CON LOGO Y TÍTULO PEQUEÑOS, FILTROS DEBAJO
# ---------------------------------------------------
header = html.Div(
    style={
        'position': 'fixed',
        'top': '0',
        'left': '0',
        'right': '0',
        'zIndex': '999',
        'backgroundColor': '#262626',
        'boxSizing': 'border-box',
        'padding': '10px 40px 5px 40px'
    },
    children=[
        # Primera fila: logo + título
        html.Div(
            style={'position': 'relative', 'width': '100%', 'height': '40px', 'marginTop': '10px'},
            children=[
                html.Img(
                    src=app.get_asset_url('tca_logo.png'),
                    style={
                        'position': 'absolute',
                        'left': '0px',
                        'height': '50px'
                    },
                    alt='Logo TCA'
                ),
                html.H4(
                    'Dashboard de Reservas Hoteleras',
                    style={
                        'position': 'absolute',
                        'left': '50%',
                        'transform': 'translateX(-50%)',
                        'color': 'white',
                        'margin': '0',
                        'fontSize': '32px'
                    }
                )
            ]
        ),
        # Segunda fila: filtros (ocupan todo el ancho)
        html.Div(
            style={
                'marginTop': '30px',
                'paddingBottom': '0px',
                'width': '100%',
                'backgroundColor': '#262626'
            },
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
# LAYOUT PRINCIPAL
# ---------------------------------------------------
app.layout = html.Div(
    style={'margin': '0', 'padding': '0'},
    children=[
        # Header fijo (logo + título + filtros)
        header,

        # Contenedor principal con paddingTop suficiente para dejar espacio al header
        dbc.Container(
            fluid=True,
            style={'paddingTop': '160px'},
            children=[
                # Separador opcional (queda debajo del header fijo)
                html.Hr(style={'borderColor': '#444444', 'margin': '0'}),

                # KPI Cards
                dbc.Row([
                    dbc.Col(dcc.Graph(id='indicador-adr',       config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-revpar',    config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-ocupacion', config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-estancia',  config={'displayModeBar': False}), xs=12, sm=6, md=3),
                ], className='mb-4 mt-4'),

                # Serie temporal de Volumen (ocupa ancho completo)
                dbc.Row([
                    dbc.Col(
                        dcc.Graph(id='grafico-volumen', config={'responsive': True}),
                        xs=12, md=12
                    )
                ], className='mb-4'),

                # Series temporales restantes: Ocupación y RevPAR
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-ocupacion', config={'responsive': True}), xs=12, md=6),
                    dbc.Col(dcc.Graph(id='grafico-revpar',    config={'responsive': True}), xs=12, md=6),
                ], className='mb-4'),

                # Distribuciones: Leadtime, Estancia y Cancel/No-Show
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
# CALLBACK PRINCIPAL
# ---------------------------------------------------
@app.callback(
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
    Input('filter-fechas-slider', 'value'),
    Input('filter-empresa',       'value'),
    Input('filter-canal',         'value'),
    Input('filter-agencia',       'value'),
    Input('filter-freq',          'value'),
)
def actualizar_dashboard(fecha_offset, empresas, canales, agencias, freq):
    # 1) Reconstruir fechas reales (sd, ed)
    fecha_base = dt.date(2019, 1, 1)
    offset_inicio, offset_fin = fecha_offset
    sd = fecha_base + dt.timedelta(days=offset_inicio)
    ed = fecha_base + dt.timedelta(days=offset_fin)
    texto_rango_label = f"Rango de fechas: Del {sd:%Y-%m-%d} al {ed:%Y-%m-%d}"

    # 2) Filtrar datos y calcular KPIs
    df_fil = de.filtrar_datos(
        df_reservas,
        sd, ed,
        empresas or None,
        canales  or None,
        agencias or None
    )
    print(f"[DEBUG callback] Rango: {sd} a {ed} | Filtrado result: {len(df_fil)} filas")
    kpis = de.calcular_kpis(df_fil, freq, total_habs=735)

    # 3) RECORTE de series temporales (idéntico a antes)…
    sd_ts = pd.to_datetime(sd)
    ed_ts = pd.to_datetime(ed)

    volumen = kpis['volumen']
    volumen = volumen[(volumen.index >= sd_ts) & (volumen.index <= ed_ts)]

    ocupacion = kpis['ocupacion']
    ocupacion = ocupacion[(ocupacion.index >= sd_ts) & (ocupacion.index <= ed_ts)]

    revpar = kpis['revpar']
    revpar = revpar[(revpar.index >= sd_ts) & (revpar.index <= ed_ts)]

    # 4) Gráficas temporales…
    fig_vol = filters.grafica_linea(
        volumen.index,
        volumen,
        titulo='Volumen de Reservas',
        eje_y='Reservas'
    )
    fig_occ = filters.grafica_linea(
        ocupacion.index,
        ocupacion,
        titulo='Tasa de Ocupación',
        eje_y='Porcentaje',
        formato_y='pct'
    )
    fig_rp = filters.grafica_linea(
        revpar.index,
        revpar,
        titulo='RevPAR',
        eje_y='RevPAR (MXN)',
        formato_y='money'
    )

    # 5) Distribuciones: Leadtime
    lead = ((df_fil['h_fec_lld'] - df_fil['h_res_fec']).dt.days
            if not df_fil.empty else [])
    fig_lead = filters.grafica_histograma(
        lead,
        titulo='Anticipación de Reserva',
        xaxis_title='Días de anticipación'
    )

    # 6) DISTRIBUCIÓN: Duración de Estancia (Boxplot)
    if not df_fil.empty:
        # a) Reproducir el mismo filtrado interno de calcular_kpis sobre h_num_noc,
        #    h_tot_hab y h_tfa_total:
        df_aux = df_fil.dropna(subset=["h_num_noc", "h_tot_hab", "h_tfa_total"]).copy()
        df_aux["h_num_noc"] = df_aux["h_num_noc"].astype(int)
        df_aux = df_aux[df_aux["h_num_noc"] > 0]
        series_estancia = df_aux["h_num_noc"].astype(float)
    else:
        series_estancia = []

    # Imprimir debug para cotejar:
    print(f"[DEBUG Avg Stay] KPI global devuelve avg_stay = {kpis['global']['avg_stay']:.2f}")
    if hasattr(series_estancia, "mean"):
        print(f"[DEBUG Avg Stay] Media en series_estancia = {series_estancia.mean():.2f}")
    else:
        print("[DEBUG Avg Stay] series_estancia vacío → media = 0.00")

    fig_stay = filters.grafica_boxplot(
        series_estancia,
        titulo='Duración de Estancia',
        yaxis_title='Noches por reserva'
    )

    # 7) Tasa de Cancelación y No-Show
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

    # 8) KPI Cards
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

    # 9) Retornar todo
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
