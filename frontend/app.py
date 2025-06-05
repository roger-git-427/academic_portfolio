import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output
import utils.data_extractor as de  # módulo refactorizado para carga y filtrado de datos
import components.filters as filters  # módulo de filtros y funciones de graficación
import datetime as dt
import plotly.express as px
import pandas as pd

# Cargar datos al iniciar la aplicación
data = de.load_data()
df_reservas = data['reservaciones']
print(f"[DEBUG MAIN] Reservas cargadas: {len(df_reservas)} filas")
df_empresas = data['empresas']
print(f"[DEBUG MAIN] Empresas cargadas: {len(df_empresas)} filas")
df_canales = data['canales']
print(f"[DEBUG MAIN] Canales cargados: {len(df_canales)} filas")
df_agencias = data['agencias']
print(f"[DEBUG MAIN] Agencias cargadas: {len(df_agencias)} filas")

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
        'padding': '10px 40px 5px 40px'  # menos padding para que quede compacto
    },
    children=[
        # Primera fila: logo + título
        html.Div(
            style={'position': 'relative', 'width': '100%', 'height': '40px', 'marginTop': '10px',},
            children=[
                html.Img(
                    src=app.get_asset_url('tca_logo.png'),
                    style={
                        'position': 'absolute',
                        'left': '0px',       # separa del borde izquierdo
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
            style={'paddingTop': '160px'},  # Ajustar si el header cambia de altura
            children=[
                # Separador opcional (queda debajo del header fijo)
                html.Hr(style={'borderColor': '#444444', 'margin': '0'}),

                # KPI Cards
                dbc.Row([
                    dbc.Col(dcc.Graph(id='indicador-adr', config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-revpar', config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-ocupacion', config={'displayModeBar': False}), xs=12, sm=6, md=3),
                    dbc.Col(dcc.Graph(id='indicador-estancia', config={'displayModeBar': False}), xs=12, sm=6, md=3),
                ], className='mb-4 mt-4'),

                # Series temporales: dos gráficos por fila
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-volumen', config={'responsive': True}), xs=12, md=6),
                    dbc.Col(dcc.Graph(id='grafico-roomnights', config={'responsive': True}), xs=12, md=6),
                ], className='mb-4'),
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-ocupacion', config={'responsive': True}), xs=12, md=6),
                    dbc.Col(dcc.Graph(id='grafico-revpar', config={'responsive': True}), xs=12, md=6),
                ], className='mb-4'),

                # Distribuciones: tres gráficos en una fila
                dbc.Row([
                    dbc.Col(dcc.Graph(id='grafico-leadtime', config={'responsive': True}), xs=12, md=4),
                    dbc.Col(dcc.Graph(id='grafico-estancia', config={'responsive': True}), xs=12, md=4),
                    dbc.Col(dcc.Graph(id='grafico-cancel', config={'responsive': True}), xs=12, md=4),
                ], className='mb-4'),
            ]
        )
    ]
)

# Callback para actualizar datos y gráficas (ahora también actualiza label-fechas)
@app.callback(
    Output('grafico-volumen', 'figure'),
    Output('grafico-roomnights', 'figure'),
    Output('grafico-ocupacion', 'figure'),
    Output('grafico-revpar', 'figure'),
    Output('grafico-leadtime', 'figure'),
    Output('grafico-estancia', 'figure'),
    Output('grafico-cancel', 'figure'),
    Output('indicador-adr', 'figure'),
    Output('indicador-revpar', 'figure'),
    Output('indicador-ocupacion', 'figure'),
    Output('indicador-estancia', 'figure'),
    Output('label-fechas', 'children'),      
    Input('filter-fechas-slider', 'value'),
    Input('filter-empresa', 'value'),
    Input('filter-canal', 'value'),
    Input('filter-agencia', 'value'),
    Input('filter-freq', 'value'),
)
def actualizar_dashboard(fecha_offset, empresas, canales, agencias, freq):
    """
    fecha_offset: [offset_inicio, offset_fin] en días desde 2019-01-01.
    """
    # 1) Reconstruir fechas reales
    fecha_base = dt.date(2019, 1, 1)
    offset_inicio, offset_fin = fecha_offset
    sd = fecha_base + dt.timedelta(days=offset_inicio)
    ed = fecha_base + dt.timedelta(days=offset_fin)

    # 2) Texto que irá en label-fechas (justo encima del slider)
    texto_rango_label = f"Rango de fechas: Del {sd:%Y-%m-%d} al {ed:%Y-%m-%d}"

    # 3) Filtrar datos y calcular KPIs
    df_fil = de.filtrar_datos(df_reservas, sd, ed, empresas or None, canales or None, agencias or None)
    print(f"[DEBUG callback] Rango: {sd} a {ed} | Filtrado result: {len(df_fil)} filas")
    kpis = de.calcular_kpis(df_fil, freq, total_habs=735)

    # 4) Crear gráficas (igual que antes)
    fig_vol = filters.grafica_linea(kpis['volumen'].index, kpis['volumen'],
                                    titulo='Volumen de Reservas', eje_y='Reservas')
    fig_rn  = filters.grafica_linea(kpis['room_nights'].index, kpis['room_nights'],
                                    titulo='Noches de Habitación Vendidas', eje_y='Noches de Habitación')
    fig_occ = filters.grafica_linea(kpis['ocupacion'].index, kpis['ocupacion'],
                                    titulo='Tasa de Ocupación', eje_y='Porcentaje', formato_y='pct')
    fig_rp  = filters.grafica_linea(kpis['revpar'].index, kpis['revpar'],
                                    titulo='RevPAR', eje_y='RevPAR (MXN)', formato_y='money')

    lead = ((df_fil['h_fec_lld'] - df_fil['h_res_fec']).dt.days
            if not df_fil.empty else [])
    fig_lead = filters.grafica_histograma(lead, titulo='Anticipación de Reserva',
                                          xaxis_title='Días de anticipación')
    fig_stay = filters.grafica_boxplot(df_fil['h_num_noc'] if not df_fil.empty else [],
                                       titulo='Duración de Estancia', yaxis_title='Noches por reserva')
    
    tasas = {'Cancelación': kpis['global']['tasa_cancel'], 'No Show': kpis['global']['tasa_noshow']}
    fig_cancel = px.bar(x=list(tasas.keys()), y=list(tasas.values()),
                        title='Tasa de Cancelación y No-Show',
                        color_discrete_sequence=[filters.PRIMARY_COLOR])
    fig_cancel.update_layout(
        yaxis_range=[0,30], template='plotly_white',
       yaxis_title='Tasa (%)',
        plot_bgcolor=filters.BACKGROUND_COLOR,
        paper_bgcolor=filters.BACKGROUND_COLOR,
        font_color=filters.TEXT_COLOR,
        margin=dict(l=20, r=20, t=30, b=20), height=350
    )

    fig_ind_adr      = filters.grafica_indicador(kpis['global']['adr'],      titulo='ADR',sufijo=' MXN')
    fig_ind_revpar   = filters.grafica_indicador(kpis['global']['revpar'],   titulo='RevPAR', sufijo=' MNX')
    fig_ind_ocup     = filters.grafica_indicador(kpis['global']['ocupacion'], titulo='Ocupación', sufijo='%')
    fig_ind_estancia = filters.grafica_indicador(kpis['global']['avg_stay'],  titulo='Estancia Prom.', sufijo=' noches')

    # 5) Retornar todas las figuras y, al final, el texto para label-fechas
    return (
        fig_vol, fig_rn, fig_occ, fig_rp,
        fig_lead, fig_stay, fig_cancel,
        fig_ind_adr, fig_ind_revpar, fig_ind_ocup, fig_ind_estancia,
        texto_rango_label
    )


if __name__ == '__main__':
    app.run(debug=True)
