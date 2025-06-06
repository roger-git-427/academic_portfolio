import dash_bootstrap_components as dbc
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import datetime as dt
import numpy as np

# Colores institucionales
PRIMARY_COLOR    = "#920F0F"  # Rojo institucional
BACKGROUND_COLOR = "white"
TEXT_COLOR       = "black"


def crear_controles(
    df_reservas: pd.DataFrame,
    df_empresas:  pd.DataFrame,
    df_canales:   pd.DataFrame,
    df_agencias:  pd.DataFrame
) -> dbc.Row:
    """
    Devuelve un dbc.Row con los controles de filtrado:
    - RangeSlider de fechas (tick-labels = años), partiendo desde 2019-01-01 hasta la fecha máxima.
    - Dropdowns para empresa, canal y agencia.
    - RadioItems para frecuencia.
    """
    # --------------------------------------------------
    # FIJAR FECHA INICIAL EN 2019-01-01
    # --------------------------------------------------
    fecha_base = dt.date(2019, 1, 1)

    print("▶️ [DEBUG crear_controles] dtype de fecha_checkin:", df_reservas["fecha_checkin"].dtype)
    if not df_reservas["fecha_checkin"].empty:
        print("   [DEBUG crear_controles] Primeras fechas:", df_reservas["fecha_checkin"].head(3).tolist())
        print("   [DEBUG crear_controles] Últimas fechas:", df_reservas["fecha_checkin"].tail(3).tolist())
    else:
        print("   [DEBUG crear_controles] df_reservas['fecha_checkin'] está vacío")

    
    # Obtener la fecha máxima real de reservas
    max_date = df_reservas["fecha_checkin"].dt.date.max()
    print("▶️ [DEBUG crear_controles] Fecha máxima de reservas:", max_date)
    if max_date is None or max_date < fecha_base:
        max_date = fecha_base
    print("   [DEBUG crear_controles] max_date calculada:", max_date)
    # Cantidad total de días entre 2019-01-01 y max_date:
    total_days = (max_date - fecha_base).days

    # --------------------------------------------------
    # GENERAR MARKS con “1 de enero de cada año”
    # --------------------------------------------------
    marks = {}
    año = fecha_base.year
    while True:
        fecha_año = dt.date(año, 1, 1)
        if fecha_año > max_date:
            break
        offset = (fecha_año - fecha_base).days
        marks[offset] = str(año)
        año += 1

    # Asegurar que el último año (del max_date) quede marcado:
    último_inicio_año = dt.date(max_date.year, 1, 1)
    offset_último = (último_inicio_año - fecha_base).days
    if offset_último <= total_days:
        marks[offset_último] = str(max_date.year)

    # --------------------------------------------------
    # CONSTRUIR LA FILA DE CONTROLES
    # --------------------------------------------------
    return dbc.Row(
        [
            # ------------------------------
            # COLUMNA 1: RangeSlider de Fechas (tick-labels = años)
            # ------------------------------
            dbc.Col(
                html.Div(
                    [
                        # Este contenedor tendrá el texto dinámico "Rango de fechas: Del X al Z"
                        html.Div(
                            id="label-fechas-filtro",
                            style={
                                "color": 'white',
                                "fontWeight": "500",
                                "marginBottom": "5px",
                                "textAlign": "center",
                                "fontSize": "16px"
                            },
                            children=""  # se llenará dinámicamente desde el callback
                        ),
                        dcc.RangeSlider(
                            id="filter-fechas-slider",
                            min=0,
                            max=total_days,
                            value=[0, total_days],
                            marks=marks,
                            tooltip={
                                "placement": "bottom",
                                "always_visible": False
                            },
                            allowCross=False,
                            pushable=1,
                            updatemode="mouseup",
                            className="mt-1 mb-3",
                            vertical=False,
                        ),
                        # Ya no requerimos html.Div para mostrar fechas, porque lo haremos en 'label-fechas-filtro'
                    ]
                ),
                xs=12,
                sm=12,
                md=6,
                lg=4,
            ),

            # ------------------------------
            # COLUMNA 2: Dropdown Empresa
            # ------------------------------
            dbc.Col(
                dcc.Dropdown(
                    id="filter-empresa",
                    options=[
                        {"label": nombre, "value": eid}
                        for eid, nombre in zip(
                            df_empresas["ID_empresa"], df_empresas["Empresa_nombre"]
                        )
                    ],
                    multi=True,
                    placeholder="Empresa",
                    style={"width": "100%", "marginTop": "10px"},
                ),
                xs=12,
                sm=6,
                md=3,
                lg=2,
            ),

            # ------------------------------
            # COLUMNA 3: Dropdown Canal
            # ------------------------------
            dbc.Col(
                dcc.Dropdown(
                    id="filter-canal",
                    options=[
                        {"label": nombre, "value": cid}
                        for cid, nombre in zip(
                            df_canales["ID_canal"], df_canales["Canal_nombre"]
                        )
                    ],
                    multi=True,
                    placeholder="Canal",
                    style={"width": "100%", "marginTop": "10px"},
                ),
                xs=12,
                sm=6,
                md=3,
                lg=2,
            ),

            # ------------------------------
            # COLUMNA 4: Dropdown Agencia
            # ------------------------------
            dbc.Col(
                dcc.Dropdown(
                    id="filter-agencia",
                    options=[
                        {"label": nombre, "value": aid}
                        for aid, nombre in zip(
                            df_agencias["ID_Agencia"], df_agencias["Agencia_nombre"]
                        )
                    ],
                    multi=True,
                    placeholder="Agencia",
                    style={"width": "100%", "marginTop": "10px"},
                ),
                xs=12,
                sm=6,
                md=3,
                lg=2,
            ),

            # ------------------------------
            # COLUMNA 5: RadioItems Frecuencia
            # ------------------------------
            dbc.Col(
                html.Div(
                    [
                        html.Label(
                            "Frecuencia:",
                            style={
                                "color": 'white',
                                "fontWeight": "500",
                                "marginBottom": "5px",
                            },
                        ),
                        dbc.RadioItems(
                            id="filter-freq",
                            options=[
                                {"label": "Diario", "value": "D"},
                                {"label": "Semanal", "value": "W"},
                                {"label": "Mensual", "value": "M"},
                            ],
                            value="D",
                            inline=True,
                            inputCheckedClassName="text-primary",
                        ),
                    ],
                    style={"textAlign": "center", "width": "100%", "color": 'white', 'marginTop': '-20px'},
                ),
                xs=12,
                sm=6,
                md=3,
                lg=2,
            ),
        ],
        className="gx-2 gy-2 mb-0",
    )


def grafica_linea(x, y, titulo="", eje_y="", formato_y=None) -> go.Figure:
    """Devuelve una figura de línea con fondo blanco y acento rojo."""
    # --- 1) Convertir x (index) a objetos datetime o a lista de valores ---
    if isinstance(x, pd.PeriodIndex):
        x_plot = x.to_timestamp()
    else:
        x_plot = [
            val.to_timestamp() if hasattr(val, "to_timestamp") else val
            for val in x
        ]
    # --- 2) Convertir y (serie) a lista de valores ---
    y_plot = y.tolist() if hasattr(y, "tolist") else list(y)

    # --- 3) Detectar listas vacías y devolver figura “vacía” ---
    if len(x_plot) == 0 or len(y_plot) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor=BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font_color=TEXT_COLOR,
            font_family="Roboto",
            title=titulo,
            yaxis_title=eje_y,
            xaxis_title="Fecha",
            margin=dict(l=20, r=20, t=30, b=20),
            height=350,
        )
        return fig

    # --- 4) Si hay datos, dibujar normalmente con px.line ---
    fig = px.line(
        x=x_plot,
        y=y_plot,
        title=titulo,
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    fig.update_traces(mode="lines+markers")
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        font_family="Roboto",
        yaxis_title=eje_y,
        xaxis_title="Fecha",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    if formato_y == "pct":
        fig.update_layout(yaxis_tickformat=".1f%")
    elif formato_y == "money":
        fig.update_layout(yaxis_tickprefix="$")
    return fig


def grafica_indicador(valor, titulo="", sufijo="") -> go.Figure:
    """
    Devuelve un indicador KPI en tarjeta con:
      - Número en PRIMARY_COLOR,
      - Título y número centrados,
      - Un marco (borde) alrededor de todo el indicador.
    """
    # 1) Construimos la figura Indicator. 
    #    - mode="number": el título se maneja por separado con title={"text":...}
    #    - En title={...} solo ponemos text, font y align="center"
    #    - Ajustamos el domain vertical para que haya espacio arriba y abajo
    fig = go.Figure(
        go.Indicator(
            mode="number",  
            value=valor,
            title={
                "text": titulo,
                "font": {"size": 18, "color": TEXT_COLOR},
                "align": "center"    # Centrado horizontal del texto
            },
            number={
                "suffix": sufijo,
                "font": {"size": 40, "color": PRIMARY_COLOR}
            },
            # domain en X=0–1, en Y dejamos margen arriba y abajo para centrar
            domain={"x": [0, 1], "y": [0.15, 0.7]},
        )
    )

    # 2) Dibujamos un rectángulo en coordenadas "paper" para simular un borde
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        font_family="Roboto",
        height=120,
        margin=dict(l=10, r=10, t=10, b=10),
        shapes=[
            {
                "type": "rect",
                "x0": 0,
                "y0": 0,
                "x1": 1,
                "y1": 1,
                "xref": "paper",
                "yref": "paper",
                "line": {"color": PRIMARY_COLOR, "width": 2},
                "fillcolor": "rgba(151, 41, 41, 0.05)",  # transparente
                "layer": "below"
            }
        ]
    )

    return fig


def grafica_histograma(data, titulo="", xaxis_title="") -> go.Figure:
    """Devuelve un histograma compacto."""
    fig = px.histogram(data, nbins=30, title=titulo, color_discrete_sequence=[PRIMARY_COLOR])
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,
        font_family="Roboto", 
        xaxis_title=xaxis_title,
        yaxis_title="Cantidad de reservas",
        margin=dict(l=20, r=20, t=30, b=20),
        height=350,
    )
    return fig


def grafica_boxplot(data, titulo="", yaxis_title="") -> go.Figure:
    """
    Devuelve un diagrama de caja con la distribución de 'data'
    (lista o pandas.Series de número de noches), y marca la media con una línea punteada.
    """
    # 1) Extraer valores numéricos reales
    # Si 'data' es un pandas.Series, tomamos solo valores no nulos como floats
    if hasattr(data, "dropna"):
        valores = data.dropna().astype(float).values.tolist()
    else:
        # Si 'data' ya es una lista/iterable, convertimos cada elemento a float
        valores = [float(x) for x in data if x is not None]

    # 2) Calcular media
    if len(valores) > 0:
        media = float(np.mean(valores))
    else:
        media = 0.0

    # 3) Crear boxplot forzando a Plotly a usar la lista de valores
    fig = px.box(
        y=valores,
        title=titulo,
        color_discrete_sequence=[PRIMARY_COLOR]  # Asegúrate de haber importado o definido PRIMARY_COLOR
    )

    # 4) Ajustar rango del eje Y (puedes modificar estos límites según tu caso)
    fig.update_yaxes(range=[-1, 10])

    # 5) Agregar una línea horizontal punteada para marcar la media
    fig.add_hline(
        y=media,
        line_dash="dash",
        line_color="royalblue",
        annotation_text=f"Media: {media:.2f}",
        annotation_position="top left",
        annotation_font_color="royalblue"
    )

    # 6) Ajustes de estilo de layout
    fig.update_layout(
        template="plotly_white",
        plot_bgcolor=BACKGROUND_COLOR,  # Asegúrate de haber importado o definido BACKGROUND_COLOR
        paper_bgcolor=BACKGROUND_COLOR,
        font_color=TEXT_COLOR,          # Asegúrate de haber importado o definido TEXT_COLOR
        font_family="Roboto",
        yaxis_title=yaxis_title,
        margin=dict(l=20, r=20, t=50, b=20),  # Se deja más espacio en 't' para la anotación
        height=350,
    )

    return fig