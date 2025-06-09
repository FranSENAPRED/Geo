# ✅ DASHBOARD DE INVERSIÓN MOP CON LOGIN Y GRAFICOS
from pyngrok import ngrok
ngrok.set_auth_token("2xpwztU97cmSHsa6AvipIg8dNQC_3rWfvDBukiemHoxopVANF")

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dcc import send_data_frame
import threading

# Simulación de datos del Ministerio de Obras Públicas de Chile (MOP)
np.random.seed(42)
regiones = ['Arica y Parinacota', 'Tarapacá', 'Antofagasta', 'Atacama', 'Coquimbo', 'Valparaíso',
            'Metropolitana', 'O’Higgins', 'Maule', 'Ñuble', 'Biobío', 'Araucanía', 'Los Ríos', 'Los Lagos',
            'Aysén', 'Magallanes']
tipos = ['Vialidad', 'Agua Potable', 'Edificación', 'Puentes', 'Infraestructura Portuaria']

data = pd.DataFrame({
    'Región': np.random.choice(regiones, 500),
    'Año': np.random.randint(2015, 2024, 500),
    'Tipo de Obra': np.random.choice(tipos, 500),
    'Monto (MM$)': np.random.randint(100, 10000, 500)
})

def construir_red(df):
    edges = df.groupby(['Región', 'Tipo de Obra']).size().reset_index(name='count')
    G = nx.from_pandas_edgelist(edges, 'Región', 'Tipo de Obra', ['count'])
    pos = nx.spring_layout(G, seed=42)
    nodes_x = [pos[node][0] for node in G.nodes]
    nodes_y = [pos[node][1] for node in G.nodes]
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    return G, pos, nodes_x, nodes_y, edge_x, edge_y

app = Dash(__name__)
app.title = "Dashboard Inversión MOP"

app.layout = html.Div([
    dcc.Store(id='auth-status', data=False),
    html.Div(id='login-div', children=[
        html.H2("Iniciar sesión"),
        dcc.Input(id='username', type='text', placeholder='Usuario'),
        dcc.Input(id='password', type='password', placeholder='Contraseña'),
        html.Button("Ingresar", id='login-button', n_clicks=0),
        html.Div(id='login-output', style={'color': 'red', 'marginTop': '10px'})
    ], style={'textAlign': 'center', 'marginTop': '100px'}),

    html.Div(id='dashboard-div', children=[
        html.H1("Dashboard de Inversión MOP", style={'textAlign': 'center'}),
        html.P("Aquí podrá explorar los datos filtrando por año y por región. Los gráficos se actualizarán automáticamente según su selección.",
               style={'textAlign': 'center', 'fontStyle': 'italic'}),
        html.Div([
            dcc.Dropdown(id='año-selector', options=[{'label': a, 'value': a} for a in sorted(data['Año'].unique())],
                         value=2023, clearable=False, style={'width': '200px', 'marginRight': '20px'}),
            dcc.Dropdown(id='region-selector', options=[{'label': r, 'value': r} for r in sorted(data['Región'].unique())],
                         placeholder='Filtrar por región...', style={'width': '300px'})
        ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
        dcc.Graph(id='barras-tipo-obra', config={'displayModeBar': False}),
        dcc.Graph(id='red-regiones-obras', config={'displayModeBar': False}),
        dcc.Graph(id='linea-inversion-anual', config={'displayModeBar': False}),
        dcc.Graph(id='area-regiones', config={'displayModeBar': False}),
        dcc.Graph(id='mapa-coropletico', config={'displayModeBar': True, 'scrollZoom': True}),
        html.H3("Vista tabular", style={'textAlign': 'center'}),
        html.Div([
            html.Button("Descargar CSV", id="download-button", n_clicks=0),
            dcc.Download(id="download-data")
        ], style={'textAlign': 'center'}),
        dash_table.DataTable(
            id='data-table',
            columns=[{"name": i, "id": i} for i in data.columns],
            data=data.to_dict('records'),
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={'display': 'none', 'width': '80%', 'margin': 'auto'})
])

@app.callback(
    Output('auth-status', 'data'),
    Output('login-output', 'children'),
    Input('login-button', 'n_clicks'),
    State('username', 'value'),
    State('password', 'value'),
    prevent_initial_call=True
)
def verify_login(n, user, pwd):
    if user == "Cravello" and pwd == "Mop2025#":
        return True, ""
    return False, "Usuario o contraseña incorrectos."

@app.callback(
    Output('login-div', 'style'),
    Output('dashboard-div', 'style'),
    Input('auth-status', 'data')
)
def toggle_layout(authenticated):
    if authenticated:
        return {'display': 'none'}, {'display': 'block', 'width': '80%', 'margin': 'auto'}
    return {'textAlign': 'center', 'marginTop': '100px'}, {'display': 'none'}

@app.callback(
    Output('barras-tipo-obra', 'figure'),
    Output('red-regiones-obras', 'figure'),
    Output('linea-inversion-anual', 'figure'),
    Output('area-regiones', 'figure'),
    Output('mapa-coropletico', 'figure'),
    Output('data-table', 'data'),
    Input('año-selector', 'value'),
    Input('region-selector', 'value')
)
def update_visuals(año, region=None):
    df_year = data[data['Año'] == año]
    if region:
        df_year = df_year[df_year['Región'] == region]
    barras = px.bar(df_year.groupby('Tipo de Obra').sum().reset_index(),
                    x='Tipo de Obra', y='Monto (MM$)', text='Monto (MM$)',
                    title=f"Distribución de Inversión por Tipo de Obra - {año}")
    barras.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    barras.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    G, pos, nodes_x, nodes_y, edge_x, edge_y = construir_red(df_year)
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')
    node_sizes = [10 + G.degree(node)*2 for node in G.nodes]
    node_colors = ['skyblue' if node in df_year['Región'].values else 'lightgreen' for node in G.nodes]
    node_trace = go.Scatter(x=nodes_x, y=nodes_y, mode='markers',
                            marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color='darkblue')),
                            text=[f"{node} - conexiones: {G.degree(node)}" for node in G.nodes],
                            hoverinfo='text', showlegend=True)
    fig_network = go.Figure(data=[edge_trace, node_trace])
    fig_network.update_layout(title='Relación entre Regiones y Tipos de Obra (Red)', showlegend=False)
    df_line = data.copy()
    if region:
        df_line = df_line[df_line['Región'] == region]
    linea = px.line(df_line.groupby('Año').sum().reset_index(), x='Año', y='Monto (MM$)', markers=True,
                    title='Evolución de Inversión Total por Año')
    df_area = data.copy()
    df_area = df_area.groupby(['Año', 'Región'], as_index=False)['Monto (MM$)'].sum()
    area = px.area(df_area, x='Año', y='Monto (MM$)', color='Región',
                   title='Participación Regional en la Inversión Total')
    df_map = df_year.groupby('Región', as_index=False)['Monto (MM$)'].sum()
    codigos = {
        'Arica y Parinacota': 15, 'Tarapacá': 1, 'Antofagasta': 2, 'Atacama': 3, 'Coquimbo': 4, 'Valparaíso': 5,
        'Metropolitana': 13, "O’Higgins": 6, 'Maule': 7, 'Ñuble': 16, 'Biobío': 8, 'Araucanía': 9,
        'Los Ríos': 14, 'Los Lagos': 10, 'Aysén': 11, 'Magallanes': 12
    }
    df_map['codregion'] = df_map['Región'].map(codigos)
    mapa = px.choropleth(locations=df_map['codregion'],
                          locationmode='geojson-id',
                          geojson='https://raw.githubusercontent.com/caracena/chile-geojson/master/regiones.json',
                          featureidkey="properties.codregion",
                          color=df_map['Monto (MM$)"],
                          color_continuous_scale='Blues',
                          title='Inversión Total por Región')
    mapa.update_geos(fitbounds="locations", visible=True)
    return barras, fig_network, linea, area, mapa, df_year.to_dict('records')

@app.callback(
    Output("download-data", "data"),
    Input("download-button", "n_clicks"),
    State("año-selector", "value"),
    State("region-selector", "value"),
    prevent_initial_call=True
)
def exportar_csv(n_clicks, año, region):
    df_export = data[data['Año'] == año]
    if region:
        df_export = df_export[df_export['Región'] == region]
    return send_data_frame(df_export.to_csv, filename=f"inversion_{año}_{region or 'todas'}.csv", index=False)

# Ejecución
port = 8050
public_url = ngrok.connect(port)
print(f"Accede al dashboard: {public_url}")

def run():
    app.run(host='0.0.0.0', port=port)

thread = threading.Thread(target=run)
thread.start()