
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from dash import Dash, html, dcc, Input, Output, State, dash_table
from dash.dcc import send_data_frame

# Simulación de datos
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

# Función red
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

# Aplicación Dash
app = Dash(__name__)
app.title = "Dashboard Inversión MOP"

app.layout = html.Div([
    html.H1("Dashboard de Inversión MOP", style={'textAlign': 'center'}),
    html.Div([
        dcc.Dropdown(id='año-selector', options=[{'label': a, 'value': a} for a in sorted(data['Año'].unique())],
                     value=2023, clearable=False, style={'width': '200px', 'marginRight': '20px'}),
        dcc.Dropdown(id='region-selector', options=[{'label': r, 'value': r} for r in sorted(data['Región'].unique())],
                     placeholder='Filtrar por región...', style={'width': '300px'})
    ], style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '20px'}),
    dcc.Graph(id='barras-tipo-obra'),
    dcc.Graph(id='red-regiones-obras'),
    html.Div([
        html.Button("Descargar CSV", id="download-button", n_clicks=0),
        dcc.Download(id="download-data")
    ], style={'textAlign': 'center', 'marginBottom': '20px'}),
    dash_table.DataTable(
        id='data-table',
        columns=[{"name": i, "id": i} for i in data.columns],
        data=data.to_dict('records'),
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'left'}
    )
])

@app.callback(
    Output('barras-tipo-obra', 'figure'),
    Output('red-regiones-obras', 'figure'),
    Output('data-table', 'data'),
    Input('año-selector', 'value'),
    Input('region-selector', 'value')
)
def update_visuals(año, region=None):
    df_year = data[data['Año'] == año]
    if region:
        df_year = df_year[df_year['Región'] == region]

    barras = px.bar(df_year.groupby('Tipo de Obra').sum().reset_index(),
                    x='Tipo de Obra', y='Monto (MM$)',
                    title=f"Inversión por Tipo de Obra - {año}")

    G, pos, nodes_x, nodes_y, edge_x, edge_y = construir_red(df_year)
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                            hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=nodes_x, y=nodes_y, mode='markers+text',
                            text=[node for node in G.nodes],
                            textposition='top center',
                            marker=dict(size=10, color='lightblue'))

    fig_network = go.Figure(data=[edge_trace, node_trace])
    fig_network.update_layout(title='Red de Regiones y Obras', showlegend=False)

    return barras, fig_network, df_year.to_dict('records')

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

if __name__ == "__main__":
    app.run_server(debug=True)
